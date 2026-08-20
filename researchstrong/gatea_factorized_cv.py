from pathlib import Path
from collections import defaultdict, Counter
import json,tempfile,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import joint_schema_alignment as js
import gatea_source_selector as ss
from gatea_pairwise_cv import CachedEncoder

OUT=Path(__file__).with_name('gatea_factorized_cv_result.json')
N=100; JOINT=(0.75,0.0,2); CS=[0.03,0.1,0.25,0.5,1.0,2.0]

# Stage-1 classes.
OMIT=0; DEFAULT=1; MEMORY=2

def slot_stage_label(s):
    if not s['present']: return OMIT
    # A default operation is considered authoritative only when it exactly produces development gold.
    for c in s['cands']:
        if c['src']=='default' and base.norm(c['value'])==base.norm(s['gold']): return DEFAULT
    return MEMORY

def slot_features(s):
    cs=s['cands']; srcs=defaultdict(list)
    for c in cs: srcs[c['src']].append(c)
    def mx(src): return max([float(c['feat'][-7]) for c in srcs.get(src,[])] + [-1.0])
    # Base feature prefix is identical across all candidates; use it without any payload/value feature.
    basef=list(cs[0]['feat'][:13]) if cs else [0.0]*13
    return basef + [
        float(bool(srcs.get('default'))), float(bool(srcs.get('joint'))), float(bool(srcs.get('typed'))), float(bool(srcs.get('operator'))),
        mx('joint'), mx('typed'), mx('operator'),
        float(len(srcs.get('joint',[]))), float(len(srcs.get('typed',[]))), float(len(srcs.get('operator',[]))),
    ]

def build_rows(enc):
    td=Path(tempfile.gettempdir())/'mab_factorized_cv';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);rows=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});slots=[]
        for p,d0 in props.items():
            d=d0 or {};cands=ss.build_slot(enc,qa,ses,p,d,jp);s={'p':p,'d':d,'cands':cands,'present':p in gold,'gold':gold.get(p)};s['stage_y']=slot_stage_label(s);s['stage_x']=slot_features(s);slots.append(s)
        rows.append({'qi':qi,'qa':qa,'slots':slots})
        if qi%10==0:print('FACT_BUILD',qi,'cache',len(enc.cache),flush=True)
    return rows,missing

def pair_data(rows,ids):
    X=[];Y=[]
    for r in rows:
        if r['qi'] not in ids:continue
        for s in r['slots']:
            if s['stage_y']!=MEMORY:continue
            pos=[];neg=[]
            for i,c in enumerate(s['cands']):
                if c['src'] in ('omit','default'):continue
                ok=s['present'] and base.norm(c['value'])==base.norm(s['gold'])
                (pos if ok else neg).append(i)
            if not pos or not neg:continue
            for pi in pos[:4]:
                pf=np.asarray(s['cands'][pi]['feat'],float)
                for ni in neg[:40]:
                    d=pf-np.asarray(s['cands'][ni]['feat'],float);X.append(d);Y.append(1);X.append(-d);Y.append(0)
    return np.asarray(X,float),np.asarray(Y,int)

def fit(rows,ids,C):
    SX=[];SY=[]
    for r in rows:
        if r['qi'] not in ids:continue
        for s in r['slots']:SX.append(s['stage_x']);SY.append(s['stage_y'])
    # scikit-learn >=1.9 removed the multi_class constructor argument; default multiclass handling is used.
    stage=LogisticRegression(C=C,max_iter=4000,class_weight='balanced',random_state=42).fit(np.asarray(SX,float),np.asarray(SY,int))
    PX,PY=pair_data(rows,ids)
    rank=LogisticRegression(C=C,max_iter=5000,fit_intercept=False,random_state=42).fit(PX,PY) if len(PY) else None
    return stage,rank,len(SY),len(PY)

def choose_default(s):
    ds=[c for c in s['cands'] if c['src']=='default']
    if not ds:return None
    # Deterministic source priority then identity/coerce preference.
    pri={'schema_default':0,'description_default':1,'optional_empty_array':2,'optional_empty_string':2,'boolean_false_default':3,'offset_zero':3,'first_page_one':3,'latest_index_zero':3,'latest_zero':3}
    def key(c):
        detail=str(c.get('detail',''));return (pri.get(detail,9),0 if c['op'] in ('identity','coerce') else 1)
    return sorted(ds,key=key)[0]

def choose_memory(s,rank):
    ms=[c for c in s['cands'] if c['src'] not in ('omit','default')]
    if not ms:return None
    if rank is None:return ms[0]
    F=np.asarray([c['feat'] for c in ms],float);return ms[int(np.argmax(rank.decision_function(F)))]

def score(rows,ids,stage,rank):
    C=P=G=exact=tasks=0;lev=defaultdict(lambda:Counter(n=0,correct=0,pred=0));src=Counter();stage_conf=Counter()
    for r in rows:
        if r['qi'] not in ids:continue
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pred={};tasks+=1
        for s in r['slots']:
            y=int(stage.predict(np.asarray([s['stage_x']],float))[0]);stage_conf[str(y)]+=1;c=None
            if y==DEFAULT:c=choose_default(s)
            elif y==MEMORY:c=choose_memory(s,rank)
            if c is not None:pred[s['p']]=c['value'];src[c['src']+':'+c['op']]+=1
            lvl=str((gi.get(s['p']) or {}).get('type','unknown'));m=lev[lvl];m['n']+=1;m['pred']+=int(s['p'] in pred);m['correct']+=int(s['p'] in pred and s['p'] in gold and base.norm(pred[s['p']])==base.norm(gold[s['p']]))
        c0,p0,g0,_,e0=base.arg_metrics(pred,gold);C+=c0;P+=p0;G+=g0;exact+=e0
    pr=C/max(1,P);rc=C/max(1,G);f=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact_argument_set':exact/max(1,tasks),'levels':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in lev.items()},'sources':dict(src),'stage_predictions':dict(stage_conf)}

def oracle(rows):
    out=defaultdict(lambda:Counter(n=0,covered=0))
    for r in rows:
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for s in r['slots']:
            if s['p'] not in gold:continue
            lvl=str((gi.get(s['p']) or {}).get('type','unknown'));out[lvl]['n']+=1;out[lvl]['covered']+=int(any(c['src']!='omit' and base.norm(c['value'])==base.norm(gold[s['p']]) for c in s['cands']))
    return {k:{'n':v['n'],'coverage':v['covered']/max(1,v['n'])} for k,v in out.items()}

def main():
    enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'));rows,missing=build_rows(enc);idx=np.arange(N);kf=KFold(n_splits=5,shuffle=True,random_state=20260820);folds=[(set(idx[tr].tolist()),set(idx[va].tolist())) for tr,va in kf.split(idx)]
    grid=[]
    for Cval in CS:
        fs=[]
        for tr,va in folds:
            st,rk,n,pn=fit(rows,tr,Cval);fs.append(score(rows,va,st,rk))
        mf=float(np.mean([x['f1'] for x in fs]));grid.append((mf,Cval,fs));print('FACT_CV_C',Cval,mf,flush=True)
    meanF,Cbest,foldres=max(grid,key=lambda z:z[0]);st,rk,n,pn=fit(rows,set(range(N)),Cbest);full=score(rows,set(range(N)),st,rk)
    result={'stage':'Gate v2 factorized development CV: slot state (OMIT/DEFAULT/MEMORY) then pairwise memory operation ranker','split':'QA001-100 development only; QA101-400 labels remain sealed','candidate_oracle_by_grounding':oracle(rows),'grid':[{'C':c,'mean_f1':m,'folds':[{'f1':x['f1'],'precision':x['precision'],'recall':x['recall'],'levels':x['levels']} for x in fs]} for m,c,fs in grid],'selected_C':Cbest,'selected_mean_cv_f1':meanF,'all_dev_refit':full,'stage_training_slots':n,'pairwise_training_rows':pn,'embedding_cache_entries':len(enc.cache),'missing_zero_based':missing,'guardrail':'Stage labels are derived only from development gold presence and exact deterministic-default equality; grounding_info is reporting-only. Candidate values are executor payloads, not input features. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_GATEA_FACTORIZED_CV='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
