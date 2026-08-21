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

OUT=Path(__file__).with_name('gatea_value_aggregate_cv_result.json')
N=100; JOINT=(0.75,0.0,2); CS=[0.01,0.03,0.1,0.25,0.5,1.0,2.0]
MARGINS=[-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]

# Feature layout inherited from gatea_source_selector.cand_features.
BASE=13; NSRC=len(ss.SRC_NAMES); NOP=len(ss.OPS); NK=len(ss.KINDS); ND=len(ss.DSRC)
SRC0=BASE; OP0=SRC0+NSRC; KIND0=OP0+NOP; DSRC0=KIND0+NK


def aggregate_slot(cands):
    """Group executor candidates by exact normalized output value.
    Values remain payloads only; aggregate features are provenance/source/operator statistics.
    """
    groups=defaultdict(list); payload={}
    for c in cands:
        key='<OMIT>' if c['src']=='omit' else base.norm(c['value'])
        groups[key].append(c)
        if key not in payload: payload[key]=None if c['src']=='omit' else c['value']
    out=[]
    for key,cs in groups.items():
        F=np.asarray([c['feat'] for c in cs],float)
        # Slot/query/schema prefix is common across candidates.
        bf=F[0,:BASE].tolist()
        src_or=np.max(F[:,SRC0:SRC0+NSRC],axis=0).tolist()
        op_or=np.max(F[:,OP0:OP0+NOP],axis=0).tolist()
        kind_or=np.max(F[:,KIND0:KIND0+NK],axis=0).tolist()
        dsrc_or=np.max(F[:,DSRC0:DSRC0+ND],axis=0).tolist()
        # Tail in source selector: score, field-name sim, current, memory, rank, has_field, nonomit.
        score=F[:,-7]; fsim=F[:,-6]; cur=F[:,-5]; mem=F[:,-4]; rank=F[:,-3]; hasfield=F[:,-2]
        counts=[sum(1 for c in cs if c['src']==s) for s in ss.SRC_NAMES]
        feat=(bf+src_or+op_or+kind_or+dsrc_or+[
            float(np.max(score)),float(np.mean(score)),float(np.max(fsim)),float(np.mean(fsim)),
            float(np.max(cur)),float(np.max(mem)),float(np.min(rank)),float(np.max(rank)),float(np.max(hasfield)),
            float(len(cs)),float(len(set(c['src'] for c in cs))),float(len(set(c['op'] for c in cs))),
        ]+[float(x) for x in counts])
        out.append({'key':key,'value':payload[key],'omit':key=='<OMIT>','feat':feat,
                    'sources':sorted(set(c['src'] for c in cs)),'ops':sorted(set(c['op'] for c in cs)),'evidence_count':len(cs)})
    return out


def build_rows(enc):
    td=Path(tempfile.gettempdir())/'mab_value_agg_cv';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);rows=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {}
        gold=((qa.get('tool_call') or {}).get('arguments') or {});slots=[]
        for p,d0 in props.items():
            d=d0 or {};raw=ss.build_slot(enc,qa,ses,p,d,jp);vals=aggregate_slot(raw)
            for v in vals:
                v['label']=int((p not in gold and v['omit']) or (p in gold and not v['omit'] and v['key']==base.norm(gold[p])))
            slots.append({'p':p,'d':d,'vals':vals,'present':p in gold,'gold':gold.get(p)})
        rows.append({'qi':qi,'qa':qa,'slots':slots})
        if qi%10==0:print('VALUE_AGG_BUILD',qi,'cache',len(enc.cache),flush=True)
    return rows,missing


def pair_data(rows,ids):
    X=[];Y=[];groups=usable=0
    for r in rows:
        if r['qi'] not in ids:continue
        for s in r['slots']:
            pos=[v for v in s['vals'] if v['label']];neg=[v for v in s['vals'] if not v['label']]
            groups+=1
            if not pos or not neg:continue
            usable+=1
            # deterministic hard-negative ordering: highest evidence count first, then stable key
            neg=sorted(neg,key=lambda v:(-v['evidence_count'],v['key']))[:40]
            for pv in pos[:4]:
                pf=np.asarray(pv['feat'],float)
                for nv in neg:
                    d=pf-np.asarray(nv['feat'],float);X.append(d);Y.append(1);X.append(-d);Y.append(0)
    return np.asarray(X,float),np.asarray(Y,int),groups,usable


def fit(rows,ids,C):
    X,Y,g,u=pair_data(rows,ids)
    clf=LogisticRegression(C=C,max_iter=5000,fit_intercept=False,random_state=42).fit(X,Y)
    return clf,len(Y),g,u


def choose(s,clf,margin=0.0,force_present=None,oracle_payload=False):
    V=s['vals'];F=np.asarray([v['feat'] for v in V],float);z=np.asarray(clf.decision_function(F),float)
    omit_idx=next((i for i,v in enumerate(V) if v['omit']),None)
    non=[i for i,v in enumerate(V) if not v['omit']]
    if force_present is False:return V[omit_idx] if omit_idx is not None else None
    if not non:return V[omit_idx] if omit_idx is not None else None
    if force_present is True:
        if oracle_payload:
            hit=next((v for v in V if v['label'] and not v['omit']),None);return hit
        return V[max(non,key=lambda i:z[i])]
    best=max(non,key=lambda i:z[i]); oz=z[omit_idx] if omit_idx is not None else -1e9
    # Positive margin requires stronger evidence to emit a value.
    emit=(z[best]-oz)>=margin
    if not emit:return V[omit_idx] if omit_idx is not None else None
    if oracle_payload:
        hit=next((v for v in V if v['label'] and not v['omit']),None);return hit if hit is not None else V[best]
    return V[best]


def metrics(rows,ids,clf,margin=0.0,mode='normal'):
    C=P=G=exact=tasks=0;lev=defaultdict(lambda:Counter(n=0,correct=0,pred=0));src=Counter()
    for r in rows:
        if r['qi'] not in ids:continue
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pred={};tasks+=1
        for s in r['slots']:
            if mode=='oracle_presence':v=choose(s,clf,margin,force_present=s['present'],oracle_payload=False)
            elif mode=='oracle_payload':v=choose(s,clf,margin,force_present=None,oracle_payload=True)
            else:v=choose(s,clf,margin)
            if v is not None and not v['omit']:
                pred[s['p']]=v['value'];src['+'.join(v['sources'])]+=1
            lvl=str((gi.get(s['p']) or {}).get('type','unknown'));m=lev[lvl];m['n']+=1;m['pred']+=int(s['p'] in pred);m['correct']+=int(s['p'] in pred and s['p'] in gold and base.norm(pred[s['p']])==base.norm(gold[s['p']]))
        c0,p0,g0,_,e0=base.arg_metrics(pred,gold);C+=c0;P+=p0;G+=g0;exact+=e0
    pr=C/max(1,P);rc=C/max(1,G);f=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact_argument_set':exact/max(1,tasks),
            'levels':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in lev.items()},'source_sets':dict(src)}


def tune_margin(rows,ids,clf):
    scored=[(metrics(rows,ids,clf,m)['f1'],m) for m in MARGINS]
    return max(scored,key=lambda x:(x[0],-abs(x[1])))


def oracle(rows):
    lev=defaultdict(lambda:Counter(n=0,covered=0));pres=Counter();absn=Counter();multi=Counter()
    for r in rows:
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for s in r['slots']:
            if s['present']:
                lvl=str((gi.get(s['p']) or {}).get('type','unknown'));hit=any(v['label'] for v in s['vals']);lev[lvl]['n']+=1;lev[lvl]['covered']+=int(hit);pres['n']+=1;pres['covered']+=int(hit)
                good=next((v for v in s['vals'] if v['label']),None)
                if good is not None:multi['covered']+=1;multi['evidence']+=good['evidence_count'];multi['multi_src']+=int(len(good['sources'])>1)
            else:absn['n']+=1;absn['covered']+=int(any(v['omit'] for v in s['vals']))
    return {'present_overall':pres['covered']/max(1,pres['n']),'absent_omit':absn['covered']/max(1,absn['n']),
            'by_grounding':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n'])} for k,v in lev.items()},
            'correct_value_mean_evidence':multi['evidence']/max(1,multi['covered']),'correct_value_multi_source_rate':multi['multi_src']/max(1,multi['covered'])}


def main():
    enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'));rows,missing=build_rows(enc)
    idx=np.arange(N);kf=KFold(n_splits=5,shuffle=True,random_state=20260820);folds=[(set(idx[tr].tolist()),set(idx[va].tolist())) for tr,va in kf.split(idx)]
    grid=[]
    for Cval in CS:
        fs=[]
        for tr,va in folds:
            clf,n,g,u=fit(rows,tr,Cval);train_f,margin=tune_margin(rows,tr,clf);res=metrics(rows,va,clf,margin);res['margin']=margin;res['train_margin_f1']=train_f;fs.append(res)
        mf=float(np.mean([x['f1'] for x in fs]));grid.append((mf,Cval,fs));print('VALUE_AGG_CV_C',Cval,mf,flush=True)
    meanF,Cbest,foldres=max(grid,key=lambda z:z[0]);clf,n,g,u=fit(rows,set(range(N)),Cbest);train_f,margin=tune_margin(rows,set(range(N)),clf)
    full=metrics(rows,set(range(N)),clf,margin);pres=metrics(rows,set(range(N)),clf,margin,mode='oracle_presence');pay=metrics(rows,set(range(N)),clf,margin,mode='oracle_payload')
    result={'stage':'Gate v2 development-only value-level evidence aggregation + within-slot pairwise ranking','split':'QA001-100 development only; QA101-400 labels remain sealed',
            'candidate_oracle':oracle(rows),'grid':[{'C':c,'mean_f1':m,'folds':[{'f1':x['f1'],'precision':x['precision'],'recall':x['recall'],'margin':x['margin'],'levels':x['levels']} for x in fs]} for m,c,fs in grid],
            'selected_C':Cbest,'selected_mean_cv_f1':meanF,'all_dev_margin':margin,'all_dev_refit':full,
            'diagnostic_oracle_presence_learned_payload':pres,'diagnostic_learned_emit_oracle_payload':pay,
            'pairwise_rows':n,'slot_groups':g,'usable_pairwise_groups':u,'aggregate_feature_dim':len(rows[0]['slots'][0]['vals'][0]['feat']),
            'embedding_cache_entries':len(enc.cache),'missing_zero_based':missing,
            'guardrail':'Exact candidate values are executor payloads only, never aggregate features. Aggregation uses source/operator/provenance statistics. Margin, C, and all diagnostics use QA001-100 development only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_GATEA_VALUE_AGG='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
