from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import joint_schema_alignment as js
import gatea_source_selector as ss

OUT=Path(__file__).with_name('gatea_pairwise_cv_result.json')
N=100; JOINT=(0.75,0.0,2); CS=[0.01,0.03,0.1,0.25,0.5,1.0,2.0]

class CachedEncoder:
    """Exact-text embedding memoizer. Scientific outputs are unchanged; repeated encode calls reuse the same vector."""
    def __init__(self, model): self.model=model; self.cache={}
    def encode(self, sentences, **kw):
        one=isinstance(sentences,str); xs=[sentences] if one else list(sentences)
        norm=bool(kw.get('normalize_embeddings',False)); key_extra=(norm,)
        missing=[]
        for x in xs:
            k=(str(x),)+key_extra
            if k not in self.cache: missing.append(str(x))
        if missing:
            # unique in stable order
            uniq=list(dict.fromkeys(missing)); arr=self.model.encode(uniq,**kw)
            arr=np.asarray(arr,dtype=np.float32)
            if arr.ndim==1: arr=arr[None,:]
            for x,v in zip(uniq,arr): self.cache[(x,)+key_extra]=np.asarray(v,dtype=np.float32)
        out=np.stack([self.cache[(str(x),)+key_extra] for x in xs],axis=0)
        return out[0] if one else out

def build_rows(enc):
    td=Path(tempfile.gettempdir())/'mab_pairwise_cv'; td.mkdir(exist_ok=True)
    qp=td/'q'; cp=td/'c'; base.fetch(base.BASE+'qa_dataset.jsonl',qp); base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N]; sessions,by=base.build_session_map(cp); rows=[]; missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None: missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        pack=js.qa_pack(enc,qa,ses) if ses is not None else None
        jp=js.predict(pack,*JOINT) if pack is not None else {}
        gold=((qa.get('tool_call') or {}).get('arguments') or {}); slots=[]
        for p,d0 in props.items():
            d=d0 or {}; cands=ss.build_slot(enc,qa,ses,p,d,jp)
            labels=[int((p not in gold and c['src']=='omit') or (p in gold and c['src']!='omit' and base.norm(c['value'])==base.norm(gold[p]))) for c in cands]
            slots.append({'p':p,'d':d,'cands':cands,'labels':labels,'gold':gold.get(p),'present':p in gold})
        rows.append({'qi':qi,'qa':qa,'slots':slots})
        if qi%10==0: print('PAIRWISE_BUILD',qi,'cache',len(enc.cache),flush=True)
    return rows,missing

def pairwise_data(rows, train_ids):
    X=[];Y=[];groups=0;usable=0
    for r in rows:
        if r['qi'] not in train_ids: continue
        for s in r['slots']:
            pos=[i for i,y in enumerate(s['labels']) if y]; neg=[i for i,y in enumerate(s['labels']) if not y]
            groups+=1
            if not pos or not neg: continue
            usable+=1
            # cap negatives deterministically to avoid huge duplicated source pools
            neg=neg[:40]
            for pi in pos[:4]:
                pf=np.asarray(s['cands'][pi]['feat'],float)
                for ni in neg:
                    nf=np.asarray(s['cands'][ni]['feat'],float); d=pf-nf
                    X.append(d);Y.append(1);X.append(-d);Y.append(0)
    return np.asarray(X,float),np.asarray(Y,int),groups,usable

def fit(rows,train_ids,C):
    X,Y,g,u=pairwise_data(rows,train_ids)
    clf=LogisticRegression(C=C,max_iter=5000,fit_intercept=False,class_weight=None,random_state=42).fit(X,Y)
    return clf,len(Y),g,u

def score(rows,ids,clf):
    C=P=G=exact=tasks=0; lev=defaultdict(lambda:Counter(n=0,correct=0,pred=0)); src=Counter()
    for r in rows:
        if r['qi'] not in ids: continue
        qa=r['qa']; gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {}); pred={}; tasks+=1
        for s in r['slots']:
            F=np.asarray([c['feat'] for c in s['cands']],float); scores=clf.decision_function(F); i=int(np.argmax(scores)); c=s['cands'][i]
            if c['src']!='omit': pred[s['p']]=c['value']; src[c['src']+':'+c['op']]+=1
            lvl=str((gi.get(s['p']) or {}).get('type','unknown')); m=lev[lvl]; m['n']+=1; m['pred']+=int(s['p'] in pred); m['correct']+=int(s['p'] in pred and s['p'] in gold and base.norm(pred[s['p']])==base.norm(gold[s['p']]))
        c0,p0,g0,_,e0=base.arg_metrics(pred,gold); C+=c0;P+=p0;G+=g0;exact+=e0
    pr=C/max(1,P); rc=C/max(1,G); f=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact_argument_set':exact/max(1,tasks),'levels':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in lev.items()},'sources':dict(src)}

def oracle(rows):
    lev=defaultdict(lambda:Counter(n=0,covered=0)); pres=Counter(); absn=Counter()
    for r in rows:
        qa=r['qa']; gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for s in r['slots']:
            if s['p'] in gold:
                lvl=str((gi.get(s['p']) or {}).get('type','unknown')); hit=any(s['labels']); lev[lvl]['n']+=1;lev[lvl]['covered']+=int(hit);pres['n']+=1;pres['covered']+=int(hit)
            else:
                hit=any(c['src']=='omit' for c in s['cands']);absn['n']+=1;absn['covered']+=int(hit)
    return {'present_overall':pres['covered']/max(1,pres['n']),'absent_omit':absn['covered']/max(1,absn['n']),'by_grounding':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n'])} for k,v in lev.items()}}

def main():
    enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'))
    rows,missing=build_rows(enc); idx=np.arange(N); kf=KFold(n_splits=5,shuffle=True,random_state=20260820); folds=[(set(idx[tr].tolist()),set(idx[va].tolist())) for tr,va in kf.split(idx)]
    grid=[]
    for Cval in CS:
        fs=[]
        for tr,va in folds:
            clf,n,g,u=fit(rows,tr,Cval); fs.append(score(rows,va,clf))
        mf=float(np.mean([x['f1'] for x in fs])); grid.append((mf,Cval,fs)); print('PAIRWISE_CV_C',Cval,mf,flush=True)
    meanF,Cbest,foldres=max(grid,key=lambda z:z[0]); clf,n,g,u=fit(rows,set(range(N)),Cbest); full=score(rows,set(range(N)),clf)
    result={'stage':'Gate v2 development-only value-blind pairwise within-slot operation ranking','split':'QA001-100 development only; QA101-400 labels remain sealed','candidate_oracle':oracle(rows),'grid':[{'C':c,'mean_f1':m,'folds':[{'f1':x['f1'],'precision':x['precision'],'recall':x['recall'],'levels':x['levels']} for x in fs]} for m,c,fs in grid],'selected_C':Cbest,'selected_mean_cv_f1':meanF,'all_dev_refit':full,'pairwise_rows':n,'slot_groups':g,'usable_pairwise_groups':u,'embedding_cache_entries':len(enc.cache),'missing_zero_based':missing,'guardrail':'Pairwise labels compare candidates only within a development slot. Candidate values are never features. grounding_info is reporting-only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('MEM2ACT_GATEA_PAIRWISE_CV='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
