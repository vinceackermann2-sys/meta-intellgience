from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,tempfile,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import joint_schema_alignment as js
import gatea_source_selector as ss

OUT=Path(__file__).with_name('gatea_dev_cv_result.json')
N=100;JOINT=(0.75,0.0,2);CS=[0.03,0.1,0.25,0.5,1.0,2.0]
_CACHE={}
def cached_retrieve(enc,eps,query_text,k=es.TOP_EPISODES):
    if not eps:return []
    fp=hashlib.sha1(('\n'.join(str(e.get('source_id',''))+'|'+e.get('text','')[:6000] for e in eps)).encode()).hexdigest()
    if fp not in _CACHE:
        _CACHE[fp]=enc.encode([e['text'][:6000] for e in eps],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    E=_CACHE[fp];q=enc.encode([query_text],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0];order=np.argsort(-(E@q))[:min(k,len(eps))]
    return [(rank,eps[int(i)],float(E[int(i)]@q)) for rank,i in enumerate(order)]
es.retrieve_episodes=cached_retrieve

def fit(rows,train_ids,C):
    X=[];Y=[]
    for r in rows:
        if r['qi'] not in train_ids:continue
        for s in r['slots']:
            for c in s['cands']:
                ok=(not s['present'] and c['src']=='omit') or (s['present'] and c['src']!='omit' and base.norm(c['value'])==base.norm(s['gold']))
                X.append(c['feat']);Y.append(int(ok))
    return LogisticRegression(C=C,max_iter=4000,class_weight='balanced',random_state=42).fit(np.asarray(X,float),np.asarray(Y,int)),len(Y),sum(Y)

def score(rows,ids,clf):
    C=P=G=exact=tasks=0;lev=defaultdict(lambda:Counter(n=0,correct=0,pred=0));src=Counter()
    for r in rows:
        if r['qi'] not in ids:continue
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pred={};tasks+=1
        for s in r['slots']:
            probs=clf.predict_proba(np.asarray([c['feat'] for c in s['cands']],float))[:,1];c=s['cands'][int(np.argmax(probs))]
            if c['src']!='omit':pred[s['p']]=c['value'];src[c['src']+':'+c['op']]+=1
            lvl=str((gi.get(s['p']) or {}).get('type','unknown'));m=lev[lvl];m['n']+=1;m['pred']+=int(s['p'] in pred);m['correct']+=int(s['p'] in pred and s['p'] in gold and base.norm(pred[s['p']])==base.norm(gold[s['p']]))
        c0,p0,g0,f0,e0=base.arg_metrics(pred,gold);C+=c0;P+=p0;G+=g0;exact+=e0
    pr=C/max(1,P);rc=C/max(1,G);F=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':F,'exact_argument_set':exact/max(1,tasks),'levels':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in lev.items()},'sources':dict(src)}

def oracle(rows):
    lev=defaultdict(lambda:Counter(n=0,covered=0));present=Counter();absent=Counter()
    for r in rows:
        qa=r['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for s in r['slots']:
            if s['p'] in gold:
                lvl=str((gi.get(s['p']) or {}).get('type','unknown'));m=lev[lvl];m['n']+=1;ok=any(c['src']!='omit' and base.norm(c['value'])==base.norm(gold[s['p']]) for c in s['cands']);m['covered']+=int(ok);present['n']+=1;present['covered']+=int(ok)
            else:
                absent['n']+=1;absent['covered']+=int(any(c['src']=='omit' for c in s['cands']))
    return {'present_overall':present['covered']/max(1,present['n']),'absent_omit':absent['covered']/max(1,absent['n']),'by_grounding':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n'])} for k,v in lev.items()}}

def main():
    td=Path(tempfile.gettempdir())/'mab_dev_cv';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rows=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});slots=[]
        for p,d0 in props.items():
            d=d0 or {};slots.append({'p':p,'d':d,'cands':ss.build_slot(enc,qa,ses,p,d,jp),'present':p in gold,'gold':gold.get(p)})
        rows.append({'qi':qi,'qa':qa,'slots':slots})
        if qi%10==0:print('DEV_CV_BUILD',qi,flush=True)
    idx=np.arange(N);kf=KFold(n_splits=5,shuffle=True,random_state=20260820);folds=[(set(idx[tr].tolist()),set(idx[va].tolist())) for tr,va in kf.split(idx)]
    grid=[]
    for Cval in CS:
        fr=[]
        for fi,(tr,va) in enumerate(folds):
            clf,n,pos=fit(rows,tr,Cval);sc=score(rows,va,clf);fr.append(sc)
        mf=float(np.mean([x['f1'] for x in fr]));grid.append((mf,Cval,fr));print('DEV_CV_C',Cval,mf,flush=True)
    best=max(grid,key=lambda z:z[0]);meanF,Cbest,foldres=best;clf,n,pos=fit(rows,set(range(N)),Cbest);full=score(rows,set(range(N)),clf)
    result={'stage':'Gate v2 development-only candidate oracle + grouped 5-fold source/operation selector CV','split':'QA001-100 development only; no QA101-400 labels read','candidate_oracle':oracle(rows),'grid':[{'C':c,'mean_f1':m,'folds':[{'f1':x['f1'],'precision':x['precision'],'recall':x['recall'],'levels':x['levels']} for x in fs]} for m,c,fs in grid],'selected_C':Cbest,'selected_mean_cv_f1':meanF,'all_dev_refit':full,'training_rows':n,'positive_candidate_rows':int(pos),'missing_zero_based':missing,'guardrail':'Candidate outputs are training labels/executor payloads only, never input features. grounding_info is used only to report level metrics. This run reads QA001-100 only. The preregistered Gate-v2 50-QA validation and 250-QA final sets remain sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_GATEA_DEV_CV='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
