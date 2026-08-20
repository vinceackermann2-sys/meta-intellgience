from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import learned_candidate_router as lr
import strong_banking77 as base

OUT=Path(__file__).with_name('crossencoder_candidate_router_result.json')
TRAIN_END=70
DEV_END=100


def target_text(s):
    d=s.get('def') or {}
    return (
        f"User request: {s.get('query','')}\n"
        f"Target tool: {s.get('target','')}\n"
        f"Target parameter: {s.get('parameter','')}\n"
        f"Parameter meaning: {d.get('description','')}\n"
        f"Parameter type: {d.get('type','')}"
    )


def candidate_text(s,c):
    return lr.candidate_context(s,c)


def add_cross_scores(slots, ce):
    pair_meta=[]; pairs=[]
    for si,s in enumerate(slots):
        for ci,c in enumerate(s.get('candidates') or []):
            pairs.append([target_text(s), candidate_text(s,c)])
            pair_meta.append((si,ci))
    scores=ce.predict(pairs,batch_size=64,show_progress_bar=True,convert_to_numpy=True)
    byslot=defaultdict(list)
    for z,(si,ci) in zip(scores,pair_meta):
        byslot[si].append((ci,float(z)))
    for si,s in enumerate(slots):
        xs=byslot.get(si,[])
        if not xs: continue
        ordered=sorted(xs,key=lambda x:-x[1]); ranks={ci:r for r,(ci,_) in enumerate(ordered)}
        smap={ci:z for ci,z in xs}
        for ci,row in enumerate(s.get('feature_rows') or []):
            row['cross_score']=smap.get(ci,-100.0)
            row['cross_rank']=float(ranks.get(ci,999))
            row['cross_top1']=float(ranks.get(ci,999)==0)
    return slots


def required_set(s):
    return lr.required_set(s)


def score_subset(slots,clf,vec,lo,hi,threshold):
    C=P=G=tasks=exact=0
    byground=defaultdict(lambda:Counter(n=0,correct=0)); coverage=Counter();examples=[]
    byqa=defaultdict(list)
    for s in slots:
        if lo<=s['qi']<hi: byqa[s['qi']].append(s)
    for qi,ss in sorted(byqa.items()):
        tasks+=1;qa=ss[0]['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});pred={};dbg={}
        for s in ss:
            cand=s.get('candidates') or []
            if not cand:continue
            X=vec.transform(s['feature_rows']);probs=clf.predict_proba(X)[:,1]
            best=int(np.argmax(probs));p=float(probs[best]);required=s['parameter'] in required_set(s)
            coverage['slots']+=1;coverage['candidate_has_gold']+=int(any(s['labels']))
            if p>=threshold or required:
                pred[s['parameter']]=base.coerce(cand[best]['value'],s['def'])
                dbg[s['parameter']]={'p':p,'cross':float(s['feature_rows'][best].get('cross_score',-100)),'source':cand[best].get('source'),'value':cand[best].get('value')}
        correct=0;ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for k,v in gold.items():
            typ=str((ground.get(k) or {}).get('type','unknown'));byground[typ]['n']+=1
            if k in pred and base.norm(pred[k])==base.norm(v):correct+=1;byground[typ]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
        if len(examples)<5:examples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':dbg})
    pr=C/max(1,P);rc=C/max(1,G);f1=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f1,'exact':exact/max(1,tasks),'candidate_coverage':coverage['candidate_has_gold']/max(1,coverage['slots']),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(byground.items())},'examples':examples}


def cross_only_subset(slots,lo,hi):
    C=P=G=tasks=0;byground=defaultdict(lambda:Counter(n=0,correct=0))
    byqa=defaultdict(list)
    for s in slots:
        if lo<=s['qi']<hi:byqa[s['qi']].append(s)
    for _,ss in sorted(byqa.items()):
        tasks+=1;qa=ss[0]['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});pred={}
        for s in ss:
            cand=s.get('candidates') or []; rows=s.get('feature_rows') or []
            if not cand:continue
            best=int(np.argmax([r.get('cross_score',-100) for r in rows]));pred[s['parameter']]=base.coerce(cand[best]['value'],s['def'])
        ground=((qa.get('tool_call') or {}).get('grounding_info') or {});correct=0
        for k,v in gold.items():
            typ=str((ground.get(k) or {}).get('type','unknown'));byground[typ]['n']+=1
            if k in pred and base.norm(pred[k])==base.norm(v):correct+=1;byground[typ]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold)
    pr=C/max(1,P);rc=C/max(1,G);f1=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'precision':pr,'recall':rc,'f1':f1,'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(byground.items())}}


def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    _,slots,missing=lr.build_slots(enc)
    ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2',device='cpu')
    add_cross_scores(slots,ce)
    train=[s for s in slots if s['qi']<TRAIN_END and any(s['labels'])]
    rows=[];ys=[]
    for s in train:rows.extend(s['feature_rows']);ys.extend(s['labels'])
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(rows)
    clf=LogisticRegression(max_iter=1500,class_weight='balanced',C=1.0,random_state=42).fit(X,ys)
    thresholds=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    curve=[]
    for t in thresholds:
        r=score_subset(slots,clf,vec,TRAIN_END,DEV_END,t);curve.append({'threshold':t,'f1':r['f1'],'precision':r['precision'],'recall':r['recall']})
    best=max(curve,key=lambda x:(x['f1'],x['precision']))['threshold']
    train_result=score_subset(slots,clf,vec,0,TRAIN_END,best)
    val_result=score_subset(slots,clf,vec,TRAIN_END,DEV_END,best)
    cross_only=cross_only_subset(slots,TRAIN_END,DEV_END)
    names=vec.get_feature_names_out();w=clf.coef_[0];coef=[]
    for i in np.argsort(-np.abs(w))[:25]:coef.append({'feature':str(names[i]),'weight':float(w[i])})
    result={'stage':'Mem2Act cross-encoder exact candidate router','representation':'MS-MARCO MiniLM cross-encoder scores target-slot semantics against provenance-aware candidate evidence, then a tiny logistic calibration layer; no generative LLM','split':'QA001-70 train, QA071-100 validation; QA101-400 gold sealed','missing_session_indices_zero_based':missing,'training':{'candidate_rows':len(ys),'positive_rows':int(sum(ys)),'slots_with_positive_candidate':len(train)},'selected_threshold':best,'validation_curve':curve,'cross_encoder_only_validation':cross_only,'train_result':train_result,'validation_result':val_result,'top_abs_coefficients':coef,'guardrail':'QA001-70 gold is used only to fit candidate correctness; QA071-100 only calibrates threshold/scores; no answer string is a feature. QA101-400 gold remains unopened.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_CROSSENCODER_ROUTER='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
