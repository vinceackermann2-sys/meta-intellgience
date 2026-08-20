from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import strong_banking77 as base

OUT=Path('researchstrong/learned_candidate_router_result.json')
TRAIN_END=70
DEV_END=100

def required_set(s):
    schema=(s.get('qa') or {}).get('target_tool_schema') or {}
    return set(((schema.get('parameters') or {}).get('required') or []))

def add_policy_candidates(s):
    p=str(s['parameter']);d=s['def'] or {};typ=str(d.get('type','')).lower();desc=str(d.get('description','')).lower();req=required_set(s);optional=p not in req;pl=p.lower().replace('_','')
    vals=[]
    if optional and typ in ('array','list'):vals.append(([], 'policy_optional_empty_array'))
    if optional and typ in ('string','str'):vals.append(('', 'policy_optional_empty_string'))
    if typ in ('bool','boolean'):vals.append((False, 'policy_boolean_false'))
    if 'offset' in pl:vals.append((0, 'policy_offset_zero'))
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl):vals.append((1, 'policy_first_page_one'))
    if pl=='index' and ('latest' in desc or 'most recent' in desc or 'starting from 0' in desc or 'start from 0' in desc):vals.append((0, 'policy_latest_index_zero'))
    cand=list(s.get('candidates') or []);seen={base.norm(c.get('value')) for c in cand}
    for v,src in vals:
        if base.norm(v) not in seen:
            cand.append({'value':v,'source':src,'evidence':'general schema/type executor policy','priority':3});seen.add(base.norm(v))
    cand=sorted(cand,key=lambda x:x.get('priority',5))[:32]
    for j,c in enumerate(cand):c['id']=f'C{j}'
    s['candidates']=cand
    return s

def toks(x):
    return set(re.findall(r'[a-z0-9]+',str(x).lower()))
def jacc(a,b):
    a=toks(a);b=toks(b)
    return len(a&b)/max(1,len(a|b))
def parse_turn(e):
    m=re.search(r'\bturn\s+(\d+)\b',str(e or ''),flags=re.I)
    return int(m.group(1)) if m else -1

def candidate_context(s,c):
    ev=str(c.get('evidence',''))
    ti=parse_turn(ev)
    if ti>=0:
        for x in s.get('evidence') or []:
            if int(x.get('turn',-2))==ti:
                ev += ' '+str(x.get('text',''))
                break
    return f"source {c.get('source','')} value {c.get('value')} evidence {ev}"

def candidate_features(s,c,sim,rank,max_turn):
    q=str(s.get('query',''));desc=str((s.get('def') or {}).get('description',''));p=str(s.get('parameter',''));value=c.get('value');source=str(c.get('source',''));ev=str(c.get('evidence',''))
    ti=parse_turn(ev);rec=(ti/max(1,max_turn)) if ti>=0 else -1.0
    nv=base.norm(value);typ=str((s.get('def') or {}).get('type','')).lower()
    return {
      'sim':float(sim),'rank':float(rank),'priority':float(c.get('priority',5)),'recency':float(rec),
      'value_len':float(len(str(value))),'query_value_exact':float(bool(nv and nv in base.norm(q))),
      'query_value_tokens':float(jacc(q,value)),'query_desc_tokens':float(jacc(q,desc+' '+p)),
      'evidence_query_tokens':float(jacc(q,ev)),'required':float(p in required_set(s)),
      'type_'+typ:1.0,'source_'+source:1.0,
      'is_default':float('default' in source),'is_policy':float(source.startswith('policy_')),
      'is_same_slot':float(source=='prior_same_slot'),'is_current':float(source=='current_request_span'),
      'is_span':float(source=='episodic_span'),'is_semantic_tool':float(source=='semantic_tool_value'),
    }

def build_slots(enc):
    td=Path(tempfile.gettempdir())/'mem2act_router';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:DEV_END];sessions,by=base.build_session_map(cp)
    slots=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        ss=base.compile_task(qa,ses,enc)
        gold=((qa.get('tool_call') or {}).get('arguments') or {});ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        max_turn=max([int(x.get('turn',0)) for s in ss for x in (s.get('evidence') or [])]+[1])
        for s in ss:
            s['qi']=qi;s['qa']=qa;add_policy_candidates(s)
            gv=gold.get(s['parameter'],'__MISSING_GOLD_SLOT__');s['gold_value']=gv;s['ground_type']=str((ground.get(s['parameter']) or {}).get('type','unknown'))
            cand=s.get('candidates') or []
            if cand:
                qtxt=f"request {s.get('query','')} target {s.get('target','')} parameter {s['parameter']} meaning {(s.get('def') or {}).get('description','')}"
                ctexts=[candidate_context(s,c) for c in cand]
                E=enc.encode([qtxt]+ctexts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
                sims=E[1:]@E[0]
                s['feature_rows']=[candidate_features(s,c,float(sims[i]),i,max_turn) for i,c in enumerate(cand)]
                s['labels']=[int(gv!='__MISSING_GOLD_SLOT__' and base.norm(c.get('value'))==base.norm(gv)) for c in cand]
            else:s['feature_rows']=[];s['labels']=[]
            slots.append(s)
    return qas,slots,missing

def score_subset(slots,clf,vec,lo,hi,threshold):
    C=P=G=tasks=exact=0;byground=defaultdict(lambda:Counter(n=0,correct=0));coverage=Counter();examples=[]
    byqa=defaultdict(list)
    for s in slots:
        if lo<=s['qi']<hi:byqa[s['qi']].append(s)
    for qi,ss in sorted(byqa.items()):
        tasks+=1;qa=ss[0]['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});pred={};dbg={}
        for s in ss:
            cand=s.get('candidates') or []
            if not cand:continue
            X=vec.transform(s['feature_rows']);probs=clf.predict_proba(X)[:,1];order=np.argsort(-probs);best=int(order[0]);p=float(probs[best])
            has_gold=any(s['labels']);coverage['slots']+=1;coverage['candidate_has_gold']+=int(has_gold)
            required=s['parameter'] in required_set(s)
            if p>=threshold or required:
                pred[s['parameter']]=base.coerce(cand[best]['value'],s['def']);dbg[s['parameter']]={'p':p,'source':cand[best].get('source'),'value':cand[best].get('value')}
        correct=0
        ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for k,v in gold.items():
            typ=str((ground.get(k) or {}).get('type','unknown'));byground[typ]['n']+=1
            if k in pred and base.norm(pred[k])==base.norm(v):correct+=1;byground[typ]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
        if len(examples)<4:examples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':dbg})
    pr=C/max(1,P);rc=C/max(1,G);f1=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f1,'exact':exact/max(1,tasks),'candidate_coverage':coverage['candidate_has_gold']/max(1,coverage['slots']),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(byground.items())},'examples':examples}

def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');qas,slots,missing=build_slots(enc)
    train=[s for s in slots if s['qi']<TRAIN_END and any(s['labels'])]
    rows=[];ys=[]
    for s in train:
        rows.extend(s['feature_rows']);ys.extend(s['labels'])
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(rows);clf=LogisticRegression(max_iter=1000,class_weight='balanced',C=1.0,random_state=42).fit(X,ys)
    thresholds=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7]
    val_curve=[]
    for t in thresholds:
        r=score_subset(slots,clf,vec,TRAIN_END,DEV_END,t);val_curve.append({'threshold':t,'f1':r['f1'],'precision':r['precision'],'recall':r['recall']})
    best=max(val_curve,key=lambda x:(x['f1'],x['precision']))['threshold']
    train_result=score_subset(slots,clf,vec,0,TRAIN_END,best);val_result=score_subset(slots,clf,vec,TRAIN_END,DEV_END,best)
    coef=[]
    names=vec.get_feature_names_out();w=clf.coef_[0]
    for i in np.argsort(-np.abs(w))[:25]:coef.append({'feature':str(names[i]),'weight':float(w[i])})
    result={'stage':'Mem2Act learned exact-candidate router','representation':'all-MiniLM-L6-v2 candidate/query features + logistic candidate ranker; no generative LLM','split':'QA001-70 train, QA071-100 validation; QA101-400 gold sealed','missing_session_indices_zero_based':missing,'training':{'candidate_rows':len(ys),'positive_rows':int(sum(ys)),'slots_with_positive_candidate':len(train)},'selected_threshold':best,'validation_curve':val_curve,'train_result':train_result,'validation_result':val_result,'top_abs_coefficients':coef,'guardrail':'Gold values are used only for QA001-70 router labels and QA071-100 threshold selection/scoring. No answer strings are model features. QA101-400 gold arguments remain unopened.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_LEARNED_ROUTER='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
