from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import strong_banking77 as base
import learned_candidate_router as lr

OUT=Path(__file__).with_name('episode_scoped_router_result.json')
TRAIN_END=70
DEV_END=100
TOP_EPISODES=2
OMIT='__MEMORY_ACTION_IR_OMIT__'


def turn_text(t):
    c=t.get('content','');c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
    tc=t.get('tool_calls') or []
    return f"{t.get('role','')}: {c}"+(f" TOOL_CALLS={json.dumps(tc,ensure_ascii=False)}" if tc else '')


def flatten_items(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            key=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten_items(v,key))
            else:out.append((key,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            key=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)):out.extend(flatten_items(v,key))
            else:out.append((key,v))
    return out


def episodes(session):
    by=defaultdict(list);order=[]
    for ti,t in enumerate(session.get('turns') or []):
        sid=str(t.get('source_id') or f'unknown_turn_{ti}')
        if sid not in by:order.append(sid)
        by[sid].append((ti,t))
    result=[]
    for sid in order:
        rows=by[sid];txt='\n'.join(turn_text(t) for _,t in rows)
        events=[]
        for ti,t in rows:
            for tc in t.get('tool_calls') or []:
                args=base.parse_args(tc)
                for key,val in flatten_items(args):events.append({'turn':ti,'tool':base.tool_name(tc),'key':key,'value':val})
        result.append({'source_id':sid,'text':txt,'rows':rows,'events':events})
    return result


def spans(text):
    return base.spans(text)


def required_set(qa):
    return set((((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('required') or []))


def add_candidate(cands,seen,value,source,evidence='',hist_key='',tool='',episode_rank=-1,turn=-1,priority=5):
    key=('omit',) if value==OMIT else ('value',base.norm(value))
    if key in seen:return
    seen.add(key);cands.append({'value':value,'source':source,'evidence':str(evidence)[:500],'hist_key':hist_key,'tool':tool,'episode_rank':episode_rank,'turn':turn,'priority':priority})


def policy_values(p,d,required):
    pl=str(p).lower().replace('_','');typ=str(d.get('type','')).lower();desc=str(d.get('description','')).lower();optional=p not in required;vals=[]
    if optional and typ in ('array','list'):vals.append(([], 'policy_optional_empty_array'))
    if optional and typ in ('string','str'):vals.append(('', 'policy_optional_empty_string'))
    if typ in ('bool','boolean'):vals.append((False,'policy_boolean_false'))
    if 'offset' in pl:vals.append((0,'policy_offset_zero'))
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl):vals.append((1,'policy_first_page_one'))
    if pl=='index' and any(x in desc for x in ['latest','most recent','starting from 0','start from 0']):vals.append((0,'policy_latest_index_zero'))
    return vals


def retrieve_episodes(enc,eps,query_text,k=TOP_EPISODES):
    if not eps:return []
    texts=[e['text'][:6000] for e in eps]
    E=enc.encode(texts,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    q=enc.encode([query_text],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
    order=np.argsort(-(E@q))[:min(k,len(eps))]
    return [(rank,eps[int(i)],float(E[int(i)]@q)) for rank,i in enumerate(order)]


def compile_slots(enc,qa,session,qi):
    eps=episodes(session);schema=qa.get('target_tool_schema') or {};params=schema.get('parameters') or {};props=params.get('properties') or {};required=set(params.get('required') or []);query=str(qa.get('query',''));slots=[]
    for p,d0 in props.items():
        d=d0 or {};target=f"Current request: {query}. Target tool: {schema.get('name','')}. Target parameter: {p}. Meaning: {d.get('description','')}. Type: {d.get('type','')}"
        picked=retrieve_episodes(enc,eps,target,TOP_EPISODES)
        cands=[];seen=set()
        # OMIT is a first-class executable operation, not an absent generation.
        add_candidate(cands,seen,OMIT,'omit','schema-aware omission candidate',priority=7)
        if 'default' in d:add_candidate(cands,seen,d['default'],'schema_default',d.get('description',''),priority=0)
        for v in base.desc_defaults(d.get('description','')):add_candidate(cands,seen,v,'description_default',d.get('description',''),priority=1)
        if isinstance(d.get('enum'),list):
            for v in d['enum']:add_candidate(cands,seen,v,'schema_enum',d.get('description',''),priority=2)
        for v,src in policy_values(p,d,required):add_candidate(cands,seen,v,src,'general schema/type policy',priority=2)
        # Only candidates from the semantically retrieved source-ID episodes.
        for erank,ep,esim in picked:
            for ev in ep['events']:
                keyleaf=str(ev['key']).split('.')[-1]
                src='episode_same_slot' if keyleaf.casefold()==p.casefold() else 'episode_tool_value'
                pr=1 if src=='episode_same_slot' else 4
                add_candidate(cands,seen,ev['value'],src,f"source_id={ep['source_id']} episode_similarity={esim:.4f}",hist_key=ev['key'],tool=ev['tool'],episode_rank=erank,turn=ev['turn'],priority=pr)
            for v in spans(ep['text']):add_candidate(cands,seen,v,'episode_span',f"source_id={ep['source_id']} episode_similarity={esim:.4f}",episode_rank=erank,priority=5)
        for v in spans(query):add_candidate(cands,seen,v,'current_request_span','current request',episode_rank=-1,priority=3)
        # Keep exact defaults/same-slot values plus bounded semantic residual pool.
        cands=sorted(cands,key=lambda c:(c['priority'],c.get('episode_rank',9)))[:40]
        qv=enc.encode([target],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
        ctexts=[]
        for c in cands:
            if c['value']==OMIT:ct='Operation OMIT: leave this parameter absent because it is not required by the current task.'
            else:ct=f"source {c['source']} historical parameter {c.get('hist_key','')} tool {c.get('tool','')} value {c.get('value')} evidence {c.get('evidence','')}"
            ctexts.append(ct)
        CE=enc.encode(ctexts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if ctexts else np.zeros((0,384),np.float32)
        sims=CE@qv if len(CE) else np.array([])
        maxturn=max([c.get('turn',-1) for c in cands]+[1])
        features=[]
        for i,c in enumerate(cands):
            value=c['value'];source=c['source'];histkey=c.get('hist_key','');ev=c.get('evidence','');typ=str(d.get('type','')).lower();leaf=histkey.split('.')[-1] if histkey else ''
            features.append({'sim':float(sims[i]) if len(sims) else -1.0,'priority':float(c['priority']),'episode_rank':float(c.get('episode_rank',-1)),'recency':float(c.get('turn',-1)/max(1,maxturn)),'required':float(p in required),'slot_name_match':float(bool(leaf) and leaf.casefold()==p.casefold()),'slot_name_jacc':float(lr.jacc(p+' '+str(d.get('description','')),histkey)),'query_value_overlap':float(lr.jacc(query,value if value!=OMIT else '')),'query_evidence_overlap':float(lr.jacc(query,ev)),'is_omit':float(value==OMIT),'type_'+typ:1.0,'source_'+source:1.0,'is_default':float('default' in source),'is_policy':float(source.startswith('policy_')),'is_same_slot':float(source=='episode_same_slot'),'is_episode':float(source.startswith('episode_'))})
        slots.append({'qi':qi,'qa':qa,'parameter':p,'def':d,'required':p in required,'candidates':cands,'features':features,'picked_episode_ids':[e['source_id'] for _,e,_ in picked]})
    return slots


def build(enc):
    td=Path(tempfile.gettempdir())/'mab_episode_router';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:DEV_END];sessions,by=base.build_session_map(cp);slots=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        gold=((qa.get('tool_call') or {}).get('arguments') or {});ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for s in compile_slots(enc,qa,ses,qi):
            if s['parameter'] in gold:
                gv=gold[s['parameter']];labels=[int(c['value']!=OMIT and base.norm(c['value'])==base.norm(gv)) for c in s['candidates']]
            else:
                gv=OMIT;labels=[int(c['value']==OMIT) for c in s['candidates']]
            s['gold']=gv;s['labels']=labels;s['grounding']=str((ground.get(s['parameter']) or {}).get('type','not_gold_slot'))
            slots.append(s)
    return qas,slots,missing


def score(slots,clf,vec,lo,hi,threshold):
    byqa=defaultdict(list)
    for s in slots:
        if lo<=s['qi']<hi:byqa[s['qi']].append(s)
    C=P=G=tasks=exact=0;byground=defaultdict(lambda:Counter(n=0,correct=0));cov=Counter();samples=[]
    for qi,ss in sorted(byqa.items()):
        qa=ss[0]['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});pred={};dbg={};tasks+=1
        for s in ss:
            X=vec.transform(s['features']);probs=clf.predict_proba(X)[:,1];best=int(np.argmax(probs));conf=float(probs[best]);cand=s['candidates'][best]
            if s['parameter'] in gold:
                cov['gold_slots']+=1;cov['gold_has_candidate']+=int(any(c['value']!=OMIT and base.norm(c['value'])==base.norm(gold[s['parameter']]) for c in s['candidates']))
            # Low confidence optional slots omit; required slots must emit best non-omit candidate.
            choose=cand
            if conf<threshold and not s['required']:choose={'value':OMIT,'source':'confidence_omit'}
            if choose['value']==OMIT and s['required']:
                non=[(float(probs[i]),c) for i,c in enumerate(s['candidates']) if c['value']!=OMIT]
                if non:choose=max(non,key=lambda z:z[0])[1]
            if choose['value']!=OMIT:pred[s['parameter']]=base.coerce(choose['value'],s['def'])
            dbg[s['parameter']]={'confidence':conf,'source':choose.get('source'),'value':None if choose['value']==OMIT else choose['value'],'episodes':s['picked_episode_ids']}
        correct=0;ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for k,v in gold.items():
            gt=str((ground.get(k) or {}).get('type','unknown'));byground[gt]['n']+=1
            if k in pred and base.norm(pred[k])==base.norm(v):correct+=1;byground[gt]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
        if len(samples)<5:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':dbg})
    pr=C/max(1,P);rc=C/max(1,G);f=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact':exact/max(1,tasks),'gold_value_candidate_coverage':cov['gold_has_candidate']/max(1,cov['gold_slots']),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(byground.items())},'samples':samples}


def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');_,slots,missing=build(enc)
    train=[s for s in slots if s['qi']<TRAIN_END]
    rows=[];ys=[]
    for s in train:rows.extend(s['features']);ys.extend(s['labels'])
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(rows);clf=LogisticRegression(max_iter=1500,class_weight='balanced',C=1.0,random_state=42).fit(X,ys)
    curve=[]
    for t in [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        r=score(slots,clf,vec,TRAIN_END,DEV_END,t);curve.append({'threshold':t,'f1':r['f1'],'precision':r['precision'],'recall':r['recall']})
    best=max(curve,key=lambda x:(x['f1'],x['precision']))['threshold'];tr=score(slots,clf,vec,0,TRAIN_END,best);va=score(slots,clf,vec,TRAIN_END,DEV_END,best)
    names=vec.get_feature_names_out();w=clf.coef_[0];coef=[{'feature':str(names[i]),'weight':float(w[i])} for i in np.argsort(-np.abs(w))[:25]]
    result={'stage':'Mem2Act episode-scoped exact-value router','architecture':'source_id episode retrieval top-2 -> exact candidates inside retrieved episodes + deterministic schema/policy state + first-class OMIT -> tiny logistic pointer ranker; no generative LLM','split':'QA001-70 train, QA071-100 validation; QA101-400 gold sealed','top_episodes':TOP_EPISODES,'missing_zero_based':missing,'training':{'candidate_rows':len(ys),'positive_rows':int(sum(ys)),'schema_slots':len(train)},'selected_threshold':best,'validation_curve':curve,'train_result':tr,'validation_result':va,'top_abs_coefficients':coef,'guardrail':'Raw released turn source_id provides episode boundaries. QA source_conversation_ids are never used to select episodes. Gold values only label QA001-70 candidate correctness and score/calibrate QA071-100. QA101-400 gold remains unopened.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_EPISODE_SCOPED_ROUTER='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
