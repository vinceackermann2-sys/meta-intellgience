from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import strong_banking77 as base
import episode_scoped_router as es
import learned_candidate_router as lr

OUT=Path(__file__).with_name('evidence_pointer_router_result.json')
TRAIN_END=70
DEV_END=100
TOP_EPISODES=2
OMIT='__MEMORY_ACTION_IR_OMIT__'
MAX_OCC=220


def safe_text(x):
    if x is None:return ''
    return json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x)

def flatten(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            key=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten(v,key))
            else:out.append((key,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            key=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)):out.extend(flatten(v,key))
            else:out.append((key,v))
    return out

def local_window(text,value,width=180):
    s=str(text);v=str(value)
    if not s:return ''
    i=s.casefold().find(v.casefold()) if v else -1
    if i<0:return s[:2*width]
    return s[max(0,i-width):min(len(s),i+len(v)+width)]

def occurrence_candidates(ep,episode_rank,episode_sim):
    occ=[]
    for ti,t in ep['rows']:
        role=str(t.get('role',''))
        content=t.get('content','')
        txt=safe_text(content)
        # Structured content leaves preserve exact path and local content.
        if isinstance(content,(dict,list)):
            for key,val in flatten(content):
                occ.append({'value':val,'source':'content_structured','role':role,'tool':'','hist_key':key,'turn':ti,'episode_rank':episode_rank,'episode_sim':episode_sim,'context':f'{role} structured field {key} = {safe_text(val)}; record {txt[:500]}'})
        elif isinstance(content,str):
            st=content.strip()
            if st[:1] in '[{':
                try:
                    obj=json.loads(st)
                    for key,val in flatten(obj):
                        occ.append({'value':val,'source':'content_json','role':role,'tool':'','hist_key':key,'turn':ti,'episode_rank':episode_rank,'episode_sim':episode_sim,'context':f'{role} JSON field {key} = {safe_text(val)}; record {local_window(st,val)}'})
                except Exception:pass
            # Text spans get the actual local sentence/window, not just episode id.
            for val in base.spans(txt):
                occ.append({'value':val,'source':'content_span','role':role,'tool':'','hist_key':'','turn':ti,'episode_rank':episode_rank,'episode_sim':episode_sim,'context':f'{role} text around value {safe_text(val)}: {local_window(txt,val)}'})
        for tc in t.get('tool_calls') or []:
            tool=base.tool_name(tc);args=base.parse_args(tc)
            for key,val in flatten(args):
                occ.append({'value':val,'source':'tool_argument','role':role,'tool':tool,'hist_key':key,'turn':ti,'episode_rank':episode_rank,'episode_sim':episode_sim,'context':f'tool call {tool}; argument path {key}; exact value {safe_text(val)}; full arguments {safe_text(args)[:700]}'})
    return occ

def policy_occurrences(p,d,required):
    out=[]
    def add(v,src,ctx):out.append({'value':v,'source':src,'role':'schema','tool':'','hist_key':p,'turn':-1,'episode_rank':-1,'episode_sim':0.0,'context':ctx})
    if 'default' in d:add(d['default'],'schema_default',f'schema default for parameter {p}: {d.get("description","")}')
    for v in base.desc_defaults(d.get('description','')):add(v,'description_default',f'description-implied default for parameter {p}: {d.get("description","")}')
    if isinstance(d.get('enum'),list):
        for v in d['enum']:add(v,'schema_enum',f'allowed enum value for parameter {p}: {d.get("description","")}')
    for v,src in es.policy_values(p,d,required):add(v,src,f'general schema/type policy for {p}: {d.get("description","")}')
    add(OMIT,'omit',f'operation OMIT parameter {p} when not required/relevant')
    return out

def feature_row(target,p,d,query,target_tool,occ,sim,maxturn):
    key=str(occ.get('hist_key',''));leaf=key.split('.')[-1] if key else '';tool=str(occ.get('tool',''));source=str(occ.get('source',''));ctx=str(occ.get('context',''));typ=str(d.get('type','')).lower();val=occ['value']
    return {
      'sim':float(sim),'episode_sim':float(occ.get('episode_sim',0.0)),'episode_rank':float(occ.get('episode_rank',-1)),
      'recency':float(occ.get('turn',-1)/max(1,maxturn)),'slot_exact':float(bool(leaf) and leaf.casefold()==p.casefold()),
      'slot_jacc':float(lr.jacc(p+' '+str(d.get('description','')),key)),'tool_exact':float(bool(tool) and re.sub(r'[^a-z0-9]','',tool.lower())==re.sub(r'[^a-z0-9]','',str(target_tool).lower())),
      'tool_jacc':float(lr.jacc(target_tool,tool)),'query_context_jacc':float(lr.jacc(query,ctx)),'slot_context_jacc':float(lr.jacc(p+' '+str(d.get('description','')),ctx)),
      'query_value_jacc':float(lr.jacc(query,'' if val==OMIT else val)),'value_len':float(len(str(val))) if val!=OMIT else 0.0,
      'is_omit':float(val==OMIT),'source_'+source:1.0,'role_'+str(occ.get('role','')):1.0,'type_'+typ:1.0,
      'is_default':float('default' in source),'is_policy':float(source.startswith('policy_')),'required':float(p in set())
    }

def compile_slot(enc,qa,session,qi,p,d,required):
    eps=es.episodes(session);schema=qa.get('target_tool_schema') or {};target_tool=schema.get('name','');query=str(qa.get('query',''))
    target=f'Current request: {query}. Target tool: {target_tool}. Target parameter: {p}. Meaning: {d.get("description","")}. Type: {d.get("type","")}'
    picked=es.retrieve_episodes(enc,eps,target,TOP_EPISODES)
    occ=policy_occurrences(p,d,required)
    for rank,ep,sim in picked:occ.extend(occurrence_candidates(ep,rank,sim))
    for val in base.spans(query):occ.append({'value':val,'source':'current_request_span','role':'current_request','tool':target_tool,'hist_key':'','turn':10**6,'episode_rank':-1,'episode_sim':1.0,'context':f'current request exact span {val}: {query}'})
    # Cheap lexical prefilter preserves all structured/default occurrences then best contextual spans.
    structured=[];resid=[]
    for o in occ:
        if o['source']!='content_span':structured.append(o)
        else:
            s=lr.jacc(p+' '+str(d.get('description',''))+' '+query,o['context'])
            resid.append((s,o))
    resid=[o for _,o in sorted(resid,key=lambda z:z[0],reverse=True)[:max(0,MAX_OCC-len(structured))]]
    occ=(structured+resid)[:MAX_OCC]
    texts=[o['context'] for o in occ]
    E=enc.encode([target]+texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    sims=E[1:]@E[0];maxturn=max([o.get('turn',-1) for o in occ if o.get('turn',-1)<10**6]+[1])
    feats=[]
    for i,o in enumerate(occ):
        r=feature_row(target,p,d,query,target_tool,o,float(sims[i]),maxturn);r['required']=float(p in required);feats.append(r)
    return {'qi':qi,'qa':qa,'parameter':p,'def':d,'required':p in required,'occurrences':occ,'features':feats,'picked_episode_ids':[e['source_id'] for _,e,_ in picked]}

def build(enc):
    td=Path(tempfile.gettempdir())/'mab_evidence_ptr';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:DEV_END];sessions,by=base.build_session_map(cp);slots=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        schema=qa.get('target_tool_schema') or {};params=schema.get('parameters') or {};props=params.get('properties') or {};required=set(params.get('required') or [])
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,d0 in props.items():
            s=compile_slot(enc,qa,ses,qi,p,d0 or {},required)
            gv=gold[p] if p in gold else OMIT
            s['labels']=[int((o['value']==OMIT and gv==OMIT) or (o['value']!=OMIT and gv!=OMIT and base.norm(o['value'])==base.norm(gv))) for o in s['occurrences']]
            s['gold']=gv;s['grounding']=str((gi.get(p) or {}).get('type','not_gold_slot'));slots.append(s)
    return qas,slots,missing

def choose_value(s,probs,threshold):
    # Aggregate repeated evidence occurrences by exact normalized value using max probability.
    agg={}
    for i,o in enumerate(s['occurrences']):
        key=('omit',) if o['value']==OMIT else ('v',base.norm(o['value']))
        cur=agg.get(key)
        if cur is None or float(probs[i])>cur[0]:agg[key]=(float(probs[i]),i,o)
    ranked=sorted(agg.values(),key=lambda z:z[0],reverse=True)
    if not ranked:return None,0.0,None
    conf,i,o=ranked[0]
    if conf<threshold and not s['required']:
        omit=[z for z in ranked if z[2]['value']==OMIT]
        if omit:return omit[0][2],omit[0][0],omit[0][1]
        return {'value':OMIT,'source':'threshold_omit','context':''},conf,None
    if o['value']==OMIT and s['required']:
        non=[z for z in ranked if z[2]['value']!=OMIT]
        if non:conf,i,o=non[0]
    return o,conf,i

def score(slots,clf,vec,lo,hi,threshold):
    byqa=defaultdict(list)
    for s in slots:
        if lo<=s['qi']<hi:byqa[s['qi']].append(s)
    C=P=G=tasks=exact=0;ground=defaultdict(lambda:Counter(n=0,correct=0));cov=Counter();samples=[]
    for qi,ss in sorted(byqa.items()):
        qa=ss[0]['qa'];gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pred={};dbg={};tasks+=1
        for s in ss:
            X=vec.transform(s['features']);probs=clf.predict_proba(X)[:,1];o,conf,idx=choose_value(s,probs,threshold)
            if s['parameter'] in gold:
                cov['slots']+=1;cov['has']+=int(any(base.norm(x['value'])==base.norm(gold[s['parameter']]) for x in s['occurrences'] if x['value']!=OMIT))
            if o is not None and o['value']!=OMIT:pred[s['parameter']]=base.coerce(o['value'],s['def'])
            dbg[s['parameter']]={'confidence':conf,'source':None if o is None else o.get('source'),'value':None if o is None or o['value']==OMIT else o['value'],'context':None if o is None else o.get('context','')[:220],'episodes':s['picked_episode_ids']}
        correct=0
        for p,g in gold.items():
            gt=str((gi.get(p) or {}).get('type','unknown'));ground[gt]['n']+=1
            if p in pred and base.norm(pred[p])==base.norm(g):correct+=1;ground[gt]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
        if len(samples)<6:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':dbg})
    pr=C/max(1,P);rc=C/max(1,G);f=2*pr*rc/max(1e-12,pr+rc)
    return {'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact':exact/max(1,tasks),'gold_occurrence_coverage':cov['has']/max(1,cov['slots']),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(ground.items())},'samples':samples}

def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');_,slots,missing=build(enc)
    train=[s for s in slots if s['qi']<TRAIN_END];rows=[];ys=[]
    for s in train:rows.extend(s['features']);ys.extend(s['labels'])
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(rows);clf=LogisticRegression(max_iter=1800,class_weight='balanced',C=1.0,random_state=42).fit(X,ys)
    curve=[]
    for t in [0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        r=score(slots,clf,vec,TRAIN_END,DEV_END,t);curve.append({'threshold':t,'f1':r['f1'],'precision':r['precision'],'recall':r['recall']})
    best=max(curve,key=lambda x:(x['f1'],x['precision']))['threshold'];tr=score(slots,clf,vec,0,TRAIN_END,best);va=score(slots,clf,vec,TRAIN_END,DEV_END,best)
    names=vec.get_feature_names_out();w=clf.coef_[0];coef=[{'feature':str(names[i]),'weight':float(w[i])} for i in np.argsort(-np.abs(w))[:30]]
    result={'stage':'Mem2Act evidence-preserving exact pointer router','architecture':'top-2 source_id episode scope -> occurrence-level exact values with local provenance/context -> tiny discriminative scorer -> max-over-occurrences exact value aggregation -> deterministic copy/omit; no generative LLM','split':'QA001-70 train, QA071-100 validation; QA101-400 gold sealed','missing_zero_based':missing,'training':{'occurrence_rows':len(ys),'positive_occurrences':int(sum(ys)),'schema_slots':len(train)},'selected_threshold':best,'validation_curve':curve,'train_result':tr,'validation_result':va,'top_abs_coefficients':coef,'guardrail':'Gold values label occurrence correctness only on QA001-70 and score/calibrate QA071-100. No gold string/provenance is a feature. QA source_conversation_ids are not used for episode retrieval. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_EVIDENCE_POINTER='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
