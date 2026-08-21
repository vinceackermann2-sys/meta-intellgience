from pathlib import Path
from collections import defaultdict
import json,re,tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import semantic_role_bridge_cv as rb

OUT=Path(__file__).with_name('structural_role_bridge_cv_result.json')
N=100;TOP_EPISODES=2;SEED=20260821

def jacc(a,b):
    A=set(re.findall(r'[a-z0-9]+',str(a).casefold()));B=set(re.findall(r'[a-z0-9]+',str(b).casefold()))
    return len(A&B)/max(1,len(A|B))
def canon(s):return re.sub(r'[^a-z0-9]+','',str(s).casefold())
def record_parts(ep):
    out=[];rows=ep['rows']
    for ti,t in rows:
        intent=rb.nearest_user(rows,ti);role=str(t.get('role',''));content=t.get('content','')
        if isinstance(content,(dict,list)):
            fields=rb.flatten(content);keys=', '.join(k for k,_ in fields[:30]);rec={'tool':'','intent':intent,'role':role,'keys':keys,'turn':ti,'fields':[]}
            for k,v in fields:rec['fields'].append({'field':k,'value':v,'kind':'structured_content'})
            if rec['fields']:out.append(rec)
        elif isinstance(content,str) and content.strip()[:1] in '[{':
            try:
                z=json.loads(content);fields=rb.flatten(z);keys=', '.join(k for k,_ in fields[:30]);rec={'tool':'','intent':intent,'role':role,'keys':keys,'turn':ti,'fields':[]}
                for k,v in fields:rec['fields'].append({'field':k,'value':v,'kind':'json_content'})
                if rec['fields']:out.append(rec)
            except Exception:pass
        for tc in t.get('tool_calls') or []:
            tool=rb.tool_name(tc);args=rb.parse_args(tc);fields=rb.flatten(args);keys=', '.join(k for k,_ in fields[:30]);rec={'tool':tool,'intent':intent,'role':role,'keys':keys,'turn':ti,'fields':[]}
            for k,v in fields:rec['fields'].append({'field':k,'value':v,'kind':'tool_argument'})
            if rec['fields']:out.append(rec)
    return out

def encode_cached(enc,texts,cache):
    need=[t for t in texts if t not in cache]
    if need:
        E=enc.encode(need,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        for t,e in zip(need,E):cache[t]=e
    return [cache[t] for t in texts]
def build(enc,qas,sessions,by):
    cache={};slots=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=rb.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=rb.episodes(ses);scope=[rb.episode_scope_text(e) for e in eps]
        sc=qa.get('target_tool_schema') or {};props=((sc.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            d=props.get(p) or {};query=str(qa.get('query',''));target_full=rb.target_role(qa,p,d)
            target_field=f'target field {rb.field_tokens(p)} meaning {d.get("description","")} type {d.get("type","")}'
            target_tool=f'target tool {sc.get("name","")}'
            target_query=f'current user intent {query}'
            tv,tf,ttool,tq=encode_cached(enc,[target_full,target_field,target_tool,target_query],cache)
            if scope:
                EE=np.stack(encode_cached(enc,scope,cache));order=np.argsort(-(EE@tv))[:min(TOP_EPISODES,len(eps))]
            else:order=[]
            cands=[]
            for erank,idx in enumerate(order):
                ep=eps[int(idx)];episode_sim=float(cache[scope[int(idx)]]@tv)
                for rec in record_parts(ep):
                    rec_schema=f'historical record schema tool {rec["tool"]} role {rec["role"]} fields {rec["keys"]}'
                    rec_intent=f'historical user intent {rec["intent"]}'
                    rec_tool=f'historical tool {rec["tool"]}'
                    rv,iv,tv2=encode_cached(enc,[rec_schema,rec_intent,rec_tool],cache)
                    for f in rec['fields']:
                        field_text=f'historical field role {rb.field_tokens(f["field"])} path {f["field"]} kind {f["kind"]}'
                        fv=encode_cached(enc,[field_text],cache)[0]
                        feats=np.array([
                            float(tf@fv),float(tv@rv),float(tq@iv),float(ttool@tv2),episode_sim,
                            jacc(p,f['field']),float(canon(rb.field_tokens(p))==canon(rb.field_tokens(f['field']))),
                            min(1.0,len(rec['fields'])/20.0),1.0/(1.0+erank),float(f['kind']=='tool_argument')
                        ],dtype=np.float32)
                        cands.append({'value':f['value'],'field':f['field'],'tool':rec['tool'],'features':feats,'field_vec':fv,'record_vec':rv,'intent_vec':iv,'tool_vec':tv2})
            labels=[int(rb.norm(c['value'])==rb.norm(g)) for c in cands]
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'parameter':p,'grounding':str((gi.get(p) or {}).get('type','unknown')),'gold':g,'target_vec':tv,'target_field_vec':tf,'target_query_vec':tq,'target_tool_vec':ttool,'cands':cands,'labels':labels})
    return slots,missing

def fit_pairwise(train):
    X=[];y=[]
    for s in train:
        pos=[i for i,z in enumerate(s['labels']) if z];neg=[i for i,z in enumerate(s['labels']) if not z]
        if not pos or not neg:continue
        # cap negative pairs so large records do not dominate
        for pi in pos[:2]:
            ps=s['cands'][pi]['features']
            for ni in neg[:20]:
                d=ps-s['cands'][ni]['features'];X.extend([d,-d]);y.extend([1,0])
    if not X:return None
    return LogisticRegression(max_iter=2000,C=0.1,class_weight='balanced',random_state=SEED).fit(np.asarray(X),np.asarray(y))
def score_candidate(s,c,method,clf=None):
    f=c['features']
    if method=='raw':return float(f[0])
    if method=='struct_fixed':
        # Frozen semantics + explicit structural message channels; no learned embedding projection.
        return float(.50*f[0]+.20*f[1]+.15*f[2]+.10*f[3]+.05*f[4])
    if method=='pairwise':return float(clf.decision_function(c['features'][None,:])[0])
    raise ValueError(method)
def eval_fold(slots,train_q,test_q,method):
    train=[s for s in slots if s['qi'] in train_q];test=[s for s in slots if s['qi'] in test_q];clf=fit_pairwise(train) if method=='pairwise' else None
    by=defaultdict(lambda:{'n':0,'covered':0,'top1':0,'top3':0});examples=[]
    for s in test:
        m=by[s['grounding']];m['n']+=1;pos=[i for i,z in enumerate(s['labels']) if z];m['covered']+=int(bool(pos))
        if not s['cands']:continue
        scores=np.asarray([score_candidate(s,c,method,clf) for c in s['cands']]);order=np.argsort(-scores)
        m['top1']+=int(bool(pos) and int(order[0]) in pos);m['top3']+=int(bool(pos) and any(int(i) in pos for i in order[:3]))
        if s['grounding']=='explicit' and pos and int(order[0]) not in pos and len(examples)<5:
            examples.append({'qa_id':s['qa_id'],'parameter':s['parameter'],'gold':s['gold'],'top':[{'score':float(scores[int(i)]),'field':s['cands'][int(i)]['field'],'tool':s['cands'][int(i)]['tool'],'value':s['cands'][int(i)]['value']} for i in order[:3]]})
    packed={}
    for typ,m in by.items():
        packed[typ]={'n':m['n'],'coverage':m['covered']/max(1,m['n']),'top1_all':m['top1']/max(1,m['n']),'top3_all':m['top3']/max(1,m['n']),'top1_given_covered':m['top1']/max(1,m['covered'])}
    n=sum(m['n'] for m in by.values());cov=sum(m['covered'] for m in by.values());one=sum(m['top1'] for m in by.values());three=sum(m['top3'] for m in by.values())
    packed['overall']={'n':n,'coverage':cov/max(1,n),'top1_all':one/max(1,n),'top3_all':three/max(1,n),'top1_given_covered':one/max(1,cov)}
    return packed,examples

def main():
    td=Path(tempfile.gettempdir())/'struct_role_bridge';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';rb.fetch('qa_dataset.jsonl',qp);rb.fetch('toolmem_conversation.jsonl',cp)
    qas=list(rb.load_jsonl(qp))[:N];sessions,by=rb.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');slots,missing=build(enc,qas,sessions,by)
    ids=np.array(sorted(set(s['qi'] for s in slots)));kf=KFold(5,shuffle=True,random_state=SEED);methods=['raw','struct_fixed','pairwise'];folds={m:[] for m in methods};examples={m:[] for m in methods}
    for fi,(tr,te) in enumerate(kf.split(ids)):
        trq=set(ids[tr].tolist());teq=set(ids[te].tolist())
        for m in methods:
            z,e=eval_fold(slots,trq,teq,m);folds[m].append(z);examples[m].append(e)
        print('STRUCT_ROLE_FOLD',fi,{m:folds[m][-1]['overall'] for m in methods},flush=True)
    summary={}
    for m in methods:
        summary[m]={'top1_all':float(np.mean([f['overall']['top1_all'] for f in folds[m]])),'top3_all':float(np.mean([f['overall']['top3_all'] for f in folds[m]])),'top1_given_covered':float(np.mean([f['overall']['top1_given_covered'] for f in folds[m]])),'explicit_top1_all':float(np.mean([f.get('explicit',{}).get('top1_all',0) for f in folds[m]])),'inferred_top1_all':float(np.mean([f.get('inferred',{}).get('top1_all',0) for f in folds[m]])),'coverage':float(np.mean([f['overall']['coverage'] for f in folds[m]]))}
    out={'stage':'QA001-100 development-only frozen semantic + structural record bridge','hypothesis':'preserve pretrained semantic geometry and add explicit record/intent/tool structural channels instead of learning a global semantic projection','summary':summary,'folds':folds,'examples':examples,'missing_zero_based':missing,'feature_order':['field_semantic','record_schema_semantic','intent_semantic','tool_semantic','episode_semantic','field_name_jacc','field_name_exact','record_size','episode_rank','tool_argument_kind'],'guardrail':'Candidate values are labels/executor payloads only and are never embedded or used in ranking features. QA101-400 gold is not read.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('STRUCTURAL_ROLE_BRIDGE='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
