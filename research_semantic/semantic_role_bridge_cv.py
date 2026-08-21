from pathlib import Path
from collections import defaultdict
import json, re, tempfile, urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
OUT=Path(__file__).with_name('semantic_role_bridge_cv_result.json')
N=100
TOP_EPISODES=2
SEED=20260821


def fetch(name, path):
    if not path.exists():
        urllib.request.urlretrieve(BASE+name, path)

def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def norm(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None: return 'null'
    if isinstance(v,(dict,list)):
        return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()

def tool_name(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {}
    return str((f or {}).get('name','')) if isinstance(f,dict) else ''

def parse_args(tc):
    if not isinstance(tc,dict): return {}
    f=tc.get('function') or {}; a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict): return a
    if isinstance(a,str):
        try:
            z=json.loads(a); return z if isinstance(z,dict) else {}
        except Exception: return {}
    return {}

def flatten(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            p=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)): out.extend(flatten(v,p))
            else: out.append((p,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            p=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)): out.extend(flatten(v,p))
            else: out.append((p,v))
    return out

def field_tokens(path):
    s=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',str(path))
    s=re.sub(r'[_\.\[\]0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip().lower()

def record_keys(x):
    if isinstance(x,dict): return ', '.join(map(str,list(x.keys())[:20]))
    return ''

def build_session_map(cp):
    sessions=[]; by=defaultdict(list)
    for s in load_jsonl(cp):
        i=len(sessions); sessions.append(s)
        for x in s.get('original_conversation_ids') or []:
            by[str(x)].append(i)
    return sessions,by

def find_session(qa,sessions,by):
    ids=[str(x) for x in qa.get('source_conversation_ids') or []]
    cand=None
    for x in ids:
        z=set(by.get(x,[])); cand=z if cand is None else cand&z
    if cand: return sessions[min(cand)]
    union=set()
    for x in ids: union.update(by.get(x,[]))
    return sessions[min(union)] if union else None

def episodes(session):
    groups=defaultdict(list); order=[]
    for ti,t in enumerate(session.get('turns') or []):
        sid=str(t.get('source_id') or f'unknown_{ti}')
        if sid not in groups: order.append(sid)
        groups[sid].append((ti,t))
    return [{'source_id':sid,'rows':groups[sid]} for sid in order]

def nearest_user(rows,ti):
    best=''
    for tj,t in rows:
        if tj>ti: break
        if str(t.get('role','')).lower()=='user':
            c=t.get('content','')
            if isinstance(c,(dict,list)): c=json.dumps(c,ensure_ascii=False)
            best=str(c)
    return best[:900]

def episode_scope_text(ep):
    # Value-blind episode routing: user intents + tool names + argument keys only.
    parts=[]
    for ti,t in ep['rows']:
        role=str(t.get('role','')).lower()
        if role=='user':
            c=t.get('content','')
            if isinstance(c,(dict,list)): c=json.dumps(c,ensure_ascii=False)
            parts.append('user intent: '+str(c)[:1200])
        for tc in t.get('tool_calls') or []:
            args=parse_args(tc)
            parts.append('tool '+tool_name(tc)+' argument roles '+', '.join(k for k,_ in flatten(args)[:30]))
    return '\n'.join(parts)[:5000]

def structured_occurrences(ep):
    out=[]
    rows=ep['rows']
    for ti,t in rows:
        role=str(t.get('role',''))
        intent=nearest_user(rows,ti)
        content=t.get('content','')
        if isinstance(content,(dict,list)):
            keys=record_keys(content)
            for key,val in flatten(content):
                r=(f'historical user intent: {intent}. source role: {role}. '
                   f'structured record field role: {field_tokens(key)}. field path: {key}. '
                   f'record schema keys: {keys}.')
                out.append({'value':val,'role':r,'field':key,'tool':'','turn':ti,'kind':'structured_content'})
        elif isinstance(content,str) and content.strip()[:1] in '[{':
            try:
                obj=json.loads(content); keys=record_keys(obj)
                for key,val in flatten(obj):
                    r=(f'historical user intent: {intent}. source role: {role}. '
                       f'json record field role: {field_tokens(key)}. field path: {key}. '
                       f'record schema keys: {keys}.')
                    out.append({'value':val,'role':r,'field':key,'tool':'','turn':ti,'kind':'json_content'})
            except Exception: pass
        for tc in t.get('tool_calls') or []:
            tool=tool_name(tc); args=parse_args(tc); keys=', '.join(map(str,list(args.keys())[:20]))
            for key,val in flatten(args):
                r=(f'historical user intent: {intent}. historical tool: {tool}. '
                   f'tool argument semantic role: {field_tokens(key)}. field path: {key}. '
                   f'sibling argument roles: {keys}.')
                out.append({'value':val,'role':r,'field':key,'tool':tool,'turn':ti,'kind':'tool_argument'})
    return out

def target_role(qa,p,d):
    schema=qa.get('target_tool_schema') or {}
    return (f'current user intent: {qa.get("query","")}. target tool: {schema.get("name","")}. '
            f'target semantic role: {field_tokens(p)}. parameter name: {p}. '
            f'parameter meaning: {d.get("description","")}. parameter type: {d.get("type","")}.')

def build_slots(enc,qas,sessions,by):
    slots=[]; missing=[]
    for qi,qa in enumerate(qas):
        ses=find_session(qa,sessions,by)
        if ses is None:
            missing.append(qi); continue
        eps=episodes(ses)
        scope=[episode_scope_text(e) for e in eps]
        if scope:
            EE=enc.encode(scope,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        else: EE=np.zeros((0,384),np.float32)
        schema=qa.get('target_tool_schema') or {}; props=((schema.get('parameters') or {}).get('properties') or {})
        gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            d=props.get(p) or {}; tr=target_role(qa,p,d)
            tv=enc.encode([tr],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
            if len(EE):
                order=np.argsort(-(EE@tv))[:min(TOP_EPISODES,len(eps))]
            else: order=[]
            cands=[]
            for rank,idx in enumerate(order):
                for c in structured_occurrences(eps[int(idx)]):
                    c=dict(c); c['episode_rank']=rank; cands.append(c)
            # Deduplicate by semantic address + exact payload. Payload is never embedded.
            seen=set(); uniq=[]
            for c in cands:
                k=(c['role'],norm(c['value']))
                if k in seen: continue
                seen.add(k); uniq.append(c)
            labels=[int(norm(c['value'])==norm(g)) for c in uniq]
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'parameter':p,'grounding':str((gi.get(p) or {}).get('type','unknown')),
                          'target_role':tr,'target_vec':tv,'cands':uniq,'labels':labels,'gold':g})
    # Batch encode every unique source role once.
    roles=[]; index={}
    for s in slots:
        for c in s['cands']:
            r=c['role']
            if r not in index: index[r]=len(roles); roles.append(r)
    RE=enc.encode(roles,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if roles else np.zeros((0,384),np.float32)
    for s in slots:
        s['source_vecs']=np.stack([RE[index[c['role']]] for c in s['cands']]) if s['cands'] else np.zeros((0,384),np.float32)
    return slots,missing

def cosine_rows(A,v):
    if not len(A): return np.array([])
    return A@v

def orthogonal_map(T,S):
    M=T.T@S
    U,_,Vt=np.linalg.svd(M,full_matrices=False)
    return U@Vt

def evaluate_fold(slots,train_q,test_q,method,alpha=10.0):
    train=[s for s in slots if s['qi'] in train_q and any(s['labels'])]
    test=[s for s in slots if s['qi'] in test_q]
    T=[];S=[]
    for s in train:
        pos=[i for i,y in enumerate(s['labels']) if y]
        # one positive prototype per slot prevents repeated-value records dominating training
        if not pos: continue
        sims=cosine_rows(s['source_vecs'][pos],s['target_vec'])
        j=pos[int(np.argmax(sims))]
        T.append(s['target_vec']);S.append(s['source_vecs'][j])
    T=np.asarray(T,np.float32);S=np.asarray(S,np.float32)
    ridge=None;W=None
    if len(T)>=4:
        ridge=Ridge(alpha=alpha,fit_intercept=True).fit(T,S)
        W=orthogonal_map(T,S)
    by=defaultdict(lambda:{'n':0,'covered':0,'top1':0,'top3':0,'top5':0})
    examples=[]
    for s in test:
        m=by[s['grounding']];m['n']+=1
        pos=[i for i,y in enumerate(s['labels']) if y]
        m['covered']+=int(bool(pos))
        if not s['cands']: continue
        if method=='raw' or ridge is None:
            q=s['target_vec']
        elif method=='ridge':
            q=ridge.predict(s['target_vec'][None,:])[0]
            q=q/(np.linalg.norm(q)+1e-9)
        elif method=='procrustes':
            q=s['target_vec']@W
            q=q/(np.linalg.norm(q)+1e-9)
        else: raise ValueError(method)
        scores=cosine_rows(s['source_vecs'],q);order=np.argsort(-scores)
        m['top1']+=int(bool(pos) and int(order[0]) in pos)
        m['top3']+=int(bool(pos) and any(int(i) in pos for i in order[:3]))
        m['top5']+=int(bool(pos) and any(int(i) in pos for i in order[:5]))
        if s['grounding']=='explicit' and pos and int(order[0]) not in pos and len(examples)<5:
            examples.append({'qa_id':s['qa_id'],'parameter':s['parameter'],'gold':s['gold'],
                             'top':[{'score':float(scores[int(i)]),'field':s['cands'][int(i)]['field'],'tool':s['cands'][int(i)]['tool'],'value':s['cands'][int(i)]['value']} for i in order[:3]]})
    packed={}
    for typ,m in by.items():
        n=max(1,m['n']);cov=max(1,m['covered'])
        packed[typ]={'n':m['n'],'coverage':m['covered']/n,'top1_all':m['top1']/n,'top3_all':m['top3']/n,'top5_all':m['top5']/n,'top1_given_covered':m['top1']/cov}
    alln=sum(x['n'] for x in by.values());allcov=sum(x['covered'] for x in by.values());all1=sum(x['top1'] for x in by.values());all3=sum(x['top3'] for x in by.values());all5=sum(x['top5'] for x in by.values())
    packed['overall']={'n':alln,'coverage':allcov/max(1,alln),'top1_all':all1/max(1,alln),'top3_all':all3/max(1,alln),'top5_all':all5/max(1,alln),'top1_given_covered':all1/max(1,allcov),'train_positive_slots':len(T)}
    return packed,examples

def main():
    td=Path(tempfile.gettempdir())/'semantic_role_bridge';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch('qa_dataset.jsonl',qp);fetch('toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N];sessions,by=build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    slots,missing=build_slots(enc,qas,sessions,by)
    ids=np.array(sorted(set(s['qi'] for s in slots)),dtype=int)
    kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
    methods=['raw','ridge','procrustes'];results={m:[] for m in methods};examples={m:[] for m in methods}
    for fold,(tri,tei) in enumerate(kf.split(ids)):
        tr=set(ids[tri].tolist());te=set(ids[tei].tolist())
        for m in methods:
            met,ex=evaluate_fold(slots,tr,te,m)
            results[m].append(met);examples[m].append(ex)
        print('ROLE_BRIDGE_FOLD',fold,{m:results[m][-1]['overall'] for m in methods},flush=True)
    summary={}
    for m in methods:
        summary[m]={k:float(np.mean([f['overall'][k] for f in results[m]])) for k in ['coverage','top1_all','top3_all','top5_all','top1_given_covered']}
        summary[m]['explicit_top1_all']=float(np.mean([f.get('explicit',{}).get('top1_all',0.0) for f in results[m]]))
        summary[m]['inferred_top1_all']=float(np.mean([f.get('inferred',{}).get('top1_all',0.0) for f in results[m]]))
    out={'stage':'QA001-100 development-only semantic role bridge CV','hypothesis':'learn a value-blind cross-schema role map from target tool roles to historical record roles, then dereference exact payloads','methods':summary,'folds':results,'missing_zero_based':missing,'examples':examples,'guardrail':'QA001-100 only. Candidate payload values are never embedded or used as ranking features; they are used only to label/score development candidates. Episode routing is value-blind (user intents + tool/field roles only). QA101-400 gold is not read.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('SEMANTIC_ROLE_BRIDGE='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
