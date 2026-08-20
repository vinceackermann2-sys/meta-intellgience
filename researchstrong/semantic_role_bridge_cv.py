from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
import gatea_source_selector as ss
import typed_span_diagnostic as ts

OUT=Path(__file__).with_name('semantic_role_bridge_cv_result.json')
N=100
TOP_EPISODES=5
ALPHAS=[1e-2,1e-1,1.0,10.0,100.0]

class CachedEncoder:
    def __init__(self,model): self.model=model; self.cache={}
    def encode(self,texts,**kw):
        one=isinstance(texts,str); xs=[texts] if one else list(texts)
        norm=bool(kw.get('normalize_embeddings',False)); keys=[(str(x),norm) for x in xs]
        miss=[]
        for x,k in zip(xs,keys):
            if k not in self.cache: miss.append(str(x))
        if miss:
            uniq=list(dict.fromkeys(miss)); arr=np.asarray(self.model.encode(uniq,**kw),dtype=np.float32)
            if arr.ndim==1: arr=arr[None,:]
            for x,v in zip(uniq,arr): self.cache[(x,norm)]=np.asarray(v,dtype=np.float32)
        out=np.stack([self.cache[k] for k in keys])
        return out[0] if one else out

def squash(s): return re.sub(r'\s+',' ',str(s)).strip()
def leaf(k):
    x=str(k).split('.')[-1]
    return re.sub(r'\[\d+\]$','',x)

def target_role(qa,p,d):
    sc=qa.get('target_tool_schema') or {}
    return squash(f"tool {sc.get('name','')}; semantic role {p}; meaning {d.get('description','')}; type {d.get('type','')}")

def episode_user_context(ep,turn):
    best=''
    for ti,t in ep['rows']:
        if ti>turn: break
        if str(t.get('role',''))=='user': best=squash(t.get('content',''))
    return best[:700]

def source_role(c,ep,op='identity'):
    key=str(c.get('key','')); tool=str(c.get('tool','')); kind=str(c.get('kind',''))
    intent=episode_user_context(ep,int(c.get('turn',-1)))
    return squash(f"historical intent {intent}; source tool {tool}; field role {leaf(key)}; full field path {key}; record kind {kind}; transform {op}")

def admissible_text_kind(p,d,k):
    m=(str(p)+' '+str((d or {}).get('description',''))+' '+str((d or {}).get('type',''))).lower()
    if any(x in m for x in ['url','link','uri','website']): return k=='url'
    if 'email' in m: return k=='email'
    if 'color' in m: return k in {'color_name','hex_color','quoted','entity_phrase'}
    if any(x in m for x in ['country','state','province','city','location','region']): return k in {'quoted','entity_phrase','code','short_code'}
    if any(x in m for x in ['symbol','ticker','pair','currency','code','formtype']): return k in {'quoted','entity_phrase','code','short_code'}
    if any(x in m for x in ['id','identifier','name','assignee','domain','component','book','text','keyword','search','query','q']): return k in {'quoted','entity_phrase','named_id','id','username','code','short_code'}
    return k in {'quoted','entity_phrase','url','email','date','number','code','short_code'}

def slot_candidates(enc,qa,session,p,d):
    if session is None: return []
    target=f"{qa.get('query','')} {target_role(qa,p,d)}"
    picked=es.retrieve_episodes(enc,es.episodes(session),target,TOP_EPISODES)
    out=[]
    for er,ep,esim in picked:
        # Structured values: every output/argument field is a world-model property candidate.
        for c in af.occurrences(ep,er,esim):
            if c['kind']=='text_span': continue
            for op in ss.allowed_ops(c['value'],p,d):
                z=ss.transform(c['value'],op,d)
                if z is None: continue
                out.append({'value':z,'role':source_role(c,ep,op),'episode_rank':er,'episode_sim':float(esim),'src':'structured','op':op,'field':str(c.get('key',''))})
        # Natural-language entity/property candidates compiled from episode text.
        for ti,t in ep['rows']:
            content=t.get('content','')
            if not isinstance(content,str): continue
            for sp in ts.typed_spans(content,p,d):
                if not admissible_text_kind(p,d,sp['kind']): continue
                pseudo={'key':'text_entity','tool':'','kind':sp['kind'],'turn':ti}
                for op in ss.allowed_ops(sp['value'],p,d):
                    z=ss.transform(sp['value'],op,d)
                    if z is None: continue
                    role=squash(f"historical intent {episode_user_context(ep,ti)}; natural-language semantic entity; kind {sp['kind']}; transform {op}; target-independent local role {sp.get('address','')[:350]}")
                    # Candidate value itself is masked from the role string.
                    role=re.sub(re.escape(str(sp['value'])), '<VALUE>', role, flags=re.I)
                    out.append({'value':z,'role':role,'episode_rank':er,'episode_sim':float(esim),'src':'text','op':op,'field':'text_entity'})
    # Deduplicate only identical output + semantic role; values remain executor payloads, never model features.
    uniq=[]; seen=set()
    for c in out:
        k=(base.norm(c['value']),c['role'])
        if k not in seen: seen.add(k); uniq.append(c)
    return uniq

def build(enc):
    td=Path(tempfile.gettempdir())/'semantic_role_bridge'; td.mkdir(exist_ok=True)
    qp=td/'q'; cp=td/'c'; base.fetch(base.BASE+'qa_dataset.jsonl',qp); base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N]; sessions,by=base.build_session_map(cp); rows=[]; missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None: missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        slots=[]
        for p,g in gold.items():
            d=props.get(p) or {}; tr=target_role(qa,p,d); cands=slot_candidates(enc,qa,ses,p,d)
            if cands:
                E=enc.encode([tr]+[c['role'] for c in cands],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
                y=E[0]; X=E[1:]
            else:
                y=enc.encode(tr,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32); X=np.zeros((0,len(y)),np.float32)
            labels=np.asarray([int(base.norm(c['value'])==base.norm(g)) for c in cands],dtype=np.int32)
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'p':p,'g':g,'grounding':str((gi.get(p) or {}).get('type','unknown')),'target_vec':y,'source_vecs':X,'cands':cands,'labels':labels})
        rows.append(slots)
        if qi%10==0: print('ROLE_BRIDGE_BUILD',qi,'cache',len(enc.cache),flush=True)
    return [s for r in rows for s in r],missing

def fit_map(slots,train_ids,alpha):
    xs=[]; ys=[]
    for s in slots:
        if s['qi'] not in train_ids: continue
        pos=np.where(s['labels']>0)[0]
        for i in pos[:4]: xs.append(s['source_vecs'][i]); ys.append(s['target_vec'])
    if not xs: return None,0
    X=np.asarray(xs,np.float32); Y=np.asarray(ys,np.float32)
    model=Ridge(alpha=alpha,fit_intercept=True).fit(X,Y)
    return model,len(X)

def score(slots,ids,model,blend):
    by=defaultdict(lambda:Counter(n=0,covered=0,base1=0,mapped1=0,top3=0,top5=0)); examples=[]
    for s in slots:
        if s['qi'] not in ids: continue
        m=by[s['grounding']]; m['n']+=1
        pos=np.where(s['labels']>0)[0]
        if len(pos)==0: continue
        m['covered']+=1
        X=s['source_vecs']; y=s['target_vec']
        base_score=X@y
        base_i=int(np.argmax(base_score)); m['base1']+=int(base_i in pos)
        if model is None: mapped_score=base_score
        else:
            Z=np.asarray(model.predict(X),np.float32)
            Z/=np.maximum(1e-8,np.linalg.norm(Z,axis=1,keepdims=True))
            mapped_score=Z@y
        final=(1.0-blend)*base_score+blend*mapped_score
        order=np.argsort(-final); m['mapped1']+=int(int(order[0]) in pos); m['top3']+=int(any(int(i) in pos for i in order[:3])); m['top5']+=int(any(int(i) in pos for i in order[:5]))
        if len(examples)<12 and int(order[0]) not in pos:
            examples.append({'qa_id':s['qa_id'],'parameter':s['p'],'grounding':s['grounding'],'gold':s['g'],'top':[{'score':float(final[int(i)]),'src':s['cands'][int(i)]['src'],'field':s['cands'][int(i)]['field'],'op':s['cands'][int(i)]['op'],'value':s['cands'][int(i)]['value'],'role':s['cands'][int(i)]['role'][:220]} for i in order[:3]]})
    packed={}
    for k,c in by.items():
        n=max(1,c['n']); cov=max(1,c['covered'])
        packed[k]={'n':c['n'],'candidate_coverage':c['covered']/n,'zero_shot_top1_all':c['base1']/n,'aligned_top1_all':c['mapped1']/n,'aligned_top3_all':c['top3']/n,'aligned_top5_all':c['top5']/n,'aligned_top1_given_covered':c['mapped1']/cov}
    return packed,examples

def objective(res):
    # Main objective is explicit+inferred top1 across present slots; defaults are reported but are not the role-bridge target.
    vals=[]
    for k in ('explicit','inferred'):
        if k in res: vals.append(res[k]['aligned_top1_all'])
    return float(np.mean(vals)) if vals else 0.0

def main():
    enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'))
    slots,missing=build(enc); idx=np.arange(N); folds=list(KFold(n_splits=5,shuffle=True,random_state=20260820).split(idx))
    grid=[]
    for alpha in ALPHAS:
        for blend in (0.25,0.5,0.75,1.0):
            fr=[]; pairs=[]
            for tr,va in folds:
                model,npairs=fit_map(slots,set(idx[tr].tolist()),alpha); r,_=score(slots,set(idx[va].tolist()),model,blend); fr.append(r); pairs.append(npairs)
            obj=float(np.mean([objective(r) for r in fr])); grid.append((obj,alpha,blend,fr,pairs)); print('ROLE_BRIDGE_GRID',alpha,blend,obj,flush=True)
    obj,alpha,blend,fr,pairs=max(grid,key=lambda z:z[0]); model,npairs=fit_map(slots,set(range(N)),alpha); full,examples=score(slots,set(range(N)),model,blend)
    result={'stage':'Semantic World Model v0: cross-schema value-blind semantic role alignment','split':'QA001-100 development-only 5-fold CV; QA101-400 gold remains sealed','architecture':'retrieve top-5 source_id episodes -> compile structured fields + typed natural-language entities into value-masked semantic roles -> learn source-role to target-role embedding map from development correspondences -> dereference/transform exact values only after role selection','selected':{'alpha':alpha,'blend':blend,'mean_cv_explicit_inferred_top1':obj,'folds':fr,'positive_role_pairs_per_fold':pairs},'all_dev_refit':full,'examples':examples,'missing_zero_based':missing,'embedding_cache_entries':len(enc.cache),'guardrail':'Candidate values are never role/ranking features. Gold values on QA001-100 provide development alignment labels/scoring only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('SEMANTIC_ROLE_BRIDGE='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
