from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile, math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import semantic_property_ingest_oracle as sw
import semantic_concept_ingest_oracle as sc
import mem2act_repaired_sessions as repair
import gatea_source_selector as ss

OUT = Path(__file__).with_name('semantic_world_binding_cv_result.json')
N = 100
TOPK = 96
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

class CachedEncoder:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        self.cache = {}
    def encode(self, texts):
        one = isinstance(texts, str)
        xs = [texts] if one else list(texts)
        miss = [str(x) for x in xs if str(x) not in self.cache]
        if miss:
            uniq = list(dict.fromkeys(miss))
            arr = self.model.encode(uniq, batch_size=64, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            if arr.ndim == 1: arr = arr[None, :]
            for x, v in zip(uniq, arr): self.cache[x] = v
        out = np.stack([self.cache[str(x)] for x in xs])
        return out[0] if one else out

def canon(s): return re.sub(r'[^a-z0-9]+', '', str(s).casefold())
def toks(s): return set(re.findall(r'[a-z0-9]+', str(s).casefold()))
def jacc(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))
def leaf(k):
    x=str(k).split('.')[-1]; return re.sub(r'\[\d+\]$','',x)
def mask(text, value):
    s=str(text); v=str(value)
    if not v: return s
    return re.sub(re.escape(v), '<VALUE>', s, flags=re.I)
def squash(s): return re.sub(r'\s+',' ',str(s)).strip()

def last_user_before(ep, turn):
    text=''
    for ti,t in ep['rows']:
        if ti>turn: break
        if str(t.get('role',''))=='user':
            c=t.get('content','')
            if isinstance(c,(dict,list)): c=json.dumps(c,ensure_ascii=False)
            text=str(c)
    return text[:900]

def variant_ops(v,p,d):
    rows=[(v,'identity')]; seen={base.norm(v)}; s=str(v).strip(); meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    for op in ss.allowed_ops(v,p,d):
        z=ss.transform(v,op,d)
        k=base.norm(z) if z is not None else ''
        if z is not None and k and k not in seen: seen.add(k); rows.append((z,'schema:'+op))
    if any(x in meta for x in ['state','province','region','location','area']):
        for name,code in sw.US_STATES.items():
            if s.casefold()==name.casefold() and base.norm(code) not in seen:
                seen.add(base.norm(code)); rows.append((code,'ontology:state_code'))
            if s.upper()==code and base.norm(name) not in seen:
                seen.add(base.norm(name)); rows.append((name,'ontology:state_name'))
    for label,lookup,attrs in [('country',sw.pycountry.countries,('alpha_2','alpha_3','name')),('language',sw.pycountry.languages,('alpha_2','alpha_3','name'))]:
        try:
            obj=lookup.lookup(s)
            for a in attrs:
                z=getattr(obj,a,None); k=base.norm(z) if z else ''
                if z and k not in seen: seen.add(k); rows.append((z,f'ontology:{label}_{a}'))
        except Exception: pass
    try:
        if re.fullmatch(r'#[0-9A-Fa-f]{6}',s): z=sw.webcolors.hex_to_name(s.lower()); op='ontology:color_name'
        else: z=sw.webcolors.name_to_hex(s); op='ontology:color_hex'
        k=base.norm(z)
        if k not in seen: seen.add(k); rows.append((z,op))
    except Exception: pass
    try:
        f=float(s.replace(',',''))
        if f.is_integer():
            for z,op in [(int(f),'numeric:int'),(str(int(f)),'numeric:int_string'),(f'{f:.1f}','numeric:one_decimal')]:
                k=base.norm(z)
                if k not in seen: seen.add(k); rows.append((z,op))
    except Exception: pass
    return rows

def compile_world(session):
    """Query-independent, value-masked addressable world state with exact payloads kept separately."""
    out=[]
    if session is None: return out
    for ep_rank, ep in enumerate(es.episodes(session)):
        sid=str(ep.get('source_id',''))
        for ti,t in ep['rows']:
            intent=last_user_before(ep,ti)
            role=str(t.get('role',''))
            recs=sw.record_values(t)
            for rec_i,rec in enumerate(recs):
                sibling=', '.join([leaf(k) for k,_ in rec][:24])
                tool=''
                for tc in t.get('tool_calls') or []:
                    try:
                        fn=tc.get('function') or {}; tool=str(fn.get('name',''))
                    except Exception: pass
                for key,v in rec:
                    address=squash(f"historical intent {mask(intent,v)}; provenance episode {sid}; role {role}; tool {tool}; field role {leaf(key)}; field path {key}; sibling roles {sibling}; structured record {rec_i}")
                    out.append({'value':v,'src':'structured','field':key,'tool':tool,'turn':ti,'episode':ep_rank,'intent':mask(intent,v),'address':address})
            content=t.get('content','')
            if isinstance(content,str):
                for v,kind in sw.generic_entities(content):
                    ctx=mask(content[:1000],v)
                    out.append({'value':v,'src':'entity:'+kind,'field':'text_entity','tool':'','turn':ti,'episode':ep_rank,'intent':mask(intent,v),'address':squash(f"historical intent {mask(intent,v)}; provenance episode {sid}; role {role}; semantic entity kind {kind}; local context {ctx}")})
                if role=='user':
                    for v,kind in sc.concepts(content):
                        ctx=mask(content[:1000],v)
                        out.append({'value':v,'src':kind,'field':'semantic_concept','tool':'','turn':ti,'episode':ep_rank,'intent':mask(intent,v),'address':squash(f"historical intent {mask(intent,v)}; provenance episode {sid}; user concept kind {kind}; local context {ctx}")})
    # Dedup only exact address+payload; keep multiple provenance paths for same value.
    uniq=[];seen=set()
    for c in out:
        k=(base.norm(c['value']),c['address'])
        if k not in seen: seen.add(k); uniq.append(c)
    return uniq

def target_text(qa,p,d):
    sch=qa.get('target_tool_schema') or {}
    return squash(f"current request {qa.get('query','')}; target tool {sch.get('name','')}; target semantic role {p}; meaning {d.get('description','')}; type {d.get('type','')}")

def expand_slot(world,p,d):
    out=[]; seen=set()
    for c in world:
        for z,op in variant_ops(c['value'],p,d):
            k=(base.norm(z),c['address'],op)
            if not k[0] or k in seen: continue
            seen.add(k)
            q=dict(c); q['value']=z; q['op']=op; out.append(q)
    return out

def build(enc):
    td=Path(tempfile.gettempdir())/'semantic_world_binding';td.mkdir(exist_ok=True); qp=td/'q'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp); qas=list(base.load_jsonl(qp))[:N]
    repaired,report=repair.build(); slots=[]; world_sizes=[]
    for qi,qa in enumerate(qas):
        ses=(repaired.get(qa.get('qa_id')) or {}).get('session'); world=compile_world(ses); world_sizes.append(len(world))
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            d=props.get(p) or {}; target=target_text(qa,p,d); cands=expand_slot(world,p,d)
            if not cands:
                slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'p':p,'g':g,'typ':str((gi.get(p) or {}).get('type','unknown')),'target':target,'cands':[],'feats':[],'labels':[]}); continue
            tvec=enc.encode(target); avec=enc.encode([c['address'] for c in cands]); sims=avec@tvec
            # Keep a broad value-blind shortlist. Candidate value never influences the shortlist.
            order=np.argsort(-sims)[:min(TOPK,len(cands))]
            maxturn=max([int(c.get('turn',-1)) for c in cands]+[1])
            feats=[]; labels=[]; kept=[]
            for ii in order:
                i=int(ii); c=cands[i]; addr=c['address']; f=leaf(c.get('field','')); tool=str(c.get('tool','')); op=str(c.get('op','identity'))
                feats.append({
                    'semantic':float(sims[i]),
                    'query_intent_sim':float(enc.encode(str(qa.get('query','')))@enc.encode(str(c.get('intent','')))) if c.get('intent') else 0.0,
                    'param_field_exact':float(bool(f) and canon(f)==canon(p)),
                    'param_field_jacc':jacc(p,f),
                    'desc_field_jacc':jacc(str(d.get('description','')),f+' '+str(c.get('field',''))),
                    'tool_exact':float(bool(tool) and canon(tool)==canon((qa.get('target_tool_schema') or {}).get('name',''))),
                    'tool_jacc':jacc(tool,(qa.get('target_tool_schema') or {}).get('name','')),
                    'query_address_jacc':jacc(str(qa.get('query','')),addr),
                    'description_address_jacc':jacc(str(d.get('description','')),addr),
                    'recency':float(int(c.get('turn',-1))/max(1,maxturn)),
                    'episode_recency':float(c.get('episode',0)),
                    'field_depth':float(str(c.get('field','')).count('.')+str(c.get('field','')).count('[')),
                    'src:'+str(c.get('src','unknown')):1.0,
                    'op:'+op:1.0,
                    'type:'+str(d.get('type','')).lower():1.0,
                })
                labels.append(int(base.norm(c['value'])==base.norm(g))); kept.append(c)
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'p':p,'g':g,'typ':str((gi.get(p) or {}).get('type','unknown')),'target':target,'cands':kept,'feats':feats,'labels':labels})
        if qi%10==0: print('SWM_BIND_BUILD',qi,'world',len(world),'cache',len(enc.cache),flush=True)
    return slots,repair_report(report),world_sizes

def repair_report(r): return r

def fit_pairwise(slots,train_ids,C):
    rows=[]; ys=[]; groups=0
    for s in slots:
        if s['qi'] not in train_ids or not any(s['labels']): continue
        pos=[i for i,y in enumerate(s['labels']) if y]; neg=[i for i,y in enumerate(s['labels']) if not y]
        if not pos or not neg: continue
        groups+=1
        # Pair up to 6 positive addresses against hardest 24 negatives by semantic score.
        pos=pos[:6]; neg=neg[:24]
        for pi in pos:
            for ni in neg:
                a=s['feats'][pi]; b=s['feats'][ni]; keys=set(a)|set(b)
                rows.append({k:a.get(k,0.0)-b.get(k,0.0) for k in keys}); ys.append(1)
                rows.append({k:b.get(k,0.0)-a.get(k,0.0) for k in keys}); ys.append(0)
    if not rows: return None,None,0,0
    vec=DictVectorizer(sparse=True); X=vec.fit_transform(rows)
    clf=LogisticRegression(max_iter=2000,C=C,class_weight='balanced',random_state=20260820).fit(X,ys)
    return vec,clf,len(rows),groups

def cand_score(vec,clf,feats):
    if vec is None: return np.asarray([f.get('semantic',0.0) for f in feats],np.float32)
    X=vec.transform(feats); return clf.decision_function(X)

def evaluate(slots,ids,vec,clf):
    by=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0)); examples=[]
    for s in slots:
        if s['qi'] not in ids: continue
        c=by[s['typ']]; c['n']+=1
        if not s['cands']: continue
        pos=[i for i,y in enumerate(s['labels']) if y]; c['covered']+=int(bool(pos))
        score=cand_score(vec,clf,s['feats']); order=np.argsort(-score)
        c['top1']+=int(bool(pos) and int(order[0]) in pos); c['top3']+=int(bool(pos) and any(int(i) in pos for i in order[:3])); c['top5']+=int(bool(pos) and any(int(i) in pos for i in order[:5]))
        if len(examples)<12 and pos and int(order[0]) not in pos:
            examples.append({'qa_id':s['qa_id'],'parameter':s['p'],'grounding':s['typ'],'gold':s['g'],'top':[{'score':float(score[int(i)]),'src':s['cands'][int(i)]['src'],'field':s['cands'][int(i)]['field'],'op':s['cands'][int(i)]['op'],'value':s['cands'][int(i)]['value'],'address':s['cands'][int(i)]['address'][:220]} for i in order[:3]]})
    packed={}
    for k,c in by.items():
        n=max(1,c['n']);cov=max(1,c['covered']);packed[k]={'n':c['n'],'candidate_coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov}
    return packed,examples

def objective(r):
    e=r.get('explicit',{}).get('top1_all',0.0); i=r.get('inferred',{}).get('top1_all',0.0)
    return 0.7*e+0.3*i

def main():
    enc=CachedEncoder(); slots,report,world_sizes=build(enc); idx=np.arange(N); folds=list(KFold(n_splits=5,shuffle=True,random_state=20260820).split(idx)); grid=[]
    for C in C_GRID:
        fr=[];train_rows=[];groups=[]
        for tr,va in folds:
            vec,clf,nr,ng=fit_pairwise(slots,set(idx[tr].tolist()),C);r,_=evaluate(slots,set(idx[va].tolist()),vec,clf);fr.append(r);train_rows.append(nr);groups.append(ng)
        obj=float(np.mean([objective(r) for r in fr]));grid.append((obj,C,fr,train_rows,groups));print('SWM_BIND_CV',C,obj,flush=True)
    obj,C,fr,train_rows,groups=max(grid,key=lambda x:x[0]);vec,clf,nr,ng=fit_pairwise(slots,set(range(N)),C);full,examples=evaluate(slots,set(range(N)),vec,clf)
    result={'stage':'SWM-B persistent semantic world-state value-blind binding CV','split':'QA001-100 development-only grouped 5-fold CV; QA101-400 gold remains sealed','architecture':'answer-blind repaired history -> persistent structured/entity/concept world state with provenance -> deterministic schema transforms -> value-blind semantic shortlist -> within-slot pairwise address/operator ranking -> exact dereference','selected':{'C':C,'mean_cv_weighted_explicit_inferred':obj,'folds':fr,'pairwise_rows_per_fold':train_rows,'covered_groups_per_fold':groups},'all_dev_refit':full,'examples':examples,'world_size':{'mean':float(np.mean(world_sizes)),'median':float(np.median(world_sizes)),'max':int(max(world_sizes))},'embedding_cache_entries':len(enc.cache),'passes_SWM_B_on_refit_only':full.get('explicit',{}).get('top1_all',0)>=0.50 and full.get('inferred',{}).get('top1_all',0)>=0.25,'guardrail':'Candidate values are executor payloads and development correctness labels only; no value text is used in address embeddings/features or shortlist ranking. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('SEMANTIC_WORLD_BINDING='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
