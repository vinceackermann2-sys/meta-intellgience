from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
import strong_banking77 as base
import episode_scoped_router as es
import record_role_diagnostic as rr

OUT=Path(__file__).with_name('structural_schema_match_cv_result.json')
N=100
TOP_EPISODES=5

def toks(s):
    s=re.sub(r'([a-z])([A-Z])',r'\1 \2',str(s))
    return set(re.findall(r'[a-z0-9]+',s.casefold()))-{'the','a','an','of','to','for','in','on'}
def jacc(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))
def leaf(k): return re.sub(r'\[\d+\]$','',str(k).split('.')[-1])
def canon(s): return re.sub(r'[^a-z0-9]+','',str(s).casefold())
def safe(x): return json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x)
def mask_values(text,fields):
    z=str(text)
    for _,v in fields:
        sv=str(v)
        if sv: z=re.sub(re.escape(sv),'<VALUE>',z,flags=re.I)
    return z

def last_user(ep,turn):
    x=''
    for ti,t in ep['rows']:
        if ti>turn: break
        if str(t.get('role',''))=='user': x=safe(t.get('content',''))
    return x[:700]
def tdoc(qa,p,d):
    sch=qa.get('target_tool_schema') or {}
    return f"current intent {qa.get('query','')}; target tool {sch.get('name','')}; parameter role {p}; meaning {d.get('description','')}; type {d.get('type','')}"
def sdoc(r,k):
    sib=' '.join(leaf(x) for x,_ in r['fields'][:40])
    return f"historical request {r['ctx']}; source tool {r['tool']}; field role {leaf(k)}; field path {k}; sibling roles {sib}; record role {r['role']} kind {r['kind']}"
def schema_doc(qa,props):
    sch=qa.get('target_tool_schema') or {}
    return f"{qa.get('query','')} tool {sch.get('name','')} schema " + ' ; '.join(f"{p} {(d or {}).get('description','')} {(d or {}).get('type','')}" for p,d in props.items())
def index_bonus(tp,sp):
    m=re.search(r'(\d+)$',str(tp)); n=re.search(r'\[(\d+)\]$',str(sp))
    return 1.0 if m and n and int(m.group(1))-1==int(n.group(1)) else 0.0

def build(enc):
    td=Path(tempfile.gettempdir())/'struct_schema_match';td.mkdir(exist_ok=True)
    qp,cp=td/'q',td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);rows=[];missing=[];cache={}
    def E(texts):
        xs=list(texts); miss=[x for x in xs if x not in cache]
        if miss:
            uniq=list(dict.fromkeys(miss)); arr=enc.encode(uniq,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
            for x,v in zip(uniq,arr):cache[x]=v
        return np.stack([cache[x] for x in xs]) if xs else np.zeros((0,384),np.float32)
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None: missing.append(qi);rows.append({'qi':qi,'qa':qa,'params':[],'records':[]});continue
        sch=qa.get('target_tool_schema') or {};props=((sch.get('parameters') or {}).get('properties') or {});params=list(props)
        eps=es.episodes(ses); picked=es.retrieve_episodes(enc,eps,schema_doc(qa,props),TOP_EPISODES); records=[]
        for er,ep,esim in picked:
            for r0 in rr.recs(ep,er,esim):
                r=dict(r0);r['ctx']=mask_values(last_user(ep,r['turn']),r['fields']);records.append(r)
        if not params or not records:rows.append({'qi':qi,'qa':qa,'params':params,'props':props,'records':records,'packs':[]});continue
        TD=[tdoc(qa,p,props.get(p) or {}) for p in params]; TE=E(TD); packs=[]
        for ri,r in enumerate(records):
            fields=r['fields'][:64]
            if not fields:continue
            SD=[sdoc(r,k) for k,_ in fields]; SE=E(SD); sem=TE@SE.T; pair=np.array(sem,copy=True)
            for pi,p in enumerate(params):
                d=props.get(p) or {};ptype=str(d.get('type','')).lower()
                for fi,(k,v) in enumerate(fields):
                    lf=leaf(k)
                    pair[pi,fi]=0.72*sem[pi,fi]+0.18*jacc(p+' '+str(d.get('description','')),lf+' '+k)+0.07*float(canon(p)==canon(lf) and bool(lf))+0.03*index_bonus(p,k)
                    # Weak value-blind structural type compatibility from Python payload type only.
                    if ptype in {'integer','number'} and isinstance(v,(int,float)) and not isinstance(v,bool):pair[pi,fi]+=0.04
                    if ptype=='boolean' and isinstance(v,bool):pair[pi,fi]+=0.04
                    if ptype=='string' and isinstance(v,str):pair[pi,fi]+=0.02
            # Hungarian assignment with dummy columns to allow unmatched target slots.
            nt,nf=pair.shape; aug=np.concatenate([pair,np.zeros((nt,nt),np.float32)],axis=1)
            rridx,ccidx=linear_sum_assignment(-aug)
            assigned={int(a):int(b) for a,b in zip(rridx,ccidx) if int(b)<nf and aug[int(a),int(b)]>0}
            positive=[float(aug[a,b]) for a,b in zip(rridx,ccidx) if aug[a,b]>0]
            coherence=float(np.mean(positive)) if positive else -1.0
            coverage=float(len(assigned)/max(1,nt))
            packs.append({'ri':ri,'fields':fields,'pair':pair,'assigned':assigned,'coherence':coherence,'coverage':coverage,'episode_sim':float(r.get('episode_sim',0.0)),'record':r})
        rows.append({'qi':qi,'qa':qa,'params':params,'props':props,'records':records,'packs':packs})
        if qi%10==0:print('STRUCT_MATCH_BUILD',qi,'records',len(records),'cache',len(cache),flush=True)
    return rows,missing,len(cache)

def predict(row,gamma,assign_bonus,top_records):
    params=row['params']; packs=row.get('packs',[]);out={}
    ranked=sorted(packs,key=lambda x:x['coherence']+0.08*x['coverage']+0.05*x['episode_sim'],reverse=True)[:top_records]
    for pi,p in enumerate(params):
        best=None
        for pack in ranked:
            rs=pack['coherence']+0.08*pack['coverage']+0.05*pack['episode_sim']
            for fi,(k,v) in enumerate(pack['fields']):
                s=float(pack['pair'][pi,fi])+gamma*rs+assign_bonus*float(pack['assigned'].get(pi)==fi)
                if best is None or s>best[0]:best=(s,v,k,pack['ri'])
        if best is not None:out[p]={'value':best[1],'score':best[0],'field':best[2],'record':best[3]}
    return out

def eval_rows(rows,ids,hyp):
    by=defaultdict(lambda:Counter(n=0,correct=0));samples=[]
    gamma,ab,tr=hyp
    for r in rows:
        if r['qi'] not in ids:continue
        qa=r['qa'];pr=predict(r,gamma,ab,tr);gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));c=by[typ];c['n']+=1;ok=p in pr and base.norm(pr[p]['value'])==base.norm(g);c['correct']+=int(ok)
            if typ=='explicit' and not ok and len(samples)<8:samples.append({'qa_id':qa.get('qa_id'),'p':p,'gold':g,'pred':pr.get(p)})
    return {k:{'n':c['n'],'accuracy':c['correct']/max(1,c['n'])} for k,c in by.items()},samples

def obj(m):return 0.75*m.get('explicit',{}).get('accuracy',0)+0.25*m.get('inferred',{}).get('accuracy',0)

def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rows,missing,ncache=build(enc);idx=np.arange(N);folds=list(KFold(n_splits=5,shuffle=True,random_state=20260820).split(idx))
    hyps=[(g,a,t) for g in (0.0,0.1,0.25,0.5) for a in (0.0,0.05,0.1,0.2) for t in (1,2,3,5)]
    grid=[]
    for h in hyps:
        mets=[]
        for _,va in folds:
            m,_=eval_rows(rows,set(idx[va].tolist()),h);mets.append(m)
        o=float(np.mean([obj(m) for m in mets]));grid.append((o,h,mets))
    o,h,mets=max(grid,key=lambda z:z[0]);full,samples=eval_rows(rows,set(range(N)),h)
    result={'stage':'Structural schema matching diagnostic','split':'QA001-100 development-only 5-fold QA CV; QA101-400 gold sealed','architecture':'top-5 historical episodes -> structured records -> value-masked whole-schema/field embeddings -> one-to-one Hungarian target-slot/source-field assignment -> record coherence -> exact dereference; payload values never enter semantic matching','selected':{'objective':o,'gamma_record':h[0],'assignment_bonus':h[1],'top_records':h[2],'folds':mets},'all_dev':full,'samples':samples,'missing_zero_based':missing,'embedding_cache_entries':ncache,'guardrail':'Gold is scoring only. Hyperparameters selected on QA001-100 development CV. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('STRUCTURAL_SCHEMA_MATCH='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
