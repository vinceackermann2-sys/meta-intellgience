from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile,numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import record_role_diagnostic as rr

OUT=Path(__file__).with_name('joint_schema_alignment_result.json')
TR=70;DEV=100;K=2

def safe(x):return json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x)
def norm_tokens(s):
 s=re.sub(r'([a-z])([A-Z])',r'\1 \2',str(s));return [x for x in re.findall(r'[a-z0-9]+',s.lower()) if x not in {'the','a','an','of','to','for'}]
def leaf(path):return re.sub(r'\[\d+\]$','',str(path).split('.')[-1])
def jacc(a,b):
 A=set(norm_tokens(a));B=set(norm_tokens(b));return len(A&B)/max(1,len(A|B))
def index_bonus(tp,sp):
 m=re.search(r'(\d+)$',str(tp));n=re.search(r'\[(\d+)\]$',str(sp))
 return 1.0 if m and n and int(m.group(1))-1==int(n.group(1)) else 0.0

def last_user(ep,turn):
 rows=[(ti,t) for ti,t in ep['rows'] if ti<=turn and str(t.get('role',''))=='user']
 return safe(rows[-1][1].get('content','')) if rows else ''
def mask_all(text,fields):
 s=str(text)
 for _,v in fields:
  q=str(v)
  if q:s=re.sub(re.escape(q),'<VALUE>',s,flags=re.I)
 return s

def target_desc(qa,p,d):
 sch=qa.get('target_tool_schema') or {}
 return f"current request {qa.get('query','')} target tool {sch.get('name','')} target parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
def schema_desc(qa,props):
 sch=qa.get('target_tool_schema') or {};parts=[]
 for p,d in props.items():parts.append(f"{p}: {(d or {}).get('description','')} type {(d or {}).get('type','')}")
 return f"current request {qa.get('query','')} target tool {sch.get('name','')} target schema {' ; '.join(parts)}"
def source_field_doc(r,k):return f"source tool {r['tool']} source field path {k} source leaf {leaf(k)} sibling paths {' '.join(x for x,_ in r['fields'][:30])}"
def source_record_doc(r,ctx):return f"historical request {ctx} role {r['role']} kind {r['kind']} tool {r['tool']} paths {' '.join(k for k,_ in r['fields'][:30])} shape {json.dumps(rr.shape(r['obj']),ensure_ascii=False)}"
def emb(enc,texts):return enc.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)

def qa_pack(enc,qa,ses):
 sch=qa.get('target_tool_schema') or {};props=((sch.get('parameters') or {}).get('properties') or {});params=list(props);eps=es.episodes(ses);qall=schema_desc(qa,props)
 picked=es.retrieve_episodes(enc,eps,qall,K);records=[]
 for er,ep,sm in picked:
  for r in rr.recs(ep,er,sm):
   r=dict(r);r['request_ctx']=mask_all(last_user(ep,r['turn']),r['fields']);records.append(r)
 if not records:return params,props,records,[],[]
 tdocs=[target_desc(qa,p,props.get(p) or {}) for p in params];TE=emb(enc,tdocs)
 all_fdocs=[];owners=[]
 for ri,r in enumerate(records):
  for fi,(k,v) in enumerate(r['fields']):all_fdocs.append(source_field_doc(r,k));owners.append((ri,fi,k,v))
 FE=emb(enc,all_fdocs) if all_fdocs else np.zeros((0,TE.shape[1]),np.float32);sem=TE@FE.T if len(FE) else np.zeros((len(params),0),np.float32)
 role=np.array(sem,copy=True)
 for pi,p in enumerate(params):
  for ci,(_,_,k,_) in enumerate(owners):role[pi,ci]=0.72*sem[pi,ci]+0.23*jacc(p,leaf(k))+0.05*index_bonus(p,k)
 RE=emb(enc,[qall]+[source_record_doc(r,r['request_ctx']) for r in records]);record_sem=RE[1:]@RE[0]
 fits=[]
 for ri,r in enumerate(records):
  cols=[ci for ci,(rj,_,_,_) in enumerate(owners) if rj==ri]
  fits.append(float(np.mean([np.max(role[pi,cols]) if cols else -1 for pi in range(len(params))])))
 return params,props,records,owners,(role,np.array(record_sem),np.array(fits))

def predict(pack,alpha,beta,topr):
 params,props,records,owners,mats=pack
 if not records or not owners or mats==[]:return {}
 role,record_sem,fits=mats;rs=alpha*record_sem+(1-alpha)*fits;top_records=list(np.argsort(-rs)[:topr]);pred={}
 for pi,p in enumerate(params):
  best=None
  for ci,(ri,fi,k,v) in enumerate(owners):
   if ri not in top_records:continue
   score=float(role[pi,ci]+beta*rs[ri])
   if best is None or score>best[0]:best=(score,v,ri,k)
  if best is not None:pred[p]={'value':best[1],'score':best[0],'record':best[2],'field':best[3]}
 return pred

def main():
 td=Path(tempfile.gettempdir())/'mab_joint_schema';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
 qas=list(base.load_jsonl(qp))[:DEV];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rows=[];missing=[]
 for qi,qa in enumerate(qas):
  ses=base.find_session(qa,sessions,by)
  if ses is None:missing.append(qi);rows.append((qi,qa,None));continue
  rows.append((qi,qa,qa_pack(enc,qa,ses)))
 # Tune only structural mixing parameters on QA001-70. Objective: exact explicit-slot accuracy.
 grid=[]
 for a in [0.0,0.25,0.5,0.75,1.0]:
  for b in [0.0,0.1,0.25,0.5]:
   for tr in [1,2,3]:
    hit=tot=0
    for qi,qa,pack in rows:
     if qi>=TR or pack is None:continue
     gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pr=predict(pack,a,b,tr)
     for p,g in gold.items():
      if str((gi.get(p) or {}).get('type','unknown'))!='explicit':continue
      tot+=1;hit+=int(p in pr and base.norm(pr[p]['value'])==base.norm(g))
    grid.append((hit/max(1,tot),a,b,tr))
 best=max(grid);_,alpha,beta,topr=best
 result={'stage':'joint target-schema → historical-record → source-field alignment','selected':{'record_semantic_weight':alpha,'record_coherence_bias':beta,'top_records':topr},'train_grid_top5':[{'explicit_acc':x[0],'alpha':x[1],'beta':x[2],'top_records':x[3]} for x in sorted(grid,reverse=True)[:5]],'splits':{},'missing':missing,'guardrail':'QA001-70 alone selects alpha/beta/top_records. Candidate values are absent from all scoring text and used only after record/field selection. QA071-100 validation only; QA101-400 remains sealed.'}
 for name,lo,hi in [('train',0,TR),('validation',TR,DEV)]:
  M=defaultdict(lambda:Counter(n=0,correct=0,covered=0));exact_n=exact_ok=0;samples=[]
  for qi,qa,pack in rows:
   if not(lo<=qi<hi):continue
   gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});pr=predict(pack,alpha,beta,topr) if pack is not None else {};allok=True
   for p,g in gold.items():
    typ=str((gi.get(p) or {}).get('type','unknown'));m=M[typ];m['n']+=1;m['covered']+=int(p in pr);ok=p in pr and base.norm(pr[p]['value'])==base.norm(g);m['correct']+=int(ok);allok&=ok
    if name=='validation' and typ=='explicit' and not ok and len(samples)<10:samples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'pred':pr.get(p)})
   exact_n+=1;exact_ok+=int(bool(gold) and allok)
  result['splits'][name]={'metrics':{t:{'n':c['n'],'coverage':c['covered']/max(1,c['n']),'accuracy':c['correct']/max(1,c['n'])} for t,c in M.items()},'exact_argument_set':exact_ok/max(1,exact_n),'samples':samples}
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_JOINT_SCHEMA='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
