from pathlib import Path
from collections import defaultdict,Counter
import json,re,tempfile,numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
OUT=Path(__file__).with_name('record_role_diagnostic_result.json');N=100;K=2

def flat(x,p=''):
 o=[]
 if isinstance(x,dict):
  for k,v in x.items():
   q=f'{p}.{k}' if p else str(k);o+=flat(v,q) if isinstance(v,(dict,list)) else [(q,v)]
 elif isinstance(x,list):
  for i,v in enumerate(x[:20]):
   q=f'{p}[{i}]';o+=flat(v,q) if isinstance(v,(dict,list)) else [(q,v)]
 return o

def shape(x,d=0):
 if d>3:return '<N>'
 if isinstance(x,dict):return {k:shape(v,d+1) for k,v in list(x.items())[:25]}
 if isinstance(x,list):return [shape(x[0],d+1)] if x else []
 return '<V>'

def nodes(x,p='$'):
 o=[]
 if isinstance(x,dict):
  if flat(x):o.append((p,x))
  for k,v in x.items():
   if isinstance(v,(dict,list)):o+=nodes(v,f'{p}.{k}')
 elif isinstance(x,list):
  for i,v in enumerate(x[:20]):
   if isinstance(v,(dict,list)):o+=nodes(v,f'{p}[{i}]')
 return o

def recs(ep,rank,sim):
 o=[]
 for ti,t in ep['rows']:
  role=str(t.get('role',''))
  for tc in t.get('tool_calls') or []:
   a=base.parse_args(tc)
   if a:o.append({'kind':'call','tool':base.tool_name(tc),'role':role,'turn':ti,'rank':rank,'sim':sim,'obj':a,'fields':flat(a)})
  c=t.get('content','');z=None
  if isinstance(c,(dict,list)):z=c
  elif isinstance(c,str) and c.strip()[:1] in '[{':
   try:z=json.loads(c)
   except:pass
  if z is not None:
   for p,n in nodes(z):o.append({'kind':'output','tool':'','role':role,'turn':ti,'rank':rank,'sim':sim,'obj':n,'fields':flat(n),'path':p})
 return o

def text(r):
 return f"role {r['role']} kind {r['kind']} tool {r['tool']} paths {' '.join(k for k,_ in r['fields'][:30])} shape {json.dumps(shape(r['obj']),ensure_ascii=False)}"

def target(qa,p,d):
 s=qa.get('target_tool_schema') or {};return f"request {qa.get('query','')} target tool {s.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"

def rank(enc,q,docs):
 if not docs:return []
 E=enc.encode([q]+docs,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);return list(np.argsort(-(E[1:]@E[0])))

def main():
 td=Path(tempfile.gettempdir())/'mab_rr';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
 qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');M=defaultdict(lambda:Counter(n=0,rcov=0,r1=0,r3=0,f1=0,f3=0,j1=0,j3=0));examples=[];missing=[]
 for qi,qa in enumerate(qas):
  ses=base.find_session(qa,sessions,by)
  if ses is None:missing.append(qi);continue
  schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});eps=es.episodes(ses)
  for p,g in gold.items():
   typ=str((gi.get(p) or {}).get('type','unknown'));m=M[typ];m['n']+=1;d=props.get(p) or {};q=target(qa,p,d);rr=[]
   for er,ep,sm in es.retrieve_episodes(enc,eps,q,K):rr+=recs(ep,er,sm)
   pos=[i for i,r in enumerate(rr) if any(base.norm(v)==base.norm(g) for _,v in r['fields'])];m['rcov']+=bool(pos);ro=rank(enc,q,[text(r) for r in rr]);m['r1']+=bool(pos) and ro and ro[0] in pos;m['r3']+=bool(pos) and any(i in pos for i in ro[:3])
   # Oracle record -> field role: diagnose field mapping independently of record selection.
   if pos:
    r=rr[pos[0]];fo=rank(enc,q,[f"tool {r['tool']} field path {k} sibling paths {' '.join(x for x,_ in r['fields'])}" for k,_ in r['fields']]);fp=[i for i,(_,v) in enumerate(r['fields']) if base.norm(v)==base.norm(g)];m['f1']+=fo and fo[0] in fp;m['f3']+=any(i in fp for i in fo[:3])
   # Actual joint record then field.
   for kk,key in [(1,'j1'),(3,'j3')]:
    hit=False
    for ri in ro[:kk]:
     r=rr[ri];fo=rank(enc,q,[f"tool {r['tool']} field path {k} sibling paths {' '.join(x for x,_ in r['fields'])}" for k,_ in r['fields']])[:kk]
     if any(base.norm(r['fields'][fi][1])==base.norm(g) for fi in fo):hit=True;break
    m[key]+=hit
   if typ=='explicit' and len(examples)<8 and pos and (not ro or ro[0] not in pos):examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'top_records':[{'tool':rr[i]['tool'],'kind':rr[i]['kind'],'paths':[k for k,_ in rr[i]['fields'][:10]],'has_gold':i in pos} for i in ro[:4]]})
 packed={t:{'n':c['n'],**{k:c[k]/max(1,c['n']) for k in ['rcov','r1','r3','f1','f3','j1','j3']}} for t,c in M.items()}
 out={'stage':'record→field-role decomposition','metrics':packed,'examples':examples,'missing':missing,'guardrail':'QA001-100 gold scores only. Values are excluded from record/field ranking text. QA101-400 remains sealed.'};OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_RECORD_ROLE='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
