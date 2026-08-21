from pathlib import Path
from collections import defaultdict,Counter
import json,re,tempfile,numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
OUT=Path(__file__).with_name('provenance_role_router_result.json');TR=70;DEV=100;K=2

def safe(x):return json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x)
def mask(s,v):
 q=str(v);return re.sub(re.escape(q),'<VALUE>',str(s),flags=re.I) if q else str(s)
def last_user(ep,turn):
 rows=[(ti,t) for ti,t in ep['rows'] if ti<=turn and str(t.get('role',''))=='user']
 return safe(rows[-1][1].get('content','')) if rows else ''
def target(qa,p,d):
 s=qa.get('target_tool_schema') or {};return f"target tool {s.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
def query(qa):return str(qa.get('query',''))
def candidates(ep,rank,sim):
 out=[]
 for c in af.occurrences(ep,rank,sim):
  if c['kind']=='text_span':continue
  u=last_user(ep,int(c.get('turn',-1)));cc=dict(c);cc['request_ctx']=mask(u,c['value']);out.append(cc)
 return out

def emb(enc,texts):return enc.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
def main():
 td=Path(tempfile.gettempdir())/'mab_prov';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp);qas=list(base.load_jsonl(qp))[:DEV];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');slots=[];missing=[]
 for qi,qa in enumerate(qas):
  ses=base.find_session(qa,sessions,by)
  if ses is None:missing.append(qi);continue
  eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
  for p,g in gold.items():
   d=props.get(p) or {};tt=target(qa,p,d);qq=query(qa);cs=[]
   for er,ep,sm in es.retrieve_episodes(enc,eps,qq+' '+tt,K):cs+=candidates(ep,er,sm)
   if not cs:slots.append((qi,str((gi.get(p) or {}).get('type','unknown')),[],[],g,qa.get('qa_id'),p));continue
   # entity/task binding ignores target parameter lexical bias as much as possible.
   A=emb(enc,[qq]+[f"historical request {c['request_ctx']} memory tool {c['tool']}" for c in cs]);record=A[1:]@A[0]
   # field role alignment uses no candidate value.
   B=emb(enc,[tt]+[f"source tool {c['tool']} source field {c['key']} kind {c['kind']} role context {c['address']}" for c in cs]);field=B[1:]@B[0]
   labels=np.array([base.norm(c['value'])==base.norm(g) for c in cs],dtype=bool);slots.append((qi,str((gi.get(p) or {}).get('type','unknown')),record,field,g,qa.get('qa_id'),p,labels,cs))
 # choose entity-vs-role weight on train only.
 weights=[0,0.2,0.4,0.6,0.8,1.0];train=[]
 for w in weights:
  hit=tot=0
  for s in slots:
   if s[0]>=TR or len(s)<8 or not len(s[2]):continue
   score=w*s[2]+(1-w)*s[3];hit+=int(np.argmax(score) in np.where(s[7])[0]);tot+=1
  train.append((hit/max(1,tot),w))
 best=max(train)[1];out={}
 for name,lo,hi in [('train',0,TR),('validation',TR,DEV)]:
  M=defaultdict(lambda:Counter(n=0,cov=0,t1=0,t3=0,t5=0));samples=[]
  for s in slots:
   if not(lo<=s[0]<hi):continue
   typ=s[1];m=M[typ];m['n']+=1
   if len(s)<8 or not len(s[2]):continue
   pos=np.where(s[7])[0];m['cov']+=int(len(pos)>0);score=best*s[2]+(1-best)*s[3];order=np.argsort(-score);m['t1']+=int(len(pos)>0 and order[0] in pos);m['t3']+=int(len(pos)>0 and any(i in pos for i in order[:3]));m['t5']+=int(len(pos)>0 and any(i in pos for i in order[:5]))
   if name=='validation' and typ=='explicit' and len(samples)<8 and len(pos)>0 and order[0] not in pos:samples.append({'qa_id':s[5],'parameter':s[6],'gold':s[4],'top':[{'score':float(score[i]),'record':float(s[2][i]),'field':float(s[3][i]),'tool':s[8][i]['tool'],'key':s[8][i]['key'],'value':s[8][i]['value'],'request_ctx':s[8][i]['request_ctx'][:180]} for i in order[:4]]})
  out[name]={'metrics':{t:{'n':c['n'],'coverage':c['cov']/max(1,c['n']),'top1':c['t1']/max(1,c['n']),'top3':c['t3']/max(1,c['n']),'top5':c['t5']/max(1,c['n'])} for t,c in M.items()},'samples':samples}
 res={'stage':'provenance-aware two-stage role router','selected_record_weight':best,'train_weight_curve':[{'weight':w,'top1':a} for a,w in train],**out,'missing':missing,'guardrail':'QA001-70 selects only the record-vs-field mixing weight. Candidate values are masked from scoring text. QA071-100 validation only; QA101-400 remains sealed.'};OUT.write_text(json.dumps(res,indent=2,ensure_ascii=False));print('MEM2ACT_PROVENANCE_ROLE='+json.dumps(res,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
