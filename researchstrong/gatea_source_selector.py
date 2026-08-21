from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile,numpy as np
import pycountry,webcolors
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
import joint_schema_alignment as js
import typed_span_diagnostic as ts
import normalization_operator_oracle as no

OUT=Path(__file__).with_name('gatea_source_selector_result.json')
TR=70;DEV=100;K=2;JOINT=(0.75,0.0,2)
SRC_NAMES=['omit','default','joint','typed','operator']
OPS=['identity','coerce','color_hex','color_name','country2','country3','language2','language3','string','number','date']
KINDS=['url','email','ipv4','hex_id','named_id','id','code','short_code','username','quoted_username','hex_color','color_name','date','relative_time','number','quoted','entity_phrase','structured']
DSRC=['schema_default','description_default','optional_empty_array','optional_empty_string','boolean_false_default','offset_zero','first_page_one','latest_index_zero','latest_zero']

def ntok(s):
 s=re.sub(r'([a-z])([A-Z])',r'\1 \2',str(s));return [x for x in re.findall(r'[a-z0-9]+',s.lower()) if x not in {'the','a','an','of','to','for','value','parameter','target','tool'}]
def simname(a,b):
 A=set(ntok(a));B=set(ntok(b));return len(A&B)/max(1,len(A|B))
def meta(p,d):return (str(p)+' '+str((d or {}).get('description',''))+' '+str((d or {}).get('type',''))).casefold()
def required(qa,p):return p in set((((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('required') or []))
def policy_default(qa,p,d):
 desc=str((d or {}).get('description',''));typ=str((d or {}).get('type','')).lower();optional=not required(qa,p)
 if 'default' in (d or {}):return d['default'],'schema_default'
 dd=base.desc_defaults(desc)
 if dd:return dd[0],'description_default'
 if optional and typ in ('array','list'):return [],'optional_empty_array'
 if optional and typ in ('string','str'):return '','optional_empty_string'
 if typ in ('bool','boolean'):return False,'boolean_false_default'
 pl=str(p).lower().replace('_','')
 if 'offset' in pl:return 0,'offset_zero'
 if pl in ('page','pageindex') or ('page' in pl and 'index' in pl):return 1,'first_page_one'
 if pl=='index' and any(x in desc.lower() for x in ['latest','most recent','starting from 0','start from 0']):return 0,'latest_index_zero'
 if typ in ('int','integer','float','number') and any(x in desc.lower() for x in ['starting from 0','start from 0']) and any(x in desc.lower() for x in ['latest','most recent']):return 0,'latest_zero'
 return None,None

def transform(v,op,d):
 try:
  if op=='identity':return v
  if op=='coerce':return base.coerce(v,d)
  if op=='color_hex':return webcolors.name_to_hex(str(v).strip())
  if op=='color_name':return webcolors.hex_to_name(str(v).strip().lower())
  if op=='country2':return pycountry.countries.lookup(str(v).strip()).alpha_2
  if op=='country3':return pycountry.countries.lookup(str(v).strip()).alpha_3
  if op=='language2':return getattr(pycountry.languages.lookup(str(v).strip()),'alpha_2')
  if op=='language3':return getattr(pycountry.languages.lookup(str(v).strip()),'alpha_3')
  if op=='string':return str(v).strip()
  if op=='number':return base.coerce(v,d)
  if op=='date':
   vals=no.date_variants(str(v));return vals[0] if vals else None
 except: return None
 return None

def allowed_ops(v,p,d):
 m=meta(p,d);typ=str((d or {}).get('type','')).lower();out=['identity','coerce']
 q=str(v).strip()
 if 'color' in m or re.fullmatch(r'#[0-9A-Fa-f]{6}',q):out += ['color_hex','color_name']
 if 'country' in m:out += ['country2','country3']
 if 'language' in m:out += ['language2','language3']
 if typ in ('string','str'):out += ['string']
 if typ in ('integer','int','number','float'):out += ['number']
 if any(x in m for x in ['date','day','time','timestamp']):out += ['date']
 seen=[]
 for x in out:
  if x not in seen:seen.append(x)
 return seen

def typed_candidates(enc,qa,session,p,d):
 # High-precision typed spans only. Deliberately excludes generic base.spans.
 q=str(qa.get('query',''));sch=qa.get('target_tool_schema') or {};tt=f"request {q} target tool {sch.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}";raw=[]
 def admissible(k):
  m=meta(p,d);typ=str((d or {}).get('type','')).lower()
  if any(x in m for x in ['url','link','uri','website']):return k=='url'
  if 'email' in m:return k=='email'
  if any(x in m for x in ['ip address','ip_address','ipv4','ipaddr']):return k=='ipv4'
  if 'color' in m:return k in {'color_name','hex_color','quoted'}
  if any(x in m for x in ['date','day','time','timestamp']):return k in {'date','relative_time','quoted','number'}
  if any(x in m for x in ['id','identifier','hash','wallet','contract']):return k in {'hex_id','named_id','id','quoted','number','code'}
  if any(x in m for x in ['user','username','handle','account']):return k in {'username','quoted_username','quoted','code'}
  if any(x in m for x in ['code','symbol','ticker','currency','pair','league']):return k in {'code','short_code','quoted'}
  if typ in ('integer','int','number','float'):return k in {'number','quoted'}
  return k in {'url','email','ipv4','hex_id','named_id','username','quoted_username','hex_color','color_name','date','quoted','entity_phrase'}
 for c in ts.typed_spans(q,p,d):
  if admissible(c['kind']):raw.append({**c,'where':'current','rank':-1})
 if session is not None:
  eps=es.episodes(session)
  for er,ep,sm in es.retrieve_episodes(enc,eps,tt,K):
   for ti,t in ep['rows']:
    content=t.get('content','');txt=json.dumps(content,ensure_ascii=False) if isinstance(content,(dict,list)) else str(content)
    for c in ts.typed_spans(txt,p,d):
     if admissible(c['kind']):raw.append({**c,'where':'memory','rank':er})
 if not raw:return []
 E=enc.encode([tt]+[c['address'][:900] for c in raw],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);ss=E[1:]@E[0]
 out=[]
 for i,c in enumerate(raw):out.append((c,float(ss[i]),c['kind'],c['where'],int(c.get('rank',0))))
 return sorted(out,key=lambda z:z[1],reverse=True)[:8]

def operator_candidates(enc,qa,session,p,d):
 # Structured/raw memory pool + deterministic operators. Values are never embedded; only masked provenance/address text is scored.
 if session is None:return []
 q=str(qa.get('query',''));sch=qa.get('target_tool_schema') or {};tt=f"request {q} target tool {sch.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}";raw=[]
 for er,ep,sm in es.retrieve_episodes(enc,es.episodes(session),tt,K):
  for c in af.occurrences(ep,er,sm):
   if c['kind']=='text_span':continue
   raw.append(c)
 if not raw:return []
 E=enc.encode([tt]+[c['address'][:900] for c in raw],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);ss=E[1:]@E[0]
 order=np.argsort(-ss)[:min(20,len(raw))];out=[]
 for i in order:
  c=raw[int(i)]
  for op in allowed_ops(c['value'],p,d):
   z=transform(c['value'],op,d)
   if z is not None:out.append((z,op,float(ss[int(i)]),c))
 return out

def base_features(enc,qa,p,d):
 q=str(qa.get('query',''));sch=qa.get('target_tool_schema') or {};tt=f"target tool {sch.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}";E=enc.encode([q,tt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);typ=str((d or {}).get('type','')).lower();m=meta(p,d)
 return [float(required(qa,p)),float('default' in d),float(typ in ('string','str')),float(typ in ('integer','int','number','float')),float(typ in ('bool','boolean')),float(typ in ('array','list')),simname(p,q),float(E[0]@E[1]),float('color' in m),float('country' in m),float('language' in m),float(any(x in m for x in ['id','identifier','hash'])),float(any(x in m for x in ['url','link','uri']))]

def cand_features(basef,src,op='identity',score=0.,field='',p='',kind='structured',where='',rank=0,dsrc=''):
 f=list(basef)+[float(src==x) for x in SRC_NAMES]+[float(op==x) for x in OPS]
 f += [float(kind==x) for x in KINDS]+[float(dsrc==x) for x in DSRC]
 f += [float(score),simname(field,p),float(where=='current'),float(where=='memory'),float(rank),float(bool(field)),float(src!='omit')]
 return f

def build_slot(enc,qa,session,p,d,jp):
 basef=base_features(enc,qa,p,d);cs=[]
 cs.append({'src':'omit','op':'identity','value':None,'feat':cand_features(basef,'omit',p=p)})
 dv,ds=policy_default(qa,p,d)
 if ds:
  for op in allowed_ops(dv,p,d):
   z=transform(dv,op,d)
   if z is not None:cs.append({'src':'default','op':op,'value':z,'feat':cand_features(basef,'default',op,p=p,dsrc=ds),'detail':ds})
 j=jp.get(p)
 if j is not None:
  for op in allowed_ops(j['value'],p,d):
   z=transform(j['value'],op,d)
   if z is not None:cs.append({'src':'joint','op':op,'value':z,'feat':cand_features(basef,'joint',op,float(j.get('score',0)),str(j.get('field','')),p,'structured','memory',int(j.get('record',0))),'detail':j})
 for c,sc,k,w,r in typed_candidates(enc,qa,session,p,d):
  for op in allowed_ops(c['value'],p,d):
   z=transform(c['value'],op,d)
   if z is not None:cs.append({'src':'typed','op':op,'value':z,'feat':cand_features(basef,'typed',op,sc,'',p,k,w,r),'detail':{'kind':k,'where':w,'score':sc}})
 for z,op,sc,c in operator_candidates(enc,qa,session,p,d):
  cs.append({'src':'operator','op':op,'value':z,'feat':cand_features(basef,'operator',op,sc,str(c.get('key','')),p,'structured','memory',int(c.get('rank',0))),'detail':{'key':c.get('key',''),'kind':c.get('kind',''),'score':sc}})
 # Deduplicate by (normalized output, source, op); value remains executor payload only.
 out=[];seen=set()
 for c in cs:
  key=(base.norm(c['value']) if c['src']!='omit' else '<omit>',c['src'],c['op'])
  if key not in seen:seen.add(key);out.append(c)
 return out

def main():
 td=Path(tempfile.gettempdir())/'mab_source_selector';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
 qas=list(base.load_jsonl(qp))[:DEV];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rows=[];missing=[]
 for qi,qa in enumerate(qas):
  session=base.find_session(qa,sessions,by)
  if session is None:missing.append(qi)
  props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,session) if session is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});slots=[]
  for p,d0 in props.items():
   d=d0 or {};cands=build_slot(enc,qa,session,p,d,jp);slots.append({'p':p,'d':d,'cands':cands,'present':p in gold,'gold':gold.get(p)})
  rows.append({'qi':qi,'qa':qa,'slots':slots})
  if qi%10==0:print('SOURCE_SELECTOR_BUILD',qi,flush=True)
 # Candidate correctness is the TRAINING LABEL only. Candidate values are never features.
 X=[];Y=[]
 for r in rows:
  if r['qi']>=TR:continue
  for s in r['slots']:
   for c in s['cands']:
    ok=(not s['present'] and c['src']=='omit') or (s['present'] and c['src']!='omit' and base.norm(c['value'])==base.norm(s['gold']))
    X.append(c['feat']);Y.append(int(ok))
 clf=LogisticRegression(C=0.25,max_iter=3000,class_weight='balanced',random_state=42).fit(np.asarray(X,float),np.asarray(Y,int))
 result={'stage':'Gate A v2 value-blind source/operation selector','model':'logistic regression C=0.25 trained QA001-70 candidate correctness only','candidate_sources':SRC_NAMES,'operators':OPS,'splits':{},'missing_session_zero_based':missing,'guardrail':'QA001-70 trains candidate operation selection. Gold values/presence are labels only, never features. grounding_info/evolution_chain/source IDs are not routing features. QA071-100 is score-only. QA101-400 gold remains sealed.'}
 for name,lo,hi in [('train',0,TR),('validation',TR,DEV)]:
  C=P=G=0;exact_n=exact_ok=0;levels=defaultdict(lambda:Counter(n=0,correct=0,pred=0));srcs=Counter();samples=[]
  for r in rows:
   if not(lo<=r['qi']<hi):continue
   pred={};dbg={};gold=((r['qa'].get('tool_call') or {}).get('arguments') or {});gi=((r['qa'].get('tool_call') or {}).get('grounding_info') or {})
   for s in r['slots']:
    probs=clf.predict_proba(np.asarray([c['feat'] for c in s['cands']],float))[:,1];i=int(np.argmax(probs));c=s['cands'][i];pr=float(probs[i]);
    if c['src']!='omit':pred[s['p']]=c['value'];srcs[c['src']+':'+c['op']]+=1
    lvl=str((gi.get(s['p']) or {}).get('type','unknown'));m=levels[lvl];m['n']+=1;m['pred']+=int(s['p'] in pred);ok=s['p'] in pred and s['p'] in gold and base.norm(pred[s['p']])==base.norm(gold[s['p']]);m['correct']+=int(ok)
    if name=='validation' and not ok and len(samples)<20:samples.append({'qa_id':r['qa'].get('qa_id'),'level':lvl,'parameter':s['p'],'gold':gold.get(s['p'],'<ABSENT>'),'pred':pred.get(s['p'],'<OMIT>'),'chosen':c['src']+':'+c['op'],'prob':pr,'detail':c.get('detail')})
   c0,p0,g0,f0,e0=base.arg_metrics(pred,gold);C+=c0;P+=p0;G+=g0;exact_n+=1;exact_ok+=e0
  prec=C/max(1,P);rec=C/max(1,G);F=2*prec*rec/max(1e-12,prec+rec);packed={k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in levels.items()};result['splits'][name]={'correct':C,'predicted':P,'gold':G,'precision':prec,'recall':rec,'parameter_f1':F,'exact_argument_set':exact_ok/max(1,exact_n),'levels':packed,'sources':dict(srcs),'samples':samples}
 v=result['splits'].get('validation',{});lev=v.get('levels',{});result['gate_a']={'global_f1':v.get('parameter_f1',0),'explicit':lev.get('explicit',{}).get('accuracy',0),'inferred':lev.get('inferred',{}).get('accuracy',0),'default':lev.get('default',{}).get('accuracy',0)};g=result['gate_a'];g['pass']=g['global_f1']>=.60 and g['explicit']>=.50 and g['inferred']>=.25 and g['default']>=.85
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_GATEA_SOURCE_SELECTOR='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
