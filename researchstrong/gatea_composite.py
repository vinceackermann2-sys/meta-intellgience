from pathlib import Path
from collections import defaultdict,Counter
import json,re,tempfile,numpy as np
import pycountry,webcolors
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import joint_schema_alignment as js
import typed_span_diagnostic as ts
import normalization_operator_oracle as no

OUT=Path(__file__).with_name('gatea_composite_result.json')
TR=70;DEV=100;K=2
JOINT=(0.75,0.0,2) # frozen by QA001-70 joint-schema grid

HIGH_KINDS={'url','email','ipv4','hex_id','named_id','id','code','short_code','username','quoted_username','hex_color','color_name','date','relative_time','number','quoted','entity_phrase'}

def ntok(s):
 s=re.sub(r'([a-z])([A-Z])',r'\1 \2',str(s));return [x for x in re.findall(r'[a-z0-9]+',s.lower()) if x not in {'the','a','an','of','to','for','value','parameter','target','tool'}]
def jac(a,b):
 A=set(ntok(a));B=set(ntok(b));return len(A&B)/max(1,len(A|B))
def target_text(qa,p,d):
 sch=qa.get('target_tool_schema') or {};return f"current request {qa.get('query','')} target tool {sch.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
def meta(p,d):return (str(p)+' '+str((d or {}).get('description',''))+' '+str((d or {}).get('type',''))).casefold()
def eligible(k,p,d,current=False):
 m=meta(p,d);typ=str((d or {}).get('type','')).casefold()
 if any(x in m for x in ['url','link','uri','website']):return k=='url'
 if 'email' in m:return k=='email'
 if any(x in m for x in ['ip address','ip_address','ipv4','ipaddr']):return k=='ipv4'
 if any(x in m for x in ['user','username','handle','account']):return k in {'username','quoted_username','quoted','code'}
 if 'color' in m:return k in {'color_name','hex_color','quoted'}
 if any(x in m for x in ['date','day','time','timestamp']):return k in {'date','relative_time','quoted','number'}
 if any(x in m for x in ['country','language']) and 'code' in m:return k in {'code','short_code','quoted','entity_phrase'}
 if any(x in m for x in ['code','symbol','ticker','currency','pair','league']):return k in {'code','short_code','quoted'}
 if any(x in m for x in ['id','identifier','hash','wallet','contract']):return k in {'hex_id','named_id','id','quoted','number','code'}
 if typ in {'integer','int','number','float'}:return k in {'number','quoted'}
 if typ in {'string','str',''} and current:return k in {'quoted','entity_phrase','code','quoted_username','url','email','hex_id','named_id'}
 return False

def typed_top(enc,qa,session,p,d):
 q=str(qa.get('query',''));tt=target_text(qa,p,d);cands=[]
 # Current request is legitimate live input. Use typed + conservative generic spans.
 for c in ts.typed_spans(q,p,d):
  if eligible(c['kind'],p,d,True):cands.append({**c,'source':'current_typed','rank':-1})
 for v in base.spans(q):
  c={'value':v,'kind':'generic_span','address':'current request context '+re.sub(re.escape(str(v)),'<VALUE>',q,flags=re.I),'source':'current_generic','rank':-1}
  # Generic spans are admitted only for string-ish fields; lower prior than typed evidence.
  if str((d or {}).get('type','')).casefold() in {'string','str',''}:cands.append(c)
 if session is not None:
  eps=es.episodes(session)
  for er,ep,sm in es.retrieve_episodes(enc,eps,tt,K):
   for ti,t in ep['rows']:
    if str(t.get('role','')) not in ('user','assistant','tool'):continue
    content=t.get('content','');txt=json.dumps(content,ensure_ascii=False) if isinstance(content,(dict,list)) else str(content)
    for c in ts.typed_spans(txt,p,d):
     if eligible(c['kind'],p,d,False):cands.append({**c,'source':'memory_typed','rank':er})
 if not cands:return None
 docs=[tt]+[c['address'][:1000] for c in cands];E=enc.encode(docs,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);s=E[1:]@E[0]
 best=None
 for i,c in enumerate(cands):
  prior=0.0
  if c['source']=='current_typed':prior+=0.10
  elif c['source']=='current_generic':prior+=0.02
  else:prior+=0.04-0.02*max(0,int(c.get('rank',0)))
  if c['kind'] in {'url','email','ipv4','hex_id','username','quoted_username','color_name','hex_color','date'}:prior+=0.04
  score=float(s[i]+prior)
  if best is None or score>best['score']:best={'value':c['value'],'score':score,'kind':c['kind'],'source':c['source'],'address':c['address'][:240]}
 return best

def required_set(qa):
 params=((qa.get('target_tool_schema') or {}).get('parameters') or {});return set(params.get('required') or [])
def policy_default(qa,p,d):
 desc=str((d or {}).get('description',''));typ=str((d or {}).get('type','')).lower();req=required_set(qa);optional=p not in req
 if 'default' in (d or {}):return {'value':base.coerce(d['default'],d),'source':'schema_default'}
 dd=base.desc_defaults(desc)
 if dd:return {'value':base.coerce(dd[0],d),'source':'description_default'}
 if optional and typ in ('array','list'):return {'value':[],'source':'optional_empty_array'}
 if optional and typ in ('string','str'):return {'value':'','source':'optional_empty_string'}
 if typ in ('bool','boolean'):return {'value':False,'source':'boolean_false_default'}
 pl=str(p).lower().replace('_','')
 if 'offset' in pl:return {'value':0,'source':'offset_zero'}
 if pl in ('page','pageindex') or ('page' in pl and 'index' in pl):return {'value':1,'source':'first_page_one'}
 if pl=='index' and any(x in desc.lower() for x in ['latest','most recent','starting from 0','start from 0']):return {'value':0,'source':'latest_index_zero'}
 if typ in ('int','integer','float','number') and any(x in desc.lower() for x in ['starting from 0','start from 0']) and any(x in desc.lower() for x in ['latest','most recent']):return {'value':0,'source':'latest_zero'}
 return None

def canonicalize(v,p,d):
 if v is None:return v
 m=meta(p,d);enum=(d or {}).get('enum') or []
 # Enum is the strongest schema-side representation contract.
 if enum:
  for x in [v]+[z for z,_ in no.variants(v)]:
   for e in enum:
    if base.norm(x)==base.norm(e):return e
 # Explicit code contracts.
 if 'country' in m and 'code' in m:
  try:
   c=pycountry.countries.lookup(str(v).strip());return c.alpha_3 if any(x in m for x in ['alpha3','alpha-3','3-letter','three-letter','iso3']) else c.alpha_2
  except:pass
 if 'language' in m and 'code' in m:
  try:
   x=pycountry.languages.lookup(str(v).strip());
   if any(z in m for z in ['alpha3','alpha-3','3-letter','three-letter','iso3']):return getattr(x,'alpha_3',v)
   return getattr(x,'alpha_2',getattr(x,'alpha_3',v))
  except:pass
 if 'color' in m:
  q=str(v).strip()
  if 'hex' in m:
   try:return webcolors.name_to_hex(q)
   except:pass
  else:
   try:return webcolors.hex_to_name(q.lower())
   except:pass
 return base.coerce(v,d)

def features(enc,qa,p,d,jpred,ttop,pdef):
 q=str(qa.get('query',''));tt=target_text(qa,p,d);src=(pdef or {}).get('source','none');typ=str((d or {}).get('type','')).lower();pt=set(ntok(p));qt=set(ntok(q));lex=len(pt&qt)/max(1,len(pt));
 E=enc.encode([q,tt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);sem=float(E[0]@E[1])
 sources=['schema_default','description_default','optional_empty_array','optional_empty_string','boolean_false_default','offset_zero','first_page_one','latest_index_zero','latest_zero']
 return [float(pdef is not None),float('default' in (d or {})),float(p in required_set(qa)),float(typ in ('bool','boolean')),float(typ in ('string','str')),float(typ in ('int','integer','number','float')),lex,sem,float(jpred is not None),float((jpred or {}).get('score',-1)),float(ttop is not None),float((ttop or {}).get('score',-1))]+[float(src==s) for s in sources]

def main():
 td=Path(tempfile.gettempdir())/'mab_gatea';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
 qas=list(base.load_jsonl(qp))[:DEV];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rows=[];missing=[]
 for qi,qa in enumerate(qas):
  ses=base.find_session(qa,sessions,by)
  if ses is None:missing.append(qi)
  props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gi=((qa.get('tool_call') or {}).get('grounding_info') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {})
  slot=[]
  for p,d0 in props.items():
   d=d0 or {};tt=typed_top(enc,qa,ses,p,d);jj=jp.get(p);pd=policy_default(qa,p,d);slot.append({'p':p,'d':d,'typed':tt,'joint':jj,'default':pd,'gold':gold.get(p),'level':str((gi.get(p) or {}).get('type','unknown')),'feat':features(enc,qa,p,d,jj,tt,pd)})
  rows.append({'qi':qi,'qa':qa,'slots':slot})
 # Train-only default router. It decides WHEN a general executor default should fire, never its value.
 X=[];y=[]
 for r in rows:
  if r['qi']>=TR:continue
  for s in r['slots']:
   if s['default'] is not None:X.append(s['feat']);y.append(int(s['level']=='default'))
 clf=LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',random_state=42).fit(np.array(X,float),np.array(y,int)) if len(set(y))>1 else None
 for r in rows:
  for s in r['slots']:
   s['pdef']=float(clf.predict_proba([s['feat']])[0,1]) if clf is not None and s['default'] is not None else 0.0
 # Tune only two routing thresholds on QA001-70. Objective is train global parameter F1 with a small balanced accuracy tie-break.
 configs=[]
 for dt in [0.35,0.45,0.55,0.65,0.75]:
  for tt in [0.35,0.45,0.55,0.65,0.75]:
   C=P=G=0;bylvl=defaultdict(lambda:[0,0])
   for r in rows:
    if r['qi']>=TR:continue
    pred={};gold=((r['qa'].get('tool_call') or {}).get('arguments') or {})
    for s in r['slots']:
     got=None
     if s['default'] is not None and s['pdef']>=dt:got=s['default']['value']
     elif s['typed'] is not None and s['typed']['score']>=tt:got=s['typed']['value']
     elif s['joint'] is not None:got=s['joint']['value']
     if got is not None:pred[s['p']]=canonicalize(got,s['p'],s['d'])
     lvl=s['level'];bylvl[lvl][1]+=1;bylvl[lvl][0]+=int(s['p'] in pred and base.norm(pred[s['p']])==base.norm(s['gold']))
    c,pn,g,f,_=base.arg_metrics(pred,gold);C+=c;P+=pn;G+=g
   prec=C/max(1,P);rec=C/max(1,G);F=2*prec*rec/max(1e-12,prec+rec);bal=np.mean([a/max(1,n) for a,n in bylvl.values()]) if bylvl else 0;configs.append((F+0.05*bal,F,dt,tt,{k:a/max(1,n) for k,(a,n) in bylvl.items()}))
 best=max(configs);_,trainF,DT,TT,train_by=best
 result={'stage':'Gate A composite: train-routed defaults + joint schema alignment + typed evidence + schema normalization','frozen_components':{'joint':{'alpha':JOINT[0],'beta':JOINT[1],'top_records':JOINT[2]},'default_router':'logistic regression trained QA001-70 only','normalizers':['schema type coercion','enum exact variant','country/language code','CSS color name/hex'],'typed_extractor':'schema-typed spans + conservative current-request spans'},'selected_train_thresholds':{'default_probability':DT,'typed_score':TT},'train_selection':{'global_f1':trainF,'level_accuracy':train_by,'top_configs':[{'objective':z[0],'f1':z[1],'default_t':z[2],'typed_t':z[3],'levels':z[4]} for z in sorted(configs,reverse=True)[:5]]},'splits':{},'missing_session_zero_based':missing,'guardrail':'All architecture, default-router fitting, joint hyperparameters, and routing thresholds use QA001-70 only. QA071-100 is scored only. QA101-400 gold remains sealed. Candidate values are dereferenced only after address/span selection; exact values are never scoring features.'}
 for name,lo,hi in [('train',0,TR),('validation',TR,DEV)]:
  C=P=G=0;exact_ok=exact_n=0;levels=defaultdict(lambda:Counter(n=0,correct=0,pred=0));sources=Counter();samples=[]
  for r in rows:
   if not(lo<=r['qi']<hi):continue
   pred={};debug={};gold=((r['qa'].get('tool_call') or {}).get('arguments') or {})
   for s in r['slots']:
    got=None;src='omit'
    if s['default'] is not None and s['pdef']>=DT:got=s['default']['value'];src='default:'+s['default']['source']
    elif s['typed'] is not None and s['typed']['score']>=TT:got=s['typed']['value'];src='typed:'+s['typed']['source']+':'+s['typed']['kind']
    elif s['joint'] is not None:got=s['joint']['value'];src='joint:'+str(s['joint'].get('field',''))
    if got is not None:pred[s['p']]=canonicalize(got,s['p'],s['d']);sources[src]+=1
    lvl=s['level'];m=levels[lvl];m['n']+=1;m['pred']+=int(s['p'] in pred);ok=s['p'] in pred and base.norm(pred[s['p']])==base.norm(s['gold']);m['correct']+=int(ok)
    if name=='validation' and not ok and len(samples)<18:samples.append({'qa_id':r['qa'].get('qa_id'),'level':lvl,'parameter':s['p'],'gold':s['gold'],'pred':pred.get(s['p']),'source':src,'default_p':s['pdef'],'typed':s['typed'],'joint':s['joint']})
   c,pn,g,f,ex=base.arg_metrics(pred,gold);C+=c;P+=pn;G+=g;exact_ok+=ex;exact_n+=1
  prec=C/max(1,P);rec=C/max(1,G);F=2*prec*rec/max(1e-12,prec+rec)
  result['splits'][name]={'correct':C,'predicted':P,'gold':G,'precision':prec,'recall':rec,'parameter_f1':F,'exact_argument_set':exact_ok/max(1,exact_n),'levels':{k:{'n':c['n'],'accuracy':c['correct']/max(1,c['n']),'prediction_rate':c['pred']/max(1,c['n'])} for k,c in levels.items()},'sources':dict(sources),'samples':samples}
 val=result['splits']['validation'];lv=val['levels'];result['gate_a']={'global_f1':val['parameter_f1'],'explicit':(lv.get('explicit') or {}).get('accuracy',0),'inferred':(lv.get('inferred') or {}).get('accuracy',0),'default':(lv.get('default') or {}).get('accuracy',0)};g=result['gate_a'];result['gate_a']['pass']=bool(g['global_f1']>=.60 and g['explicit']>=.50 and g['inferred']>=.25 and g['default']>=.85)
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_GATEA_COMPOSITE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
