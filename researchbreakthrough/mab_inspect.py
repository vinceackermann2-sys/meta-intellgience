import json, urllib.request, re
from collections import defaultdict,Counter
from pathlib import Path
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
N=100

def stream(url):
    with urllib.request.urlopen(url,timeout=180) as r:
        for raw in r:
            if raw.strip(): yield json.loads(raw.decode('utf-8'))
def norm(v):
    if isinstance(v,bool):return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(dict,list)):return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()
def parse_args(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {};a=(f or {}).get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:z=json.loads(a);return z if isinstance(z,dict) else {}
        except:return {}
    return {}
def flatten_items(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            key=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten_items(v,key))
            else:out.append((key,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            key=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)):out.extend(flatten_items(v,key))
            else:out.append((key,v))
    return out
def literal(v):
    if not isinstance(v,str):return v
    s=v.strip().strip('"\'')
    if s.lower() in ('true','false'):return s.lower()=='true'
    if s.lower() in ('null','none'):return None
    if re.fullmatch(r'-?\d+',s):
        try:return int(s)
        except:pass
    if re.fullmatch(r'-?\d+(?:\.\d+)',s):
        try:return float(s)
        except:pass
    return s
def desc_defaults(desc):
    vals=[];d=str(desc or '')
    for p in [r"(?i)default\s+value\s+(?:is|=|:)\s*['\"]?([^'\".,;\)]+)",r"(?i)defaults?\s+to\s*['\"]?([^'\".,;\)]+)",r"(?i)by\s+default[^\w]+(?:the\s+)?(?:value\s+)?(?:is\s+)?['\"]?([^'\".,;\)]+)",r"(?i)([A-Za-z0-9._/-]+)\s*\((?:the\s+)?default\)"]:
        for m in re.finditer(p,d):vals.append(literal(m.group(1).strip()))
    return vals
def policy_candidates(name,d,required):
    p=str(name);pl=p.lower().replace('_','');typ=str(d.get('type','')).lower();desc=str(d.get('description','')).lower();optional=p not in required;vals=[]
    if optional and typ in ('array','list'):vals.append([])
    if optional and typ in ('string','str'):vals.append('')
    if typ in ('bool','boolean'):vals.append(False)
    if 'offset' in pl:vals.append(0)
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl):vals.append(1)
    if pl=='index' and ('latest' in desc or 'most recent' in desc or 'starting from 0' in desc or 'start from 0' in desc):vals.append(0)
    if typ in ('int','integer','float','number') and ('starting from 0' in desc or 'start from 0' in desc) and ('latest' in desc or 'most recent' in desc):vals.append(0)
    return vals
def coerce(v,d):
    typ=str(d.get('type','')).lower()
    try:
        if typ in ('int','integer'):return int(float(v))
        if typ in ('float','number'):return float(v)
        if typ in ('bool','boolean'):
            if isinstance(v,bool):return v
            if norm(v) in ('true','yes','1','on'):return True
            if norm(v) in ('false','no','0','off'):return False
        if typ in ('string','str') and not isinstance(v,(dict,list)):return str(v)
    except:pass
    return v

def resolve(q,sessions,idx):
    u=set()
    for x in q.get('source_conversation_ids') or []:u.update(idx.get(str(x),[]))
    return sessions[min(u)] if u else None

qas=[]
for q in stream(BASE+'qa_dataset.jsonl'):
    qas.append(q)
    if len(qas)>=N:break
sessions=[];idx=defaultdict(list)
for s in stream(BASE+'toolmem_conversation.jsonl'):
    i=len(sessions);sessions.append(s)
    for x in s.get('original_conversation_ids') or []:idx[str(x)].append(i)

C=P=G=0;tasks=0;exact=0;by_ground=defaultdict(lambda:Counter(n=0,correct=0));sources=Counter();samples=[]
for q in qas:
    s=resolve(q,sessions,idx)
    if s is None:continue
    tasks+=1;params=((q.get('target_tool_schema') or {}).get('parameters') or {});props=params.get('properties') or {};required=set(params.get('required') or [])
    histories=defaultdict(list)
    for ti,t in enumerate(s.get('turns') or []):
        for tc in t.get('tool_calls') or []:
            for k,v in flatten_items(parse_args(tc)):
                histories[k.split('.')[-1].casefold()].append((ti,v))
    pred={};chosen={}
    for k,d0 in props.items():
        d=d0 or {};vals=histories.get(k.casefold()) or []
        got=False;v=None;src=None
        if vals:
            v=vals[-1][1];got=True;src='latest_same_slot'
        elif 'default' in d:
            v=d['default'];got=True;src='json_default'
        else:
            ds=desc_defaults(d.get('description',''))
            if ds:v=ds[0];got=True;src='description_default'
            else:
                ps=policy_candidates(k,d,required)
                if ps:v=ps[0];got=True;src='general_policy'
        if got:
            pred[k]=coerce(v,d);chosen[k]=src;sources[src]+=1
    gold=((q.get('tool_call') or {}).get('arguments') or {});ground=((q.get('tool_call') or {}).get('grounding_info') or {})
    correct=0
    for k,v in gold.items():
        typ=str((ground.get(k) or {}).get('type','unknown'));by_ground[typ]['n']+=1
        if k in pred and norm(pred[k])==norm(v):correct+=1;by_ground[typ]['correct']+=1
    C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
    if len(samples)<8:samples.append({'qa_id':q.get('qa_id'),'pred':pred,'gold':gold,'chosen':chosen})
prec=C/max(1,P);rec=C/max(1,G);f1=2*prec*rec/max(1e-12,prec+rec)
result={'stage':'Mem2Act zero-LLM deterministic slot VM dev baseline','tasks':tasks,'correct_params':C,'predicted_params':P,'gold_params':G,'global_precision':prec,'global_recall':rec,'global_f1':f1,'exact_argument_set':exact/max(1,tasks),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(by_ground.items())},'prediction_sources':dict(sources),'samples':samples,'executor':'latest same-name historical slot > explicit schema default > parsed description default > general type/schema policy; no embedding model and no LLM','guardrail':'QA001-100 development labels only. QA101-400 gold remains unopened. Gold is used only after prediction for scoring.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_ZERO_LLM_VM='+json.dumps(result,ensure_ascii=False))
