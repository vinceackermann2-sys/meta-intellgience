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
    if isinstance(v,bool): return str(v).lower()
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
def flatten(x):
    out=[]
    if isinstance(x,dict):
        for v in x.values(): out.extend(flatten(v)) if isinstance(v,(dict,list)) else out.append(v)
    elif isinstance(x,list):
        for v in x: out.extend(flatten(v)) if isinstance(v,(dict,list)) else out.append(v)
    return out
def spans(text):
    vals=[]; text=str(text)
    pats=[r'https?://[^\s\]\[\)\(<>"\']+',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?\b',r'\b(?:19|20)\d{2}\b',r'\b[A-Z][A-Z0-9._/-]{1,20}\b',r'\b\d+(?:\.\d+)?(?:h|hr|hrs|d|days|w|weeks|min|minutes)?\b',r'"([^"\n]{1,120})"',r"'([^'\n]{1,120})'"]
    for p in pats:
        for m in re.finditer(p,text): vals.append((m.group(1) if m.lastindex else m.group(0)).strip())
    for m in re.finditer(r'\b(?:[A-Z][\w.-]+)(?:\s+(?:[A-Z][\w.-]+)){0,4}\b',text): vals.append(m.group(0).strip())
    return vals
def schema_candidates(d):
    vals=[]
    if 'default' in d: vals.append(d['default'])
    if isinstance(d.get('enum'),list): vals.extend(d['enum'])
    desc=str(d.get('description',''))
    for p in [r"(?i)default\s+value\s+(?:is|=|:)\s*['\"]?([^'\".,;\)]+)",r"(?i)defaults?\s+to\s*['\"]?([^'\".,;\)]+)",r"(?i)by\s+default[^\w]+(?:the\s+)?(?:value\s+)?(?:is\s+)?['\"]?([^'\".,;\)]+)",r"(?i)([A-Za-z0-9._/-]+)\s*\((?:the\s+)?default\)"]:
        for m in re.finditer(p,desc): vals.append(m.group(1).strip())
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

qas=[]
for q in stream(BASE+'qa_dataset.jsonl'):
    qas.append(q)
    if len(qas)>=N:break
sessions=[];idx=defaultdict(list)
for s in stream(BASE+'toolmem_conversation.jsonl'):
    i=len(sessions);sessions.append(s)
    for x in s.get('original_conversation_ids') or []:idx[str(x)].append(i)
def resolve(q):
    u=set()
    for x in q.get('source_conversation_ids') or []:u.update(idx.get(str(x),[]))
    return sessions[min(u)] if u else None
C=defaultdict(Counter);tasks=0;total=0
for q in qas:
    s=resolve(q)
    if s is None:continue
    tasks+=1;turns=s.get('turns') or [];alltext='\n'.join(str(t.get('content','')) for t in turns);toolvals=[]
    for t in turns:
        for tc in t.get('tool_calls') or []:toolvals.extend(flatten(parse_args(tc)))
    allspans=spans(alltext)+spans(q.get('query',''))
    params=((q.get('target_tool_schema') or {}).get('parameters') or {});props=params.get('properties') or {};required=set(params.get('required') or [])
    gold=((q.get('tool_call') or {}).get('arguments') or {});ground=((q.get('tool_call') or {}).get('grounding_info') or {})
    for k,v in gold.items():
        total+=1;typ=str((ground.get(k) or {}).get('type','unknown'));c=C[typ];c['n']+=1;nv=norm(v);d=props.get(k) or {}
        hit_tool=any(norm(z)==nv for z in toolvals);hit_span=any(norm(z)==nv for z in allspans);hit_schema=any(norm(z)==nv for z in schema_candidates(d));hit_policy=any(norm(z)==nv for z in policy_candidates(k,d,required))
        c['tool_value']+=hit_tool;c['extractable_span']+=hit_span;c['schema_candidate']+=hit_schema;c['policy_candidate']+=hit_policy;c['copy_oracle_base']+=(hit_tool or hit_span or hit_schema);c['copy_oracle_policy']+=(hit_tool or hit_span or hit_schema or hit_policy)
def ratios(c):
    n=max(1,c['n']);return {k:c[k]/n for k in ['tool_value','extractable_span','schema_candidate','policy_candidate','copy_oracle_base','copy_oracle_policy']}
Nslots=sum(v['n'] for v in C.values())
result={'stage':'Mem2Act pointer+schema-policy copy oracle on dev only','dev_tasks_resolved':tasks,'gold_slots':Nslots,'by_grounding':{k:{'n':v['n'],**ratios(v)} for k,v in sorted(C.items())},'overall':{'n':Nslots,'base_copy_oracle':sum(v['copy_oracle_base'] for v in C.values())/max(1,Nslots),'policy_copy_oracle':sum(v['copy_oracle_policy'] for v in C.values())/max(1,Nslots)},'guardrail':'QA001-100 development labels only. QA101-400 gold remains unopened. Gold is used only to measure candidate-set ceiling; it does not construct candidates.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_POLICY_POINTER_ORACLE='+json.dumps(result,ensure_ascii=False))
