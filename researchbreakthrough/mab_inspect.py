import json, urllib.request, re
from collections import defaultdict,Counter
from pathlib import Path
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
N=100

def stream(url):
    with urllib.request.urlopen(url,timeout=180) as r:
        for raw in r:
            if raw.strip():yield json.loads(raw.decode('utf-8'))
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
def flatten(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(dict,list)):out.extend(flatten(v))
            else:out.append(v)
    elif isinstance(x,list):
        for v in x:out.extend(flatten(v))
    return out
def spans(text):
    vals=[];text=str(text)
    pats=[r'https?://[^\s\]\[\)\(<>"\']+',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?\b',r'\b(?:19|20)\d{2}\b',r'\b[A-Z][A-Z0-9._/-]{1,20}\b',r'\b\d+(?:\.\d+)?(?:h|hr|hrs|d|days|w|weeks|min|minutes)?\b',r'"([^"\n]{1,120})"',r"'([^'\n]{1,120})'"]
    for p in pats:
        for m in re.finditer(p,text):vals.append((m.group(1) if m.lastindex else m.group(0)).strip())
    for m in re.finditer(r'\b(?:[A-Z][\w.-]+)(?:\s+(?:[A-Z][\w.-]+)){0,4}\b',text):vals.append(m.group(0).strip())
    return vals
def desc_defaults(desc):
    d=str(desc or '');vals=[]
    pats=[r"(?i)default\s+value\s+(?:is|=|:)\s*['\"]?([^'\".,;\)]+)",r"(?i)defaults?\s+to\s*['\"]?([^'\".,;\)]+)",r'(?i)\b([A-Za-z_][\w]*)\s*=\s*([^\s\),;]+)']
    for p in pats:
        for m in re.finditer(p,d):vals.append(m.group(m.lastindex).strip())
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
    ids=[str(x) for x in q.get('source_conversation_ids') or []];u=set()
    for x in ids:u.update(idx.get(x,[]))
    return sessions[min(u)] if u else None
C=defaultdict(Counter);total=0;tasks=0
for q in qas:
    s=resolve(q)
    if s is None:continue
    tasks+=1;turns=s.get('turns') or [];alltext='\n'.join(str(t.get('content','')) for t in turns);toolvals=[]
    for t in turns:
        for tc in t.get('tool_calls') or []:toolvals.extend(flatten(parse_args(tc)))
    allspans=spans(alltext)+spans(q.get('query',''))
    schema=q.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {})
    gold=((q.get('tool_call') or {}).get('arguments') or {});ground=((q.get('tool_call') or {}).get('grounding_info') or {})
    for k,v in gold.items():
        total+=1;typ=str((ground.get(k) or {}).get('type','unknown'));c=C[typ];c['n']+=1;nv=norm(v)
        d=props.get(k) or {};schema_vals=[]
        if 'default' in d:schema_vals.append(d['default'])
        if isinstance(d.get('enum'),list):schema_vals.extend(d['enum'])
        schema_vals.extend(desc_defaults(d.get('description','')))
        hit_tool=any(norm(z)==nv for z in toolvals);hit_span=any(norm(z)==nv for z in allspans);hit_schema=any(norm(z)==nv for z in schema_vals)
        c['tool_value']+=hit_tool;c['extractable_span']+=hit_span;c['schema_candidate']+=hit_schema;c['copy_oracle']+=(hit_tool or hit_span or hit_schema)
def ratios(c):return {k:c[k]/max(1,c['n']) for k in ['tool_value','extractable_span','schema_candidate','copy_oracle']}
out={'stage':'Mem2Act pointer candidate copy-oracle on dev only','dev_tasks_resolved':tasks,'gold_slots':total,'by_grounding':{k:{'n':v['n'],**ratios(v)} for k,v in sorted(C.items())},'overall':{'n':sum(v['n'] for v in C.values()),'copy_oracle':sum(v['copy_oracle'] for v in C.values())/max(1,sum(v['n'] for v in C.values()))},'guardrail':'Uses gold values only on QA001-100 development set to measure whether exact values are present in legitimate candidate sources. QA101-400 gold remains unopened.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_POINTER_ORACLE='+json.dumps(out,ensure_ascii=False))
