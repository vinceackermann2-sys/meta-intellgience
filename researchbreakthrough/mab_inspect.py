import json, urllib.request, tempfile, re
from collections import defaultdict, Counter
from pathlib import Path

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
N=100

def stream(url):
    with urllib.request.urlopen(url,timeout=90) as r:
        for raw in r:
            if raw.strip(): yield json.loads(raw.decode('utf-8'))

def norm(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(int,float)):return str(v)
    return re.sub(r'\s+',' ',str(v).strip()).casefold()

def contains(text,v):
    z=norm(v)
    if not z:return False
    return z in norm(text)

def parse_args(tc):
    if not isinstance(tc,dict):return {}
    f=tc.get('function') or {}; a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:
            z=json.loads(a); return z if isinstance(z,dict) else {}
        except Exception:return {}
    return {}

def flatten_scalar(d):
    out=[]
    if isinstance(d,dict):
        for k,v in d.items():
            if isinstance(v,(dict,list)): out.extend(flatten_scalar(v))
            else:out.append((str(k),v))
    elif isinstance(d,list):
        for v in d: out.extend(flatten_scalar(v))
    return out

qas=[]
for x in stream(BASE+'qa_dataset.jsonl'):
    qas.append(x)
    if len(qas)>=N:break
needed=Counter(str(s) for q in qas for s in (q.get('source_conversation_ids') or []))
sessions=[];by_source=defaultdict(list)
for s in stream(BASE+'toolmem_conversation.jsonl'):
    ids=[str(x) for x in s.get('original_conversation_ids') or []]
    if any(x in needed for x in ids):
        i=len(sessions);sessions.append(s)
        for x in ids:by_source[x].append(i)

def find_session(q):
    ids=[str(x) for x in q.get('source_conversation_ids') or []];cand=None
    for x in ids:
        z=set(by_source.get(x,[]));cand=z if cand is None else cand&z
    if cand:return sessions[min(cand)]
    u=set()
    for x in ids:u.update(by_source.get(x,[]))
    return sessions[min(u)] if u else None

counts=defaultdict(Counter); examples=[]; missing=0; total_params=0
for q in qas:
    s=find_session(q)
    if s is None:missing+=1;continue
    turns=s.get('turns') or []; text='\n'.join(str(t.get('content','')) for t in turns)
    query=str(q.get('query','')); schema=q.get('target_tool_schema') or {}; schema_text=json.dumps(schema,ensure_ascii=False)
    events=[]
    for ti,t in enumerate(turns):
        for tc in (t.get('tool_calls') or []):
            args=parse_args(tc)
            if args:events.append((ti,args))
    gold=((q.get('tool_call') or {}).get('arguments') or {}); grounding=((q.get('tool_call') or {}).get('grounding_info') or {})
    for k,v in gold.items():
        total_params+=1;typ=str((grounding.get(k) or {}).get('type','unknown')); C=counts[typ];C['n']+=1
        in_query=contains(query,v);in_schema=contains(schema_text,v);in_text=contains(text,v)
        any_arg=False;same_name=False;latest_same=None
        for ti,args in events:
            for kk,vv in flatten_scalar(args):
                if norm(vv)==norm(v):any_arg=True
                if norm(kk)==norm(k):
                    if norm(vv)==norm(v):same_name=True
                    latest_same=vv
        latest_match=latest_same is not None and norm(latest_same)==norm(v)
        C['query_exact']+=int(in_query);C['schema_exact']+=int(in_schema);C['memory_text_exact']+=int(in_text);C['any_tool_arg_exact']+=int(any_arg);C['same_slot_exact']+=int(same_name);C['latest_same_slot_exact']+=int(latest_match)
        recoverable=in_query or in_schema or in_text or any_arg
        C['literal_recoverable_anywhere']+=int(recoverable)
        if not recoverable and len(examples)<12:examples.append({'qa_id':q.get('qa_id'),'param':k,'grounding_type':typ,'value_type':type(v).__name__})

def ratio(c,k):return c[k]/max(1,c['n'])
out={'stage':'Mem2Act offline argument-location diagnostic','tasks':len(qas),'sessions_loaded':len(sessions),'missing_sessions':missing,'total_gold_params':total_params,'guardrail':'Gold arguments are used only to measure where their values occur. They do not choose memories or generate predictions.',
     'by_grounding_type':{typ:{'n':c['n'], **{k:ratio(c,k) for k in ['query_exact','schema_exact','memory_text_exact','any_tool_arg_exact','same_slot_exact','latest_same_slot_exact','literal_recoverable_anywhere']}} for typ,c in sorted(counts.items())},
     'unrecoverable_examples_structure_only':examples}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_SLOT_CEILING='+json.dumps(out,ensure_ascii=False))
