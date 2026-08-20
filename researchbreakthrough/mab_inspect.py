import json, urllib.request, re
from collections import defaultdict, Counter
from pathlib import Path

ROOT='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/'
FINAL=ROOT+'Mem2ActBench/'

def stream(url):
    with urllib.request.urlopen(url,timeout=180) as r:
        for raw in r:
            if raw.strip():
                try:yield json.loads(raw.decode('utf-8'))
                except Exception:pass

def ntext(s):return re.sub(r'\s+',' ',str(s).strip()).casefold()
def strings_at_paths(x,path=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from strings_at_paths(v,f'{path}.{k}' if path else str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from strings_at_paths(v,f'{path}[{i}]')
    elif isinstance(x,str):yield path,x

def safe_id_metadata(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            kl=str(k).lower()
            if any(z in kl for z in ['thread_id','source_id','conversation_id','record_id']):
                if isinstance(v,(str,int,float)):out.append((str(k),str(v)))
                elif isinstance(v,list):out.append((str(k),[str(a) for a in v if isinstance(a,(str,int,float))]))
            if isinstance(v,(dict,list)):out.extend(safe_id_metadata(v))
    elif isinstance(x,list):
        for v in x:out.extend(safe_id_metadata(v))
    return out

qas=list(stream(FINAL+'qa_dataset.jsonl'))
sessions=[];by_original=defaultdict(list);by_turn=defaultdict(list)
for s in stream(FINAL+'toolmem_conversation.jsonl'):
    i=len(sessions);sessions.append(s)
    for x in s.get('original_conversation_ids') or []:by_original[str(x)].append(i)
    for t in s.get('turns') or []:
        if t.get('source_id') is not None:by_turn[str(t.get('source_id'))].append(i)

def resolve(q,index):
    ids=[str(x) for x in q.get('source_conversation_ids') or []];cand=None
    for x in ids:
        z=set(index.get(x,[]));cand=z if cand is None else cand&z
    if cand:return min(cand),'intersection'
    u=set()
    for x in ids:u.update(index.get(x,[]))
    return (min(u),'union') if u else (None,'missing')

missing=[]
for q in qas:
    oi,om=resolve(q,by_original);ti,tm=resolve(q,by_turn)
    if oi is None:
        missing.append({'qa_id':q.get('qa_id'),'query_norm':ntext(q.get('query','')),'source_conversation_ids':q.get('source_conversation_ids') or [],'turn_source_resolved':ti is not None})

# Match missing final questions against the pre-normalized QA construction file, but emit only provenance/id metadata.
need={m['query_norm']:m for m in missing if m['query_norm']}
pre_matches=defaultdict(list);pre_count=0
for row in stream(ROOT+'processed_data/memory_driven_qa_Kimi-K2-Instruct-0905_v2.jsonl'):
    pre_count+=1
    hits=[]
    for path,s in strings_at_paths(row):
        ns=ntext(s)
        if ns in need:hits.append((ns,path))
    if hits:
        ids=safe_id_metadata(row)
        top_keys=sorted(row.keys()) if isinstance(row,dict) else []
        for ns,path in hits:
            pre_matches[ns].append({'matched_path':path,'top_keys':top_keys,'id_metadata':ids[:40]})

for m in missing:m['pre_normalized_matches']=pre_matches.get(m['query_norm'],[])[:5]

needed=Counter(str(s) for q in qas for s in (q.get('source_conversation_ids') or []))
out={
 'stage':'Mem2Act full-release mapping forensic',
 'tasks':len(qas),'sessions':len(sessions),'pre_normalized_rows':pre_count,
 'resolved_by_original':sum(resolve(q,by_original)[0] is not None for q in qas),
 'missing_count':len(missing),
 'missing_records':[{k:v for k,v in m.items() if k!='query_norm'} for m in missing],
 'unique_needed_ids':len(needed),
 'needed_ids_absent_from_session_original_index':[x for x in needed if x not in by_original],
 'needed_ids_absent_from_turn_source_index':[x for x in needed if x not in by_turn],
 'guardrail':'This script does not read or emit gold tool arguments/grounding_info. Pre-normalized rows are inspected only for provenance identifiers matching the final natural-language query.'
}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_400_MAPPING='+json.dumps(out,ensure_ascii=False))
