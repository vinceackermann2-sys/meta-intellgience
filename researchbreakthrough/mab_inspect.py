import json, urllib.request
from collections import defaultdict
from pathlib import Path
ROOT='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/'
BASE=ROOT+'Mem2ActBench/'

def stream(url):
    with urllib.request.urlopen(url,timeout=240) as r:
        for raw in r:
            if raw.strip(): yield json.loads(raw.decode('utf-8'))

def scalar_paths(x,p=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            q=f'{p}.{k}' if p else str(k)
            if isinstance(v,(dict,list)):out.extend(scalar_paths(v,q))
            else:out.append((q,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            q=f'{p}[{i}]'
            if isinstance(v,(dict,list)):out.extend(scalar_paths(v,q))
            else:out.append((q,v))
    return out

def turn_summary(row):
    fields=[]
    for k in ['turns','conversation_history','conversation','messages']:
        v=row.get(k) if isinstance(row,dict) else None
        if isinstance(v,list):
            roles=[]
            for t in v[:20]:
                if isinstance(t,dict):roles.append(str(t.get('role',t.get('from','?'))))
            fields.append({'field':k,'count':len(v),'roles':roles})
    return fields

qas=list(stream(BASE+'qa_dataset.jsonl'))
sessions=list(stream(BASE+'toolmem_conversation.jsonl'))
idx=defaultdict(list)
for i,s in enumerate(sessions):
    for x in s.get('original_conversation_ids') or []:idx[str(x)].append(i)
missing=[];wanted=set()
for q in qas:
    ids=[str(x) for x in q.get('source_conversation_ids') or []]
    u=set()
    for x in ids:u.update(idx.get(x,[]))
    if not u:
        missing.append({'qa_id':q.get('qa_id'),'source_ids':ids})
        wanted.update(ids)
found={}
files=['toolace_formatted_conversations.jsonl','bfcl_formatted_conversations.jsonl','oasst1_formatted_conversations.jsonl']
for fn in files:
    remaining=wanted-set(found)
    if not remaining:break
    for rownum,row in enumerate(stream(ROOT+fn),1):
        vals=scalar_paths(row)
        exact={sid for sid in remaining if any(str(v)==sid for _,v in vals)}
        if exact:
            idish=[(p,v) for p,v in vals if any(t in p.lower() for t in ['id','source'])][:25]
            for sid in exact:
                found[sid]={'source_file':fn,'row_number':rownum,'top_keys':list(row)[:30] if isinstance(row,dict) else [],'matching_paths':[p for p,v in vals if str(v)==sid][:10],'id_metadata':idish,'turn_fields':turn_summary(row)}
            remaining=wanted-set(found)
            if not remaining:break
no_source=[m['qa_id'] for m in missing if not m['source_ids']]
result={'stage':'Mem2Act public-release session repair audit','qas':len(qas),'released_sessions':len(sessions),'unresolved_qas':len(missing),'unresolved_records':missing,'unique_missing_source_ids':sorted(wanted),'upstream_source_ids_found':found,'source_ids_still_missing':sorted(wanted-set(found)),'unresolved_with_no_source_ids':no_source,'repair_plan':'For unresolved QAs with source IDs, reconstruct a minimal evidence session from the corresponding upstream formatted conversation row, preserving its original turn order. For QAs with no source IDs, use an empty memory session and schema/current query only; label these separately. Never read tool_call.arguments, grounding_info, or evolution_chain to repair context.','guardrail':'This audit reads only QA IDs/source_conversation_ids and released/upstream conversation provenance. It does not inspect QA gold arguments, grounding_info, or evolution_chain. QA101-400 answers remain sealed.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_RELEASE_REPAIR='+json.dumps(result,ensure_ascii=False))
