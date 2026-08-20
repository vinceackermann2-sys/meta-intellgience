import json, urllib.request
from collections import defaultdict, Counter
from pathlib import Path

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
N=100

def stream(url):
    with urllib.request.urlopen(url,timeout=120) as r:
        for raw in r:
            if raw.strip(): yield json.loads(raw.decode('utf-8'))

qas=[]
for x in stream(BASE+'qa_dataset.jsonl'):
    qas.append(x)
    if len(qas)>=N: break

sessions=[]
by_original=defaultdict(list)
by_turn_source=defaultdict(list)
for s in stream(BASE+'toolmem_conversation.jsonl'):
    i=len(sessions); sessions.append(s)
    for x in (s.get('original_conversation_ids') or []): by_original[str(x)].append(i)
    for t in (s.get('turns') or []):
        if t.get('source_id') is not None: by_turn_source[str(t.get('source_id'))].append(i)

def resolve(q,index):
    ids=[str(x) for x in (q.get('source_conversation_ids') or [])]
    cand=None
    for x in ids:
        z=set(index.get(x,[])); cand=z if cand is None else cand & z
    if cand:return min(cand),'intersection'
    u=set()
    for x in ids:u.update(index.get(x,[]))
    return (min(u),'union') if u else (None,'missing')

missing=[]; recovered_by_turn=0; discrepancies=[]
for q in qas:
    oi,om=resolve(q,by_original); ti,tm=resolve(q,by_turn_source)
    if oi is None:
        rec={'qa_id':q.get('qa_id'),'source_conversation_ids':q.get('source_conversation_ids'),'turn_source_resolved':ti is not None,'turn_source_mode':tm}
        if ti is not None:
            recovered_by_turn+=1
            rec['session_id']=sessions[ti].get('session_id')
            rec['session_original_ids']=sessions[ti].get('original_conversation_ids')
        missing.append(rec)
    if oi is not None and ti is not None and oi!=ti:
        discrepancies.append({'qa_id':q.get('qa_id'),'original_session':sessions[oi].get('session_id'),'turn_session':sessions[ti].get('session_id')})

all_needed=Counter(str(s) for q in qas for s in (q.get('source_conversation_ids') or []))
not_in_original=[x for x in all_needed if x not in by_original]
not_in_turn=[x for x in all_needed if x not in by_turn_source]

out={
 'stage':'Mem2Act session-mapping forensic',
 'tasks':len(qas),'total_sessions':len(sessions),
 'missing_by_original_count':len(missing),'recovered_by_turn_source':recovered_by_turn,
 'missing_records':missing,
 'needed_ids_absent_from_original_index':not_in_original,
 'needed_ids_absent_from_turn_source_index':not_in_turn,
 'resolved_index_discrepancies':discrepancies[:20],
 'guardrail':'No gold tool arguments, grounding_info, or answers are read; this inspects only QA source_conversation_ids and input-session provenance.'
}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MEM2ACT_MAPPING_FORENSIC='+json.dumps(out,ensure_ascii=False))
