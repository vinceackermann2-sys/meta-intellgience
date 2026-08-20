import json, urllib.request
from collections import Counter
from pathlib import Path

URL='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/toolmem_conversation.jsonl'
sessions=[]
with urllib.request.urlopen(URL,timeout=60) as r:
    for raw in r:
        if not raw.strip(): continue
        sessions.append(json.loads(raw.decode('utf-8')))
        if len(sessions)>=12: break

role_counts=Counter(); turn_keys=Counter(); tool_shapes=Counter(); tc_keys=Counter(); session_keys=Counter(); source_counts=[]
examples=[]
for si,s in enumerate(sessions):
    session_keys.update(s.keys()); source_counts.append(len(s.get('original_conversation_ids') or []))
    for ti,t in enumerate(s.get('turns') or []):
        role=str(t.get('role','?')); role_counts[role]+=1; turn_keys.update(t.keys())
        tc=t.get('tool_calls')
        if tc is not None:
            tool_shapes[type(tc).__name__]+=1
            arr=tc if isinstance(tc,list) else [tc]
            for z in arr:
                if isinstance(z,dict):
                    tc_keys.update(z.keys())
                    # Structural example only: key names/types, not QA labels or full content.
                    if len(examples)<8:
                        examples.append({'role':role,'turn_keys':sorted(t.keys()),'tool_call_keys':sorted(z.keys()),'tool_call_value_types':{k:type(v).__name__ for k,v in z.items()}})

out={
  'stage':'Mem2Act input-session schema inspection only',
  'guardrail':'QA dataset/tool_call gold labels are not opened by this script; only conversation input structure is inspected',
  'sessions_sampled':len(sessions),
  'session_keys':dict(session_keys),
  'turn_role_counts':dict(role_counts),
  'turn_keys':dict(turn_keys),
  'tool_calls_container_types':dict(tool_shapes),
  'tool_call_keys':dict(tc_keys),
  'source_ids_per_session':source_counts,
  'structural_examples':examples,
}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MEM2ACT_INPUT_SCHEMA='+json.dumps(out,ensure_ascii=False))
