import json
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
out={'stage':'question-grammar inspection only','guardrail':'answers intentionally omitted','rows':{}}
for row in ds:
    src=(row.get('metadata') or {}).get('source')
    if src not in srcs:continue
    meta=row.get('metadata') or {}
    out['rows'][src]={
        'questions':list(row.get('questions') or []),
        'question_types':list(meta.get('question_types') or []),
        'keypoints':list(meta.get('keypoints') or []),
        'metadata_keys':list(meta.keys())
    }
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_QUESTION_GRAMMAR='+json.dumps(out,ensure_ascii=False))
