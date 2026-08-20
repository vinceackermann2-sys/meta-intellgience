import json
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
compact=[]
for i,row in enumerate(ds):
    meta=row.get('metadata') or {}
    compact.append({
      'row_idx':i,'source':meta.get('source'),
      'context_chars':len(row.get('context') or ''),'context_prefix':(row.get('context') or '')[:3200],
      'questions_count':len(row.get('questions') or []),'questions':(row.get('questions') or [])[:8],
      'answers':(row.get('answers') or [])[:8],
      'question_types':(meta.get('question_types') or [])[:20],
      'keypoints':(meta.get('keypoints') or [])[:20],
      'previous_events':(meta.get('previous_events') or [])[:20],
      'question_dates':(meta.get('question_dates') or [])[:20],
      'demo_prefix':str(meta.get('demo') or '')[:3200],
      'metadata_keys':list(meta.keys()),
    })
out={'num_rows':len(ds),'compact':compact}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_CONFLICT_ROWS='+json.dumps(compact,ensure_ascii=False))
