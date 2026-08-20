import json
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('ai-hyz/MemoryAgentBench',split='Test_Time_Learning',revision='main')
compact=[]
for i,row in enumerate(ds):
    meta=row.get('metadata') or {}
    compact.append({
      'row_idx':i,'source':meta.get('source'),
      'context_chars':len(row.get('context') or ''),'context_prefix':(row.get('context') or '')[:2200],
      'questions_count':len(row.get('questions') or []),'questions':(row.get('questions') or [])[:5],
      'answers':(row.get('answers') or [])[:5],
      'question_types':(meta.get('question_types') or [])[:10],
      'keypoints':(meta.get('keypoints') or [])[:10],
      'demo_prefix':str(meta.get('demo') or '')[:2200],
      'metadata_keys':list(meta.keys()),
    })
out={'num_rows':len(ds),'compact':compact}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_ROWS='+json.dumps(compact,ensure_ascii=False))
