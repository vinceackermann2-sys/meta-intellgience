import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k')
rels=Counter();lengths=Counter();shapes=Counter()
for row in ds:
    ts=list(row.get('new_triples_labeled') or [])
    lengths[len(ts)]+=1
    for t in ts:
        if isinstance(t,(list,tuple)) and len(t)>=3:
            rels[str(t[1])]+=1
            shapes[type(t).__name__]+=1
out={'stage':'fresh validation schema audit','guardrail':'questions and answers intentionally not read or emitted',
     'cases':len(ds),'unique_relation_labels':sorted(rels),'relation_counts':dict(rels),
     'chain_length_histogram':dict(lengths),'triple_container_types':dict(shapes)}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('REMASTERED_RELATION_SCHEMA='+json.dumps(out,ensure_ascii=False))
