import json
from collections import defaultdict,Counter
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k')

def triples(row,key):
    return list(row.get(key) or [])

req=defaultdict(lambda:defaultdict(list)); new=defaultdict(lambda:defaultdict(list)); edit=defaultdict(lambda:defaultdict(list))
case_rows=[]
for row in ds:
    cid=row.get('case_id')
    for rw in row.get('requested_rewrite') or []:
        s=str(rw.get('subject'));r=str(rw.get('relation_id'));o=str((rw.get('target_new') or {}).get('str'))
        req[(s,r)][o].append(cid)
    for key,dst in [('new_triples_labeled',new),('edit_triples',edit)]:
        for t in triples(row,key):
            if isinstance(t,(list,tuple)) and len(t)>=3:
                s,r,o=map(str,t[:3]);dst[(s,r)][o].append(cid)
    case_rows.append(cid)

def summarize(d):
    conflicts={k:v for k,v in d.items() if len(v)>1}
    depths=Counter(len(v) for v in d.values())
    return {'logical_registers':len(d),'conflicting_registers':len(conflicts),
            'conflict_fraction':len(conflicts)/max(1,len(d)),'distinct_value_depth_histogram':dict(depths),
            'max_distinct_values':max(depths) if depths else 0,
            'examples':[{'subject':k[0],'relation':k[1],'values':{o:ids[:10] for o,ids in vals.items()}} for k,vals in list(conflicts.items())[:30]]}

out={'stage':'MQuAKE-Remastered global mergeability audit','dataset':'henryzhongsc/MQuAKE-Remastered CF3k','cases':len(ds),
     'note':'The HF split-selection arrays are empty, so this audits all corrected cases jointly. Remastered itself uses dynamic masking for contamination-free scoped edit sets; this test asks whether one global current-value graph can represent all cases simultaneously.',
     'requested_rewrite':summarize(req),'new_triples_labeled':summarize(new),'edit_triples':summarize(edit)}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('REMASTERED_GLOBAL_MERGE='+json.dumps(out,ensure_ascii=False))
