import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k')
out={'stage':'locate live MQuAKE-Remastered split labels','counts':{},'examples':{}}
for n in ['100','1000','3000','6334']:
    c=Counter();ex=[]
    for i,row in enumerate(ds):
        v=(row.get('split') or {}).get(n,[])
        c[repr(v)]+=1
        if v and len(ex)<20:
            ex.append({'i':i,'case_id':row.get('case_id'),'split_value':v,'requested_rewrite':row.get('requested_rewrite'),
                       'new_triples_labeled':row.get('new_triples_labeled')})
    out['counts'][n]=dict(c);out['examples'][n]=ex
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('REMASTERED_SPLIT_VALUES='+json.dumps(out,ensure_ascii=False))
