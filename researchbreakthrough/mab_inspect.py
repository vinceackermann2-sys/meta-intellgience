import json
from pathlib import Path
from datasets import load_dataset

ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k')
s=[]
for i in range(min(5,len(ds))):
    row=ds[i];v=row.get('split')
    s.append({'i':i,'split_type':type(v).__name__,'split_repr':repr(v)[:5000],'keys':list(row.keys())})
out={'stage':'inspect live MQuAKE-Remastered split schema','samples':s}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('REMASTERED_SCHEMA='+json.dumps(out,ensure_ascii=False))
