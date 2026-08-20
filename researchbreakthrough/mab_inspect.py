import json
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

OUT=Path('researchbreakthrough/mab_inspection.json')

def labels_for(row,n):
    sp=row.get('split') or {}
    return list((sp.get(str(n),[]) if isinstance(sp,dict) else []) or [])

def audit(n,ds):
    edited=[row for row in ds if any('edited' in str(x) for x in labels_for(row,n))]
    req=defaultdict(set); chain=defaultdict(set); rows=[]
    for row in edited:
        for rw in row.get('requested_rewrite') or []:
            req[(rw.get('subject'),rw.get('relation_id'))].add((rw.get('target_new') or {}).get('str'))
        triples=((row.get('orig') or {}).get('new_triples_labeled') or [])
        for t in triples:
            if len(t)>=3:chain[(str(t[0]),str(t[1]))].add(str(t[2]))
        rows.append(triples)
    req_conf={str(k):sorted(str(x) for x in v) for k,v in req.items() if len(v)>1}
    chain_conf={str(k):sorted(str(x) for x in v) for k,v in chain.items() if len(v)>1}
    merged={k:next(iter(v)) for k,v in chain.items() if len(v)==1}
    preserved=[]
    for triples in rows:
        ok=all(len(t)<3 or merged.get((str(t[0]),str(t[1])))==str(t[2]) for t in triples)
        preserved.append(ok)
    return {'edit_num':n,'edited_cases':len(edited),'requested_registers':len(req),
            'requested_conflicting_registers':len(req_conf),'chain_registers':len(chain),
            'chain_conflicting_registers':len(chain_conf),'cases_fully_preserved_by_single_value_merge':sum(preserved),
            'case_preservation_rate':sum(preserved)/max(1,len(preserved)),
            'requested_conflict_examples':list(req_conf.items())[:20],'chain_conflict_examples':list(chain_conf.items())[:20]}

def main():
    ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k')
    out={'stage':'clean-benchmark structural merge audit','dataset':'henryzhongsc/MQuAKE-Remastered CF3k',
         'purpose':'test whether corrected edit sets admit a coherent single-value compiled graph before language QA',
         'results':[audit(n,ds) for n in [100,1000,3000]]}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('REMASTERED_MERGE_AUDIT='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
