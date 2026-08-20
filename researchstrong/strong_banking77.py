import json, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT=Path('researchstrong/strong_banking77_result.json')
SOURCE_URL='https://raw.githubusercontent.com/vinceackermann2-sys/meta-intellgience/research/real-memory-gates-20260820/researchbreakthrough/full_banking77.py'

def load_frozen_source():
    src=urllib.request.urlopen(SOURCE_URL,timeout=30).read().decode('utf-8')
    src=src.replace("if __name__=='__main__':main()",'')
    ns={'__name__':'frozen_mab_parser'}
    exec(compile(src,'frozen_full_banking77.py','exec'),ns)
    return ns

def eval_split(ns, model, rel_emb, rel_ids, clf, split):
    LABEL2PID=ns['LABEL2PID']; type_map=ns['type_map']; remastered_answers=ns['remastered_answers']
    anchor=ns['anchor']; paths=ns['paths']; seq=ns['seq']; feature_rows=ns['feature_rows']; prefilter=ns['prefilter']; choose_path=ns['choose_path']; exact=ns['exact']
    ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split=split)
    agg=defaultdict(lambda:{'questions':0,'answer_exact':0,'anchor_coverage':0,'gold_endpoint_path':0,'mapped_cases':0})
    unmapped_relations=defaultdict(int)
    cases=0
    for row in ds:
        cases+=1
        ts=list(row.get('new_triples_labeled') or [])
        adj=defaultdict(list); mapped=True
        for k,t in enumerate(ts):
            if not isinstance(t,(list,tuple)) or len(t)<3:
                mapped=False; unmapped_relations['MALFORMED']+=1; break
            rlabel=str(t[1])
            if rlabel not in LABEL2PID:
                mapped=False; unmapped_relations[rlabel]+=1; break
            s,_,o=map(str,t[:3]); adj[s].append((LABEL2PID[rlabel],o,k,0))
        if not mapped: continue
        for s,es in list(adj.items()):
            for r,o,ser,rank in list(es):
                if r=='P26': adj[o].append((r,s,ser,rank))
        typ,roles=type_map(adj); answers=remastered_answers(row); qs=row.get('questions') or []
        if isinstance(qs,str): qs=[qs]
        bucket=str(len(ts)); agg[bucket]['mapped_cases']+=1
        for q in qs:
            A=agg[bucket]; A['questions']+=1
            st=anchor(q,adj); A['anchor_coverage']+=int(st is not None)
            ps=paths(st,adj); by=defaultdict(list)
            for p in ps: by[seq(p)].append(p)
            A['gold_endpoint_path']+=int(any(exact(p[-1][2],answers) for p in ps))
            programs=list(by)
            if not programs: continue
            feats=feature_rows(q,st,programs,by,model,rel_emb,rel_ids,typ,roles)
            e={'programs':programs,'features':feats}; idx=prefilter(e,12)
            if not idx: continue
            probs=clf.predict_proba(np.asarray([feats[i] for i in idx]))[:,1]
            best=idx[int(np.argmax(probs))]
            pred=choose_path(by[programs[best]],typ,roles,0)
            A['answer_exact']+=int(pred is not None and exact(pred[-1][2],answers))
    keys=['questions','answer_exact','anchor_coverage','gold_endpoint_path']
    total={k:sum(v[k] for v in agg.values()) for k in keys}
    def fmt(v):
        n=max(1,v['questions'])
        return {'questions':v['questions'],'exact_match':v['answer_exact']/n,'anchor_coverage':v['anchor_coverage']/n,'gold_endpoint_path_oracle':v['gold_endpoint_path']/n}
    return {'cases':cases,'by_chain_length':{k:fmt(v) for k,v in sorted(agg.items())},'overall':fmt(total),'unmapped_relations':dict(unmapped_relations)}

def main():
    ns=load_frozen_source()
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    rel_ids=list(ns['RELNAME']); rel_text=[ns['RELNAME'][r]+'. '+', '.join(ns['CUES'].get(r,[])) for r in rel_ids]
    rel_emb=model.encode(rel_text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    D={s:ns['build'](rows[s],model,rel_emb,rel_ids) for s in srcs}
    held={ns['norm'](e['q']) for s in srcs[2:] for e in D[s]['examples']}
    dev=[e for s in srcs[:2] for e in D[s]['examples'] if ns['norm'](e['q']) not in held]
    clf=ns['train_ranker'](dev)
    results={}
    for split in ['CF9k','T']:
        results[split]=eval_split(ns,model,rel_emb,rel_ids,clf,split)
    out={
      'stage':'frozen cross-dataset replication after CF3k first-shot success',
      'protocol':'parser/ranker/cues frozen from MAB 6K+32K; no MQuAKE questions or answers used for fitting; CF9k reported for scale and T is the independent temporal transfer gate',
      'training_examples':len(dev),
      'results':results,
      'guardrail':'No parameter, cue, threshold, or ranking weight is changed from the system that produced the earlier CF3k result.'
    }
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('FROZEN_REPLICATION='+json.dumps(out,ensure_ascii=False))

if __name__=='__main__': main()
