import json, urllib.request
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT=Path('researchstrong/strong_banking77_result.json')
# Immutable source: exact parser that produced the first-shot CF3k result.
SOURCE_URL='https://raw.githubusercontent.com/vinceackermann2-sys/meta-intellgience/04bcc0ef1ecceee1e9dfce26427efad92b4ae070/researchbreakthrough/full_banking77.py'

def load_frozen_source():
    src=urllib.request.urlopen(SOURCE_URL,timeout=30).read().decode('utf-8')
    src=src.replace("if __name__=='__main__':main()",'')
    ns={'__name__':'frozen_mab_parser'}
    exec(compile(src,'frozen_full_banking77.py','exec'),ns)
    return ns

def build_global_graph(ns, rows):
    LABEL2PID=ns['LABEL2PID']
    triples=set(); rel_objects=defaultdict(set); unmapped=Counter()
    for row in rows:
        for t in list(row.get('new_triples_labeled') or []):
            if not isinstance(t,(list,tuple)) or len(t)<3:
                unmapped['MALFORMED']+=1; continue
            s,rlabel,o=map(str,t[:3])
            if rlabel not in LABEL2PID:
                unmapped[rlabel]+=1; continue
            r=LABEL2PID[rlabel]; triples.add((s,r,o)); rel_objects[(s,r)].add(o)
    adj=defaultdict(list)
    # All benchmark edits are concurrent here; no arbitrary case-order recency signal.
    for s,r,o in triples: adj[s].append((r,o,0,0))
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26': adj[o].append((r,s,0,0))
    collisions={f'{s}|||{r}':len(os) for (s,r),os in rel_objects.items() if len(os)>1}
    return adj, {'unique_triples':len(triples),'registers':len(rel_objects),'conflicting_registers':len(collisions),'max_objects_per_register':max(collisions.values()) if collisions else 1,'unmapped_relations':dict(unmapped)}

def train_frozen_ranker(ns,model,rel_emb,rel_ids):
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    D={s:ns['build'](rows[s],model,rel_emb,rel_ids) for s in srcs}
    held={ns['norm'](e['q']) for s in srcs[2:] for e in D[s]['examples']}
    dev=[e for s in srcs[:2] for e in D[s]['examples'] if ns['norm'](e['q']) not in held]
    return ns['train_ranker'](dev),len(dev)

def evaluate_global(ns,model,rel_emb,rel_ids,clf,split='CF3k'):
    rows=list(load_dataset('henryzhongsc/MQuAKE-Remastered',split=split))
    adj,state=build_global_graph(ns,rows); typ,roles=ns['type_map'](adj)
    agg=defaultdict(lambda:{'questions':0,'answer_exact':0,'anchor':0,'endpoint_oracle':0,'program_oracle':0})
    candidate_program_counts=[]; path_counts=[]
    for ci,row in enumerate(rows):
        answers=ns['remastered_answers'](row); qs=row.get('questions') or []
        if isinstance(qs,str): qs=[qs]
        bucket=str(len(list(row.get('new_triples_labeled') or [])))
        for q in qs:
            A=agg[bucket]; A['questions']+=1
            st=ns['anchor'](q,adj); A['anchor']+=int(st is not None)
            ps=ns['paths'](st,adj,maxh=4,cap=50000); path_counts.append(len(ps)); by=defaultdict(list)
            for p in ps: by[ns['seq'](p)].append(p)
            A['endpoint_oracle']+=int(any(ns['exact'](p[-1][2],answers) for p in ps))
            programs=list(by); candidate_program_counts.append(len(programs))
            gold_programs={s for s,pp in by.items() if any(ns['exact'](p[-1][2],answers) for p in pp)}
            if gold_programs: A['program_oracle']+=1
            if not programs: continue
            feats=ns['feature_rows'](q,st,programs,by,model,rel_emb,rel_ids,typ,roles)
            e={'programs':programs,'features':feats}; idx=ns['prefilter'](e,12)
            if not idx: continue
            probs=clf.predict_proba(np.asarray([feats[i] for i in idx]))[:,1]
            best=idx[int(np.argmax(probs))]; pred=ns['choose_path'](by[programs[best]],typ,roles,0)
            A['answer_exact']+=int(pred is not None and ns['exact'](pred[-1][2],answers))
        if (ci+1)%250==0: print(f'GLOBAL_PROGRESS {ci+1}/{len(rows)}',flush=True)
    keys=['questions','answer_exact','anchor','endpoint_oracle','program_oracle']
    total={k:sum(v[k] for v in agg.values()) for k in keys}
    def fmt(v):
        n=max(1,v['questions']); return {'questions':v['questions'],'exact_match':v['answer_exact']/n,'anchor_coverage':v['anchor']/n,'endpoint_path_oracle':v['endpoint_oracle']/n,'program_oracle':v['program_oracle']/n}
    return {'split':split,'cases':len(rows),'global_state':state,'by_chain_length':{k:fmt(v) for k,v in sorted(agg.items())},'overall':fmt(total),'mean_paths':float(np.mean(path_counts)),'p95_paths':float(np.percentile(path_counts,95)),'mean_programs':float(np.mean(candidate_program_counts))}

def main():
    ns=load_frozen_source(); model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    rel_ids=list(ns['RELNAME']); rel_text=[ns['RELNAME'][r]+'. '+', '.join(ns['CUES'].get(r,[])) for r in rel_ids]
    rel_emb=model.encode(rel_text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    clf,n=train_frozen_ranker(ns,model,rel_emb,rel_ids)
    result=evaluate_global(ns,model,rel_emb,rel_ids,clf,'CF3k')
    out={'stage':'MQuAKE-Remastered CF3k structured-memory 3000-edit interference gate','protocol':'all 3000 cases are loaded simultaneously into one global structured memory; parser/ranker/cues remain frozen from MAB 6K+32K','training_examples':n,'result':result,'interpretation_guardrail':'This consumes new_triples_labeled (benchmark structured annotations), so it tests memory interference + question-to-relation-program execution. It is NOT directly comparable to end-to-end model-editing methods that ingest requested_rewrite and must derive downstream knowledge themselves.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)); print('GLOBAL_3000_EDIT='+json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
