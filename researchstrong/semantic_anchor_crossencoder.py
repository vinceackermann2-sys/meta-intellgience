from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile
import numpy as np
from sentence_transformers import CrossEncoder
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_entity_anchor_oracle as a
import semantic_anchor_role_binding_cv as ar
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_anchor_crossencoder_result.json')
N=100
MODEL='cross-encoder/ms-marco-MiniLM-L6-v2'


def main():
    td=Path(tempfile.gettempdir())/'semantic_anchor_crossencoder';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N];repaired,report=repair.build();enc=wb.CachedEncoder();rerank=CrossEncoder(MODEL,device='cpu')
    stats=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0));examples=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'))
        direct,graph,labels=ar.graph_scope(nodes,qa.get('query',''))
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        # Value-masked semantic fallback nodes, only when exact discourse/entity graph has little scope.
        strictE=enc.encode([n['strict_desc'] for n in nodes]) if nodes else np.zeros((0,384),np.float32)
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};m=stats[typ];m['n']+=1
            sv=enc.encode(h.target_scope(qa,p,d));ns=strictE@sv if len(nodes) else np.asarray([])
            fallback={int(i) for i in np.argsort(-ns)[:4]} if len(nodes) else set()
            scoped=set(graph)|fallback
            rows=[]
            for ni in scoped:
                n=nodes[ni]
                for c in h.expanded(n,p,d):
                    rows.append({'value':c['value'],'node':ni,'field':c['prop'].get('field',''),'op':c.get('op'),'source':c['role'],'scope':labels.get(ni,'semantic_fallback')})
            # Aggregate exact same executor value only after scoring; preserve all role evidence by taking max score.
            if not rows:continue
            positives=[i for i,r in enumerate(rows) if base.norm(r['value'])==base.norm(g)]
            m['covered']+=int(bool(positives))
            target=f"Current request: {qa.get('query','')}. Future tool: {sch.get('name','')}. Required semantic role: {p}. Meaning: {d.get('description','')}. Type: {d.get('type','')}"
            pairs=[(target,r['source']) for r in rows]
            scores=np.asarray(rerank.predict(pairs,batch_size=64,show_progress_bar=False),dtype=np.float32).reshape(-1)
            # Combine multiple evidence occurrences resolving to the same exact payload without exposing payload to model.
            groups=defaultdict(list)
            for i,r in enumerate(rows):groups[base.norm(r['value'])].append(i)
            packed=[]
            for k,ids in groups.items():
                if not k:continue
                best=max(ids,key=lambda i:float(scores[i]));packed.append((float(max(scores[i] for i in ids)),best,ids))
            order=sorted(range(len(packed)),key=lambda j:-packed[j][0])
            gold_groups={j for j,(_,_,ids) in enumerate(packed) if any(i in positives for i in ids)}
            m['top1']+=int(bool(order) and order[0] in gold_groups)
            m['top3']+=int(any(j in gold_groups for j in order[:3]));m['top5']+=int(any(j in gold_groups for j in order[:5]))
            if len(examples)<12 and gold_groups and (not order or order[0] not in gold_groups):
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'top':[{'score':packed[j][0],'field':rows[packed[j][1]]['field'],'op':rows[packed[j][1]]['op'],'scope':rows[packed[j][1]]['scope']} for j in order[:4]],'gold_fields':[rows[packed[j][1]]['field'] for j in gold_groups if j<len(packed)][:4]})
        if qi%10==0:print('ANCHOR_XENC_BUILD',qi,'nodes',len(nodes),flush=True)
    out={}
    for typ,c in stats.items():
        n=max(1,c['n']);cov=max(1,c['covered']);out[typ]={'n':c['n'],'candidate_coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov}
    result={'stage':'SWM-B anchor-scoped pretrained cross-encoder semantic-role binder','split':'QA001-100 development diagnostic only; QA101-400 gold remains sealed','model':MODEL,'architecture':'query-independent world nodes -> exact entity/shared-value graph + top-4 masked semantic fallback -> value-masked target/source role cross-encoder -> aggregate evidence by exact executor payload -> dereference','results':out,'examples':examples,'repair_report':report,'guardrail':'Candidate payload values are never included in cross-encoder pair text. Gold values score exact dereference only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ANCHOR_CROSSENCODER='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
