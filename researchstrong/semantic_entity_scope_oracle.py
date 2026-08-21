from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile
import numpy as np
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT = Path(__file__).with_name('semantic_entity_scope_oracle_result.json')
N = 100
TOPKS = (1,2,3,5,10)


def pack(stats):
    out={}
    for typ,c in stats.items():
        n=max(1,c['n'])
        row={'n':c['n'],'gold_node_anywhere':c['covered']/n}
        for k in TOPKS: row[f'top{k}_gold_node']=c[f'top{k}']/n
        out[typ]=row
    return out


def main():
    td=Path(tempfile.gettempdir())/'semantic_entity_scope_oracle';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N]
    repaired,report=repair.build();enc=wb.CachedEncoder()
    stats={m:defaultdict(lambda:Counter(n=0,covered=0,**{f'top{k}':0 for k in TOPKS})) for m in ('strict_masked','context_alias_visible')}
    examples=[];node_counts=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'));node_counts.append(len(nodes))
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {})
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        E={}
        if nodes:
            E['strict_masked']=enc.encode([n['strict_desc'] for n in nodes])
            E['context_alias_visible']=enc.encode([n['context_desc'] for n in nodes])
        else:
            E['strict_masked']=E['context_alias_visible']=np.zeros((0,384),np.float32)
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};tv=enc.encode(h.target_scope(qa,p,d))
            positives=[]
            for ni,n in enumerate(nodes):
                found=False
                for prop in n['props']:
                    for z,_ in wb.variant_ops(prop['value'],p,d):
                        if base.norm(z)==base.norm(g):found=True;break
                    if found:break
                if found:positives.append(ni)
            pset=set(positives)
            for mode in stats:
                c=stats[mode][typ];c['n']+=1
                if not positives or not nodes:continue
                c['covered']+=1;sc=E[mode]@tv;order=np.argsort(-sc)
                for k in TOPKS:c[f'top{k}']+=int(any(int(i) in pset for i in order[:k]))
                if mode=='context_alias_visible' and len(examples)<12 and int(order[0]) not in pset:
                    best=max(positives,key=lambda i:float(sc[i]));examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'gold':g,'top_node':nodes[int(order[0])]['context_desc'][:260],'top_score':float(sc[int(order[0])]),'gold_node':nodes[best]['context_desc'][:260],'gold_score':float(sc[best])})
        if qi%10==0:print('ENTITY_SCOPE_BUILD',qi,'nodes',len(nodes),'cache',len(enc.cache),flush=True)
    result={'stage':'SWM-B entity/record scope oracle before field-role binding','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'query-independent world nodes -> current request/target semantics retrieve entity or record node; no field-role ranker in this diagnostic','results':{m:pack(s) for m,s in stats.items()},'node_count':{'mean':float(np.mean(node_counts)),'median':float(np.median(node_counts)),'max':int(max(node_counts) if node_counts else 0)},'examples':examples,'repair_report':report,'embedding_cache_entries':len(enc.cache),'guardrail':'Gold values only mark which node contains a scoreable property. strict_masked masks all node payloads. context_alias_visible exposes query-independent historical context only for entity-scope ablation. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ENTITY_SCOPE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
