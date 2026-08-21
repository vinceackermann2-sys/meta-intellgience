from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile
import numpy as np
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_property_ingest_oracle as sw
import semantic_concept_ingest_oracle as sc
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_entity_anchor_oracle_result.json')
N=100


def normset(vals):
    return {base.norm(v) for v in vals if base.norm(v)}


def node_aliases(node):
    return normset([p.get('value') for p in node.get('props',[])])


def query_aliases(q):
    vals=[v for v,_ in sw.generic_entities(q)]
    vals += [v for v,_ in sc.concepts(q)]
    return normset(vals)


def pack(stats):
    return {k:{'n':v['n'],'gold_property_anywhere':v['covered']/max(1,v['n']),'direct_anchor_hits_gold_node':v['direct']/max(1,v['n']),'same_episode_expansion_hits':v['episode']/max(1,v['n']),'shared_value_onehop_hits':v['onehop']/max(1,v['n']),'query_has_any_entity_anchor':v['has_anchor']/max(1,v['n'])} for k,v in stats.items()}


def main():
    td=Path(tempfile.gettempdir())/'semantic_entity_anchor';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N];repaired,report=repair.build();stats=defaultdict(lambda:Counter(n=0,covered=0,direct=0,episode=0,onehop=0,has_anchor=0));examples=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'));aliases=[node_aliases(n) for n in nodes]
        # Query aliases are extracted without target schema or gold.
        qa_alias=query_aliases(str(qa.get('query','')));anchored={i for i,a in enumerate(aliases) if a & qa_alias}
        # Query-independent node graph: same episode or shared exact entity/property alias.
        same_episode={i:set() for i in range(len(nodes))};shared={i:set() for i in range(len(nodes))}
        ep_groups=defaultdict(list)
        for i,n in enumerate(nodes):ep_groups[n.get('episode')].append(i)
        for ids in ep_groups.values():
            S=set(ids)
            for i in ids:same_episode[i]|=(S-{i})
        inv=defaultdict(list)
        for i,a in enumerate(aliases):
            for x in a:inv[x].append(i)
        for ids in inv.values():
            if len(ids)>1:
                S=set(ids)
                for i in ids:shared[i]|=(S-{i})
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};c=stats[typ];c['n']+=1;c['has_anchor']+=int(bool(anchored))
            positives=set()
            for ni,n in enumerate(nodes):
                ok=False
                for prop in n.get('props',[]):
                    for z,_ in wb.variant_ops(prop.get('value'),p,d):
                        if base.norm(z)==base.norm(g):ok=True;break
                    if ok:break
                if ok:positives.add(ni)
            if not positives:continue
            c['covered']+=1
            direct=bool(anchored & positives);c['direct']+=int(direct)
            ep_exp=set(anchored)
            for i in list(anchored):ep_exp|=same_episode.get(i,set())
            c['episode']+=int(bool(ep_exp & positives))
            one=set(anchored)
            for i in list(anchored):one|=shared.get(i,set())
            c['onehop']+=int(bool(one & positives))
            if len(examples)<16 and positives and anchored and not direct:
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'query':qa.get('query',''),'query_aliases':sorted(qa_alias)[:12],'anchored_nodes':[nodes[i]['context_desc'][:220] for i in list(anchored)[:2]],'gold_nodes':[nodes[i]['context_desc'][:220] for i in list(positives)[:2]],'episode_rescues':bool(ep_exp & positives),'shared_value_rescues':bool(one & positives)})
        if qi%20==0:print('ENTITY_ANCHOR_BUILD',qi,'nodes',len(nodes),'query_aliases',len(qa_alias),flush=True)
    result={'stage':'SWM-B exact entity-anchor/co-occurrence routing oracle','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'query-independent entity aliases on compiled world nodes; current-request entity aliases -> exact node anchors -> same-episode or shared-value graph expansion; no embedding model or learned entity resolver','results':pack(stats),'examples':examples,'repair_report':report,'guardrail':'Gold only marks whether reachable nodes contain the target property. Entity aliases and graph edges are compiled without QA gold. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ENTITY_ANCHOR='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
