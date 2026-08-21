from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_entity_anchor_oracle as a
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT = Path(__file__).with_name('semantic_discourse_referent_oracle_result.json')
N = 100
TOPKS = (1, 2, 3, 5)
SALIENCE_WORDS = {
    'track','tracked','tracking','always','usually','prefer','preferred','favorite',
    'current','upcoming','planned','planning','plan','remember','discussed','talked',
    'earlier','again','my','our','we','latest','recent'
}


def toks(x): return set(re.findall(r'[a-z0-9]+', str(x).casefold()))
def squash(x): return re.sub(r'\s+', ' ', str(x)).strip()


def cluster_nodes(nodes):
    n=len(nodes); parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y: parent[y]=x
    by_ep=defaultdict(list); inv=defaultdict(list); aliases=[]
    for i,node in enumerate(nodes):
        by_ep[node.get('episode')].append(i)
        vals={base.norm(p.get('value')) for p in node.get('props',[]) if base.norm(p.get('value'))}
        aliases.append(vals)
        for v in vals: inv[v].append(i)
    for ids in by_ep.values():
        if ids:
            for i in ids[1:]: union(ids[0],i)
    for ids in inv.values():
        if 1 < len(ids) <= 8:
            for i in ids[1:]: union(ids[0],i)
    groups=defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return list(groups.values()), aliases


def cluster_desc(nodes, ids):
    tools=[]; fields=[]; intents=[]; turns=[]
    for i in ids:
        n=nodes[i];turns.append(int(n.get('turn',-1)))
        if n.get('tool'): tools.append(str(n.get('tool')))
        if n.get('intent'): intents.append(str(n.get('intent')))
        fields.extend(str(n.get('siblings','')).split(','))
        fields.extend(str(p.get('field','')) for p in n.get('props',[]))
    return squash('discourse object; tools '+', '.join(tools[:12])+'; semantic roles '+', '.join(fields[:45])+'; historical user intents '+' | '.join(intents[-5:]))[:3500]


def salience(nodes,ids,max_turn):
    text=' '.join(str(nodes[i].get('intent','')) for i in ids).casefold();tt=toks(text)
    marker=sum(1 for w in SALIENCE_WORDS if w in tt)
    turns=[int(nodes[i].get('turn',-1)) for i in ids]
    rec=max(turns or [-1])/max(1,max_turn)
    eps=len({nodes[i].get('episode') for i in ids})
    structured=sum(1 for i in ids if nodes[i].get('kind')=='structured_record')
    return {'recency':float(rec),'marker':float(min(marker,6)/6),'recurrence':float(min(eps,4)/4),'structured':float(min(structured,4)/4)}


def implicit_ref(q):
    s=' '+str(q).casefold()+' '
    cues=(' that ',' those ',' these ',' it ',' them ',' my ',' our ',' we ',' earlier ',' always ',' usually ',' tracked ',' track ',' discussed ',' talked ',' upcoming ',' current ',' latest ',' same ')
    return any(c in s for c in cues)


def pack(stats):
    out={}
    for typ,c in stats.items():
        n=max(1,c['n']);row={'n':c['n'],'gold_property_anywhere':c['covered']/n,'implicit_reference_rate':c['implicit']/n,'exact_anchor_baseline':c['anchor']/n}
        for k in TOPKS:
            row[f'discourse_top{k}_hits_gold']=c[f'top{k}']/n
            row[f'union_anchor_top{k}_hits_gold']=c[f'union{k}']/n
        out[typ]=row
    return out


def main():
    td=Path(tempfile.gettempdir())/'semantic_discourse_ref';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N];repaired,report=repair.build();enc=wb.CachedEncoder()
    stats=defaultdict(lambda:Counter(n=0,covered=0,implicit=0,anchor=0,**{f'top{k}':0 for k in TOPKS},**{f'union{k}':0 for k in TOPKS}))
    examples=[];cluster_counts=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'));groups,aliases=cluster_nodes(nodes);cluster_counts.append(len(groups));max_turn=max([int(n.get('turn',-1)) for n in nodes]+[1])
        descs=[cluster_desc(nodes,ids) for ids in groups];sals=[salience(nodes,ids,max_turn) for ids in groups];E=enc.encode(descs) if descs else np.zeros((0,384),np.float32)
        q_alias=a.query_aliases(str(qa.get('query','')));anchored_nodes={i for i,x in enumerate(aliases) if x&q_alias};anchored_groups={gi for gi,ids in enumerate(groups) if any(i in anchored_nodes for i in ids)}
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});meta=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((meta.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};c=stats[typ];c['n']+=1;c['implicit']+=int(implicit_ref(qa.get('query','')))
            positive=set()
            for ni,n in enumerate(nodes):
                ok=False
                for prop in n.get('props',[]):
                    for z,_ in wb.variant_ops(prop.get('value'),p,d):
                        if base.norm(z)==base.norm(g):ok=True;break
                    if ok:break
                if ok:positive.add(ni)
            if not positive:continue
            c['covered']+=1;pg={j for j,ids in enumerate(groups) if any(i in positive for i in ids)};c['anchor']+=int(bool(anchored_groups&pg))
            if not groups:continue
            target=squash(f"current user intent {qa.get('query','')}; future action {sch.get('name','')}; requested semantic role {p}; meaning {d.get('description','')}; expected type {d.get('type','')}")
            tv=enc.encode(target);sem=E@tv;scores=[]
            for j,x in enumerate(sem):
                s=sals[j];scores.append(.72*float(x)+.12*s['recency']+.08*s['marker']+.05*s['recurrence']+.03*s['structured'])
            order=np.argsort(-np.asarray(scores))
            for k in TOPKS:
                top={int(i) for i in order[:k]};c[f'top{k}']+=int(bool(top&pg));c[f'union{k}']+=int(bool((top|anchored_groups)&pg))
            if len(examples)<14 and not (anchored_groups&pg) and any(int(i) in pg for i in order[:3]):
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'gold':g,'query':qa.get('query',''),'implicit':implicit_ref(qa.get('query','')),'top_clusters':[{'score':float(scores[int(i)]),'desc':descs[int(i)][:280],'salience':sals[int(i)]} for i in order[:3]]})
        if qi%10==0:print('DISCOURSE_REF_BUILD',qi,'nodes',len(nodes),'clusters',len(groups),'cache',len(enc.cache),flush=True)
    result={'stage':'SWM-B development-only discourse referent reachability control','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'query-independent world nodes -> episode/shared-identity discourse clusters -> recency/recurrence/salience state + semantic type/topic match -> referent cluster -> later exact field dereference','results':pack(stats),'cluster_count':{'mean':float(np.mean(cluster_counts)),'median':float(np.median(cluster_counts)),'max':int(max(cluster_counts) if cluster_counts else 0)},'examples':examples,'repair_report':report,'embedding_cache_entries':len(enc.cache),'guardrail':'Gold only scores reachability. Exact values are not discourse-ranking features. This is a coreference/control baseline, not a novelty claim. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_DISCOURSE_REFERENT='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
