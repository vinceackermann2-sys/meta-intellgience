from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
import strong_banking77 as base
import episode_scoped_router as es
import semantic_property_ingest_oracle as sw
import semantic_concept_ingest_oracle as sc
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT = Path(__file__).with_name('semantic_entity_hierarchy_diagnostic_result.json')
N = 100
TOPKS = (1, 2, 3, 5)
ALPHAS = (0.25, 0.5, 0.75)


def squash(x):
    return re.sub(r'\s+', ' ', str(x)).strip()


def leaf(k):
    x = str(k).split('.')[-1]
    return re.sub(r'\[\d+\]$', '', x)


def mask_many(text, values):
    s = str(text)
    vals = sorted({str(v) for v in values if v is not None and len(str(v).strip()) >= 2}, key=len, reverse=True)
    for v in vals:
        s = re.sub(re.escape(v), '<ENTITY>', s, flags=re.I)
    return s


def prior_user(ep, turn):
    out = ''
    for ti, t in ep['rows']:
        if ti > turn:
            break
        if str(t.get('role', '')) == 'user':
            c = t.get('content', '')
            out = json.dumps(c, ensure_ascii=False) if isinstance(c, (dict, list)) else str(c)
    return squash(out)[:900]


def tool_name(turn):
    calls = turn.get('tool_calls') or []
    if calls:
        try:
            return base.tool_name(calls[0])
        except Exception:
            pass
    return str(turn.get('name', '') or '')


def build_nodes(session):
    nodes = []
    if session is None:
        return nodes
    for ep_rank, ep in enumerate(es.episodes(session)):
        sid = str(ep.get('source_id', ''))
        for ti, t in ep['rows']:
            role = str(t.get('role', ''))
            tool = tool_name(t)
            intent = prior_user(ep, ti)
            # Structured records remain atomic world objects.
            for rec_i, rec in enumerate(sw.record_values(t)):
                vals = [v for _, v in rec]
                fields = [k for k, _ in rec]
                siblings = ', '.join(leaf(k) for k in fields[:30])
                strict_intent = mask_many(intent, vals)
                base_meta = f"episode {sid}; historical role {role}; source tool {tool}; record type structured; field roles {siblings}; record index {rec_i}"
                nodes.append({
                    'kind': 'structured_record', 'episode': ep_rank, 'turn': ti, 'tool': tool,
                    'strict_desc': squash(f"historical intent {strict_intent}; {base_meta}"),
                    'context_desc': squash(f"historical intent {intent}; {base_meta}"),
                    'props': [{'field': k, 'value': v, 'src': 'structured'} for k, v in rec],
                    'intent': intent, 'siblings': siblings,
                })
            # Natural-language entity/concept packet as another world object.
            content = t.get('content', '')
            if isinstance(content, str) and content.strip():
                props = []
                for v, kind in sw.generic_entities(content):
                    props.append({'field': 'text_entity:' + kind, 'value': v, 'src': 'entity:' + kind})
                if role == 'user':
                    for v, kind in sc.concepts(content):
                        props.append({'field': 'semantic_concept:' + kind, 'value': v, 'src': kind})
                if props:
                    vals = [p['value'] for p in props]
                    strict_text = mask_many(content[:1200], vals)
                    meta = f"episode {sid}; historical role {role}; source tool {tool}; record type semantic_text; semantic property kinds " + ', '.join(sorted({p['field'].split(':')[0] for p in props}))
                    nodes.append({
                        'kind': 'semantic_text', 'episode': ep_rank, 'turn': ti, 'tool': tool,
                        'strict_desc': squash(f"historical context {strict_text}; {meta}"),
                        # Alias/context-visible mode is a diagnostic: query-independent historical text may expose entity names.
                        'context_desc': squash(f"historical context {content[:1200]}; {meta}"),
                        'props': props, 'intent': intent, 'siblings': '',
                    })
    return nodes


def target_scope(qa, p, d):
    sch = qa.get('target_tool_schema') or {}
    return squash(f"current request {qa.get('query','')}; future action {sch.get('name','')}; requested object for semantic role {p}; {d.get('description','')}")


def target_role(qa, p, d):
    sch = qa.get('target_tool_schema') or {}
    return squash(f"target tool {sch.get('name','')}; target semantic role {p}; meaning {d.get('description','')}; type {d.get('type','')}")


def prop_role(node, prop, op):
    # Mask only this property's payload from historical intent. Exact payload remains executor-only.
    intent = wb.mask(node.get('intent', ''), prop.get('value'))
    return squash(
        f"historical intent {intent}; object type {node.get('kind')}; source tool {node.get('tool','')}; "
        f"source semantic role {leaf(prop.get('field',''))}; full field {prop.get('field','')}; "
        f"sibling roles {node.get('siblings','')}; transform {op}"
    )


def expanded(node, p, d):
    out = []
    seen = set()
    for prop in node['props']:
        for z, op in wb.variant_ops(prop['value'], p, d):
            k = (base.norm(z), prop['field'], op)
            if not k[0] or k in seen:
                continue
            seen.add(k)
            out.append({'value': z, 'op': op, 'prop': prop, 'role': prop_role(node, prop, op)})
    return out


def metric_bucket():
    return defaultdict(lambda: Counter(n=0, covered=0, top1=0, top2=0, top3=0, top5=0,
                                       field1=0, field3=0))


def pack(counter):
    out = {}
    for typ, c in counter.items():
        n = max(1, c['n']); cov = max(1, c['covered'])
        out[typ] = {
            'n': c['n'], 'gold_node_anywhere': c['covered']/n,
            'node_top1_gold_coverage': c['top1']/n,
            'node_top2_gold_coverage': c['top2']/n,
            'node_top3_gold_coverage': c['top3']/n,
            'node_top5_gold_coverage': c['top5']/n,
            'field_top1_given_best_gold_node': c['field1']/cov,
            'field_top3_given_best_gold_node': c['field3']/cov,
        }
    return out


def main():
    td = Path(tempfile.gettempdir())/'semantic_entity_hierarchy'; td.mkdir(exist_ok=True)
    qp = td/'qa.jsonl'; base.fetch(base.BASE+'qa_dataset.jsonl', qp)
    qas = list(base.load_jsonl(qp))[:N]
    repaired, repair_report = repair.build()
    enc = wb.CachedEncoder()
    modes = {'strict_masked': metric_bucket(), 'context_alias_visible': metric_bucket()}
    e2e = {m: {a: defaultdict(lambda: Counter(n=0, covered=0, top1=0, top3=0)) for a in ALPHAS} for m in modes}
    examples = []
    node_counts = []

    for qi, qa in enumerate(qas):
        rr = repaired.get(qa.get('qa_id')) or {}
        session = rr.get('session')
        nodes = build_nodes(session); node_counts.append(len(nodes))
        sch = qa.get('target_tool_schema') or {}
        defs = ((sch.get('parameters') or {}).get('properties') or {})
        gold = ((qa.get('tool_call') or {}).get('arguments') or {})
        gi = ((qa.get('tool_call') or {}).get('grounding_info') or {})

        # Cache node embeddings once per routing mode/session.
        node_E = {}
        for mode, key in [('strict_masked','strict_desc'), ('context_alias_visible','context_desc')]:
            node_E[mode] = enc.encode([n[key] for n in nodes]) if nodes else np.zeros((0,384), np.float32)

        for p, g in gold.items():
            typ = str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):
                continue
            d = defs.get(p) or {}
            scope_v = enc.encode(target_scope(qa,p,d))
            role_v = enc.encode(target_role(qa,p,d))
            node_expanded = [expanded(n,p,d) for n in nodes]
            positive_nodes = [ni for ni, cs in enumerate(node_expanded)
                              if any(base.norm(c['value']) == base.norm(g) for c in cs)]

            for mode in modes:
                m = modes[mode][typ]; m['n'] += 1
                if not positive_nodes or len(nodes) == 0:
                    continue
                m['covered'] += 1
                ns = node_E[mode] @ scope_v
                order = np.argsort(-ns)
                pset = set(positive_nodes)
                for k in TOPKS:
                    if any(int(i) in pset for i in order[:k]): m[f'top{k}'] += 1

                # Field-role diagnostic inside the highest-ranked gold-containing node.
                best_gold_node = max(positive_nodes, key=lambda i: float(ns[i]))
                cs = node_expanded[best_gold_node]
                if cs:
                    F = enc.encode([c['role'] for c in cs]); fs = F @ role_v; fo = np.argsort(-fs)
                    pos_fields = {i for i,c in enumerate(cs) if base.norm(c['value']) == base.norm(g)}
                    m['field1'] += int(bool(fo) and int(fo[0]) in pos_fields)
                    m['field3'] += int(any(int(i) in pos_fields for i in fo[:3]))

                # End-to-end: shortlist top-3 nodes, then combine scope and field compatibility.
                top_nodes = [int(i) for i in order[:3]]
                cand_rows = []
                for ni in top_nodes:
                    cs = node_expanded[ni]
                    if not cs: continue
                    F = enc.encode([c['role'] for c in cs]); fs = F @ role_v
                    for ci,c in enumerate(cs): cand_rows.append((ni,c,float(ns[ni]),float(fs[ci])))
                for a in ALPHAS:
                    z = e2e[mode][a][typ]; z['n'] += 1; z['covered'] += int(bool(positive_nodes))
                    if not cand_rows: continue
                    scores = [a*r[2] + (1-a)*r[3] for r in cand_rows]
                    eo = np.argsort(-np.asarray(scores))
                    z['top1'] += int(base.norm(cand_rows[int(eo[0])][1]['value']) == base.norm(g))
                    z['top3'] += int(any(base.norm(cand_rows[int(i)][1]['value']) == base.norm(g) for i in eo[:3]))

                if typ == 'explicit' and len(examples) < 10 and positive_nodes and not any(int(i) in pset for i in order[:1]):
                    examples.append({
                        'qa_id': qa.get('qa_id'), 'parameter': p, 'gold': g, 'mode': mode,
                        'top_nodes': [{'score':float(ns[int(i)]),'kind':nodes[int(i)]['kind'],'tool':nodes[int(i)]['tool'],'desc':nodes[int(i)][('strict_desc' if mode=='strict_masked' else 'context_desc')][:260]} for i in order[:3]],
                        'gold_node_desc': nodes[best_gold_node][('strict_desc' if mode=='strict_masked' else 'context_desc')][:300],
                    })
        if qi % 10 == 0:
            print('HIERARCHY_BUILD', qi, 'nodes', len(nodes), 'cache', len(enc.cache), flush=True)

    e2e_pack = {}
    for mode, byalpha in e2e.items():
        e2e_pack[mode] = {}
        for a, bytyp in byalpha.items():
            e2e_pack[mode][str(a)] = {}
            for typ,c in bytyp.items():
                n=max(1,c['n'])
                e2e_pack[mode][str(a)][typ] = {'n':c['n'],'top1_all':c['top1']/n,'top3_all':c['top3']/n}

    result = {
        'stage': 'SWM-B hierarchy diagnostic: entity/record scope before field-role binding',
        'split': 'QA001-100 development only; QA101-400 gold remains sealed',
        'architecture': 'query-independent semantic world state -> record/entity node routing -> semantic role binding inside top node(s) -> deterministic transform -> exact dereference',
        'modes': {
            'strict_masked': 'all node property payloads masked during node routing',
            'context_alias_visible': 'query-independent historical context may expose entity aliases for node routing; property payload still excluded from field-role score',
        },
        'hierarchy_oracles': {m:pack(v) for m,v in modes.items()},
        'end_to_end_no_learning': e2e_pack,
        'node_count': {'mean':float(np.mean(node_counts)),'median':float(np.median(node_counts)),'max':int(max(node_counts) if node_counts else 0)},
        'repair_report': repair_report,
        'examples': examples,
        'embedding_cache_entries': len(enc.cache),
        'guardrail': 'Gold QA001-100 is scoring-only. Exact property payloads are never field-role features. Context-visible mode is explicitly an entity-routing ablation, not a value-blind role claim. No QA101-400 gold is read.'
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('SEMANTIC_ENTITY_HIERARCHY='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
