from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile, itertools
import numpy as np
from sklearn.model_selection import KFold
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_entity_anchor_oracle as a
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT = Path(__file__).with_name('semantic_discourse_focus_cv_result.json')
N = 100
SEED = 20260821

SALIENT = re.compile(r'\b(my|our|track|always|usually|preferred|favorite|favourite|same|again|earlier|discussed|talked|planned|upcoming|current location|we use)\b', re.I)


def normset(vals):
    return {base.norm(v) for v in vals if base.norm(v)}


def cluster_episode(nodes):
    by = defaultdict(list)
    for i, n in enumerate(nodes):
        by[int(n.get('episode', -1))].append((i, n))
    if not by:
        return []
    max_ep = max(by)
    max_turn = max([int(n.get('turn', -1)) for n in nodes] + [1])
    clusters = []
    for ep, rows in sorted(by.items()):
        ids = [i for i, _ in rows]
        ns = [n for _, n in rows]
        aliases = set()
        fields = []
        tools = []
        intents = []
        structured = 0
        last_turn = -1
        salient_hits = 0
        desc_parts = []
        for n in ns:
            aliases |= a.node_aliases(n)
            last_turn = max(last_turn, int(n.get('turn', -1)))
            if n.get('kind') == 'structured_record':
                structured += 1
            if n.get('tool'):
                tools.append(str(n.get('tool')))
            if n.get('intent'):
                intent = str(n.get('intent'))
                intents.append(intent)
                salient_hits += len(SALIENT.findall(intent))
            for p in n.get('props', []):
                fields.append(str(p.get('field', '')))
            # strict_desc masks property payloads; safe for semantic focus scoring.
            if len(desc_parts) < 12:
                desc_parts.append(str(n.get('strict_desc', ''))[:700])
        role_desc = ' ; '.join((fields[:60] + tools[:20]))[:1800]
        desc = ' || '.join(desc_parts)[:5000]
        clusters.append({
            'episode': ep,
            'node_ids': ids,
            'aliases': aliases,
            'desc': desc,
            'role_desc': role_desc,
            'last_turn': last_turn,
            'recency': last_turn / max(1, max_turn),
            'episode_recency': ep / max(1, max_ep),
            'structured_ratio': structured / max(1, len(ns)),
            'size_log': float(np.log1p(len(ns))),
            'salience': min(1.0, salient_hits / 6.0),
            'intent_count': len(intents),
        })
    return clusters


def positive_episodes(nodes, p, d, g):
    pos = set()
    for i, n in enumerate(nodes):
        ok = False
        for c in h.expanded(n, p, d):
            if base.norm(c['value']) == base.norm(g):
                ok = True
                break
        if ok:
            pos.add(int(n.get('episode', -1)))
    return pos


def build(enc):
    td = Path(tempfile.gettempdir()) / 'semantic_discourse_focus'; td.mkdir(exist_ok=True)
    qp = td / 'qa.jsonl'; base.fetch(base.BASE + 'qa_dataset.jsonl', qp)
    qas = list(base.load_jsonl(qp))[:N]
    repaired, report = repair.build()
    slots = []
    for qi, qa in enumerate(qas):
        rr = repaired.get(qa.get('qa_id')) or {}
        nodes = h.build_nodes(rr.get('session'))
        clusters = cluster_episode(nodes)
        if clusters:
            descE = enc.encode([c['desc'] for c in clusters])
            roleE = enc.encode([c['role_desc'] for c in clusters])
        else:
            descE = np.zeros((0,384), np.float32); roleE = np.zeros((0,384), np.float32)
        qaliases = a.query_aliases(str(qa.get('query','')))
        sch = qa.get('target_tool_schema') or {}
        defs = ((sch.get('parameters') or {}).get('properties') or {})
        gold = ((qa.get('tool_call') or {}).get('arguments') or {})
        gi = ((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p, g in gold.items():
            typ = str((gi.get(p) or {}).get('type', 'unknown'))
            if typ not in ('explicit','inferred'):
                continue
            d = defs.get(p) or {}
            scope_v = enc.encode(h.target_scope(qa,p,d))
            role_v = enc.encode(h.target_role(qa,p,d))
            pos_eps = positive_episodes(nodes,p,d,g)
            feats = []
            for ci,c in enumerate(clusters):
                feats.append({
                    'semantic': float(descE[ci] @ scope_v),
                    'role': float(roleE[ci] @ role_v),
                    'alias': float(bool(c['aliases'] & qaliases)),
                    'recency': c['recency'],
                    'episode_recency': c['episode_recency'],
                    'structured': c['structured_ratio'],
                    'salience': c['salience'],
                    'size_log': c['size_log'],
                })
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'parameter':p,'typ':typ,'clusters':clusters,'feats':feats,'positive_episodes':pos_eps})
        if qi % 10 == 0:
            print('DISCOURSE_BUILD', qi, 'clusters', len(clusters), 'slots', len(slots), flush=True)
    return slots, report


WEIGHTS = []
for role_w in (0.0,0.25,0.5):
    for alias_w in (0.0,0.5,1.0):
        for rec_w in (0.0,0.1,0.25):
            for sal_w in (0.0,0.1,0.25):
                WEIGHTS.append((role_w,alias_w,rec_w,sal_w))


def rank_slot(s, w):
    rw, aw, recw, sw = w
    scores = []
    for f in s['feats']:
        z = f['semantic'] + rw*f['role'] + aw*f['alias'] + recw*f['recency'] + sw*f['salience'] + 0.05*f['structured']
        scores.append(z)
    return np.argsort(-np.asarray(scores)) if scores else np.asarray([], dtype=np.int64)


def eval_slots(slots, qset, w):
    by = defaultdict(lambda: Counter(n=0,covered=0,top1=0,top2=0,top3=0,top5=0))
    for s in slots:
        if s['qi'] not in qset: continue
        m = by[s['typ']]; m['n'] += 1
        pos = s['positive_episodes']
        if not pos or not s['clusters']: continue
        m['covered'] += 1
        order = rank_slot(s,w)
        ranked_eps = [s['clusters'][int(i)]['episode'] for i in order]
        for k in (1,2,3,5):
            if any(ep in pos for ep in ranked_eps[:k]): m[f'top{k}'] += 1
    out = {}
    for typ,c in by.items():
        n=max(1,c['n'])
        out[typ]={'n':c['n'],'gold_episode_anywhere':c['covered']/n,'top1':c['top1']/n,'top2':c['top2']/n,'top3':c['top3']/n,'top5':c['top5']/n}
    return out


def objective(r):
    return 0.7*r.get('explicit',{}).get('top1',0.0) + 0.3*r.get('inferred',{}).get('top1',0.0)


def main():
    enc = wb.CachedEncoder()
    slots, report = build(enc)
    idx = np.arange(N)
    folds = list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(idx))
    fold_rows=[]
    for fi,(tr,va) in enumerate(folds):
        trset=set(idx[tr].tolist()); vaset=set(idx[va].tolist())
        best=None
        for w in WEIGHTS:
            r=eval_slots(slots,trset,w);o=objective(r)
            if best is None or o>best[0]:best=(o,w,r)
        vr=eval_slots(slots,vaset,best[1])
        fold_rows.append({'fold':fi,'weights':best[1],'train_objective':best[0],'validation':vr})
        print('DISCOURSE_FOLD',fi,'weights',best[1],json.dumps(vr),flush=True)
    mean=defaultdict(dict)
    for typ in ('explicit','inferred'):
        for k in ('gold_episode_anywhere','top1','top2','top3','top5'):
            vals=[f['validation'].get(typ,{}).get(k,0.0) for f in fold_rows]
            mean[typ][k]=float(np.mean(vals))
    # Fixed semantic-only baseline over all dev for reference (not used for tuning).
    baseline=eval_slots(slots,set(range(N)),(0.0,0.0,0.0,0.0))
    result={
      'stage':'SWM-B discourse referent/focus-state diagnostic',
      'split':'QA001-100 development-only five-fold QA CV; QA101-400 gold remains sealed',
      'architecture':'query-independent episode focus clusters with masked summaries + compiled recency/structured/salience state; query-time semantic role + exact alias compatibility; no candidate payload used in focus score',
      'mean_cv':dict(mean),'folds':fold_rows,'semantic_only_dev_reference':baseline,'repair_report':report,
      'comparison_target':{'explicit_shared_anchor_reachability':0.7142857142857143,'inferred_shared_anchor_reachability':0.375},
      'continue_if':'top-k focused reachability materially improves exact-anchor/shared-value routing; otherwise reject discourse salience as missing primitive',
      'guardrail':'Gold values only mark positive episodes in QA001-100. Cluster summaries use payload-masked strict descriptions; exact aliases are used only for literal current-query entity identity matches. No QA101-400 gold is read.'
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_DISCOURSE_FOCUS='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
