from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import strong_banking77 as base
import semantic_world_binding_cv as wb
import semantic_entity_hierarchy_diagnostic as h
import semantic_entity_anchor_oracle as anchor
import semantic_discourse_focus_cv as focus
import mem2act_repaired_sessions as repair

OUT = Path(__file__).with_name('semantic_focus_field_binding_cv_result.json')
N = 100
SEED = 20260821
FOCUS_KS = (1, 2, 3, 999)
C_FIXED = 0.1


def canon(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).casefold())


def toks(s):
    return set(re.findall(r'[a-z0-9]+', str(s).casefold()))


def jacc(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))


def leaf(k):
    x = str(k).split('.')[-1]
    return re.sub(r'\[\d+\]$', '', x)


def focus_score(feat, w):
    rw, aw, recw, sw = w
    return (feat['semantic'] + rw * feat['role'] + aw * feat['alias'] +
            recw * feat['recency'] + sw * feat['salience'] + 0.05 * feat['structured'])


def rank_focus(slot, w):
    if not slot['clusters']:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    sc = np.asarray([focus_score(f, w) for f in slot['focus_feats']], dtype=np.float32)
    return np.argsort(-sc), sc


def focus_eval(slots, qset, w):
    by = defaultdict(lambda: Counter(n=0, top1=0))
    for s in slots:
        if s['qi'] not in qset:
            continue
        by[s['typ']]['n'] += 1
        if not s['positive_episodes']:
            continue
        order, _ = rank_focus(s, w)
        eps = [s['clusters'][int(i)]['episode'] for i in order[:1]]
        by[s['typ']]['top1'] += int(any(ep in s['positive_episodes'] for ep in eps))
    out = {}
    for typ, c in by.items():
        out[typ] = c['top1'] / max(1, c['n'])
    return 0.7 * out.get('explicit', 0.0) + 0.3 * out.get('inferred', 0.0)


def choose_focus_weights(slots, train_ids):
    best = None
    for w in focus.WEIGHTS:
        z = focus_eval(slots, train_ids, w)
        if best is None or z > best[0]:
            best = (z, w)
    return best[1]


def build(enc):
    td = Path(tempfile.gettempdir()) / 'semantic_focus_field_binding'
    td.mkdir(exist_ok=True)
    qp = td / 'qa.jsonl'
    base.fetch(base.BASE + 'qa_dataset.jsonl', qp)
    qas = list(base.load_jsonl(qp))[:N]
    repaired, repair_report = repair.build()
    slots = []

    for qi, qa in enumerate(qas):
        rr = repaired.get(qa.get('qa_id')) or {}
        nodes = h.build_nodes(rr.get('session'))
        clusters = focus.cluster_episode(nodes)
        node_E = enc.encode([n['strict_desc'] for n in nodes]) if nodes else np.zeros((0,384), np.float32)
        descE = enc.encode([c['desc'] for c in clusters]) if clusters else np.zeros((0,384), np.float32)
        roleE = enc.encode([c['role_desc'] for c in clusters]) if clusters else np.zeros((0,384), np.float32)
        qaliases = anchor.query_aliases(str(qa.get('query','')))

        sch = qa.get('target_tool_schema') or {}
        defs = ((sch.get('parameters') or {}).get('properties') or {})
        gold = ((qa.get('tool_call') or {}).get('arguments') or {})
        gi = ((qa.get('tool_call') or {}).get('grounding_info') or {})

        for p, g in gold.items():
            typ = str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):
                continue
            d = defs.get(p) or {}
            scope_text = h.target_scope(qa,p,d)
            role_text = h.target_role(qa,p,d)
            scope_v = enc.encode(scope_text)
            role_v = enc.encode(role_text)

            ffeats = []
            for ci, c in enumerate(clusters):
                ffeats.append({
                    'semantic': float(descE[ci] @ scope_v),
                    'role': float(roleE[ci] @ role_v),
                    'alias': float(bool(c['aliases'] & qaliases)),
                    'recency': c['recency'],
                    'episode_recency': c['episode_recency'],
                    'structured': c['structured_ratio'],
                    'salience': c['salience'],
                    'size_log': c['size_log'],
                })

            candidates = []
            positive_eps = set()
            max_turn = max([int(n.get('turn',-1)) for n in nodes] + [1])
            max_ep = max([int(n.get('episode',-1)) for n in nodes] + [1])
            for ni, node in enumerate(nodes):
                expanded = h.expanded(node,p,d)
                pos_here = any(base.norm(c['value']) == base.norm(g) for c in expanded)
                if pos_here:
                    positive_eps.add(int(node.get('episode',-1)))
                if not expanded:
                    continue
                node_scope = float(node_E[ni] @ scope_v)
                roles = [c['role'] for c in expanded]
                role_mat = enc.encode(roles)
                role_sims = role_mat @ role_v
                intent_sim = float(enc.encode(str(node.get('intent',''))) @ scope_v) if node.get('intent') else 0.0
                for ci, c in enumerate(expanded):
                    prop = c['prop']
                    field = str(prop.get('field',''))
                    lf = leaf(field)
                    candidates.append({
                        'node': ni,
                        'episode': int(node.get('episode',-1)),
                        'value': c['value'],
                        'label': int(base.norm(c['value']) == base.norm(g)),
                        'feature': {
                            'field_semantic': float(role_sims[ci]),
                            'node_scope': node_scope,
                            'intent_scope': intent_sim,
                            'param_field_exact': float(bool(lf) and canon(lf)==canon(p)),
                            'param_field_jacc': jacc(p,lf),
                            'desc_field_jacc': jacc(str(d.get('description','')), lf+' '+field),
                            'target_tool_source_tool_jacc': jacc(sch.get('name',''), node.get('tool','')),
                            'target_tool_source_tool_exact': float(bool(node.get('tool')) and canon(sch.get('name',''))==canon(node.get('tool',''))),
                            'query_intent_jacc': jacc(str(qa.get('query','')), str(node.get('intent',''))),
                            'recency': float(int(node.get('turn',-1))/max(1,max_turn)),
                            'episode_recency': float(int(node.get('episode',-1))/max(1,max_ep)),
                            'field_depth': float(field.count('.')+field.count('[')),
                            'kind:'+str(node.get('kind','unknown')): 1.0,
                            'src:'+str(prop.get('src','unknown')): 1.0,
                            'op:'+str(c.get('op','identity')): 1.0,
                            'type:'+str(d.get('type','')).lower(): 1.0,
                        }
                    })

            slots.append({
                'qi': qi, 'qa_id': qa.get('qa_id'), 'parameter': p, 'typ': typ,
                'gold': g, 'clusters': clusters, 'focus_feats': ffeats,
                'positive_episodes': positive_eps, 'candidates': candidates,
            })
        if qi % 10 == 0:
            print('FOCUS_FIELD_BUILD', qi, 'nodes', len(nodes), 'clusters', len(clusters), 'slots', len(slots), 'cache', len(enc.cache), flush=True)
    return slots, repair_report


def restricted_candidates(slot, w, k):
    if k >= 999:
        allowed = None
        focus_rank = {}
        focus_sc = {}
    else:
        order, scores = rank_focus(slot,w)
        chosen = [int(i) for i in order[:k]]
        allowed = {slot['clusters'][i]['episode'] for i in chosen}
        focus_rank = {slot['clusters'][i]['episode']: r for r,i in enumerate(chosen)}
        focus_sc = {slot['clusters'][i]['episode']: float(scores[i]) for i in chosen}
    rows = []
    for c in slot['candidates']:
        if allowed is not None and c['episode'] not in allowed:
            continue
        f = dict(c['feature'])
        if allowed is None:
            f['focus_rank'] = -1.0
            f['focus_score'] = 0.0
        else:
            f['focus_rank'] = float(focus_rank.get(c['episode'], k))
            f['focus_score'] = float(focus_sc.get(c['episode'], 0.0))
        rows.append((c,f))
    return rows


def fit_pairwise(slots, train_ids, w, k):
    rows=[]; ys=[]; groups=0
    for s in slots:
        if s['qi'] not in train_ids:
            continue
        rc = restricted_candidates(s,w,k)
        pos=[i for i,(c,_) in enumerate(rc) if c['label']]
        neg=[i for i,(c,_) in enumerate(rc) if not c['label']]
        if not pos or not neg:
            continue
        groups += 1
        # Hard negatives: strongest raw field-semantic candidates.
        neg = sorted(neg, key=lambda i: rc[i][1].get('field_semantic',0.0), reverse=True)[:24]
        for pi in pos[:6]:
            for ni in neg:
                a=rc[pi][1]; b=rc[ni][1]; keys=set(a)|set(b)
                rows.append({x:a.get(x,0.0)-b.get(x,0.0) for x in keys}); ys.append(1)
                rows.append({x:b.get(x,0.0)-a.get(x,0.0) for x in keys}); ys.append(0)
    if not rows:
        return None,None,0,groups
    vec=DictVectorizer(sparse=True); X=vec.fit_transform(rows)
    clf=LogisticRegression(max_iter=2000,C=C_FIXED,class_weight='balanced',random_state=SEED).fit(X,ys)
    return vec,clf,len(rows),groups


def candidate_scores(rc, vec, clf, mode='learned'):
    if not rc:
        return np.asarray([],dtype=np.float32)
    if mode=='raw':
        return np.asarray([f.get('field_semantic',0.0)+0.20*f.get('node_scope',0.0)+0.05*f.get('focus_score',0.0) for _,f in rc],dtype=np.float32)
    X=vec.transform([f for _,f in rc])
    return np.asarray(X @ clf.coef_[0]).reshape(-1)


def evaluate(slots, qset, w, k, vec=None, clf=None, mode='learned'):
    by=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0))
    examples=[]
    for s in slots:
        if s['qi'] not in qset:
            continue
        m=by[s['typ']]; m['n']+=1
        rc=restricted_candidates(s,w,k)
        pos={i for i,(c,_) in enumerate(rc) if c['label']}
        m['covered']+=int(bool(pos))
        if not rc:
            continue
        sc=candidate_scores(rc,vec,clf,mode); order=np.argsort(-sc)
        m['top1']+=int(bool(pos) and int(order[0]) in pos)
        m['top3']+=int(any(int(i) in pos for i in order[:3]))
        m['top5']+=int(any(int(i) in pos for i in order[:5]))
        if s['typ']=='explicit' and pos and int(order[0]) not in pos and len(examples)<6:
            examples.append({'qa_id':s['qa_id'],'parameter':s['parameter'],'gold':s['gold'],'top':[{'score':float(sc[int(i)]),'episode':rc[int(i)][0]['episode'],'value':rc[int(i)][0]['value']} for i in order[:3]]})
    out={}
    for typ,c in by.items():
        n=max(1,c['n']); cov=max(1,c['covered'])
        out[typ]={'n':c['n'],'candidate_coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov}
    return out,examples


def objective(r):
    return 0.7*r.get('explicit',{}).get('top1_all',0.0)+0.3*r.get('inferred',{}).get('top1_all',0.0)


def main():
    enc=wb.CachedEncoder(); slots,report=build(enc)
    idx=np.arange(N); folds=list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(idx))
    all_results={}
    for k in FOCUS_KS:
        key='all_world' if k>=999 else f'top{k}_focus'
        fold_rows=[]
        for fi,(tr,va) in enumerate(folds):
            trset=set(idx[tr].tolist()); vaset=set(idx[va].tolist())
            w=choose_focus_weights(slots,trset)
            vec,clf,nrows,ngroups=fit_pairwise(slots,trset,w,k)
            learned,ex=evaluate(slots,vaset,w,k,vec,clf,'learned')
            raw,_=evaluate(slots,vaset,w,k,None,None,'raw')
            fold_rows.append({'fold':fi,'focus_weights':w,'pair_rows':nrows,'train_groups':ngroups,'learned':learned,'raw':raw,'examples':ex})
            print('FOCUS_FIELD_FOLD',key,fi,'obj',objective(learned),json.dumps(learned),flush=True)
        summary={}
        for mode in ('learned','raw'):
            summary[mode]={}
            for typ in ('explicit','inferred'):
                summary[mode][typ]={}
                for metric in ('candidate_coverage','top1_all','top3_all','top5_all'):
                    vals=[f[mode].get(typ,{}).get(metric,0.0) for f in fold_rows]
                    summary[mode][typ][metric]=float(np.mean(vals))
        all_results[key]={'summary':summary,'folds':fold_rows}
    result={
        'stage':'SWM-C focus-conditioned local field binding',
        'split':'QA001-100 development-only 5-fold QA CV; QA101-400 gold remains sealed',
        'architecture':'query-independent semantic world ingestion -> query-time discourse focus ranking -> restrict candidates to top-k focus episodes -> value-masked local field-role pairwise binding -> exact payload dereference after selection',
        'results':all_results,
        'comparison_targets':{'world_ingest_explicit':0.9340659340659341,'world_ingest_inferred':0.425,'discourse_top3_explicit':0.8892,'discourse_top3_inferred':0.4162,'gold_node_field_top1_explicit':0.4353,'gold_node_field_top1_inferred':0.4118},
        'repair_report':report,
        'guardrail':'Gold values only construct training labels and score held-out QA001-100 folds. Candidate payload values are never model features. Focus weights are selected on each training fold only. No QA101-400 gold is read.'
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('SEMANTIC_FOCUS_FIELD_BINDING='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
