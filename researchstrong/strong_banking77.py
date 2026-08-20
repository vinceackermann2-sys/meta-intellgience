import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SOURCE = 'icl_banking77_5900shot_balance'
OUT = Path('researchstrong/strong_banking77_result.json')

def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def acc(pred, gold):
    return float(np.mean(np.asarray(pred, dtype=int) == np.asarray(gold, dtype=int)))

def load_official():
    ds = load_dataset('ai-hyz/MemoryAgentBench', split='Test_Time_Learning', revision='main')
    rows = [r for r in ds if (r.get('metadata') or {}).get('source') == SOURCE]
    if len(rows) != 1:
        raise RuntimeError(f'expected one Banking77 row, got {len(rows)}')
    row = rows[0]
    context = row['context']

    # Consume the released row exactly. The source name says "5900shot", but the
    # released context currently contains 5,897 actual label records. The row is
    # the benchmark input, so its record count—not the human-readable source name—is authoritative.
    label_records = sum(1 for raw in context.splitlines() if raw.strip().startswith('label:'))
    pairs, buf = [], []
    for raw in context.splitlines():
        s = raw.strip()
        if s.startswith('label:'):
            text = '\n'.join(buf).strip()
            if not text:
                raise RuntimeError('label without preceding utterance')
            pairs.append((text, int(s.split(':', 1)[1].strip())))
            buf = []
        elif s:
            buf.append(raw)
    if buf:
        raise RuntimeError('unterminated final demonstration')

    questions = list(row.get('questions') or [])
    answers = list(row.get('answers') or [])
    gold = np.asarray([int((a if isinstance(a, list) else [a])[0]) for a in answers], dtype=np.int64)
    if len(pairs) != label_records or not (5800 <= label_records <= 6000):
        raise RuntimeError(f'parsed={len(pairs)} label_records={label_records}')
    if len(questions) != 100 or len(gold) != 100:
        raise RuntimeError((len(questions), len(gold)))
    if sorted(set(y for _, y in pairs)) != list(range(77)):
        raise RuntimeError('unexpected label set')
    return pairs, questions, gold, len(context), label_records

class OnlineProto:
    def __init__(self, classes, k):
        self.k = k
        self.p = [[] for _ in range(classes)]
        self.n = [[] for _ in range(classes)]
    def write(self, x, y):
        if len(self.p[y]) < self.k:
            self.p[y].append(x.copy()); self.n[y].append(1); return
        P = norm(np.stack(self.p[y]))
        j = int((P @ x).argmax()); n = self.n[y][j] + 1
        self.p[y][j] += (x - self.p[y][j]) / n
        self.n[y][j] = n
    def predict(self, q):
        s = np.full((len(q), len(self.p)), -1e9, np.float32)
        for c, p in enumerate(self.p):
            if p: s[:, c] = (q @ norm(np.stack(p)).T).max(1)
        return s.argmax(1)
    def vectors(self):
        return sum(map(len, self.p))

class HardExceptionMemory:
    """One semantic centroid per class plus a tiny cache of hard boundary cases."""
    def __init__(self, X, y, classes, k):
        self.c = np.stack([norm(X[y == c].mean(0, keepdims=True))[0] for c in range(classes)])
        base = X @ self.c.T
        pred = base.argmax(1)
        E, Ey = [], []
        for c in range(classes):
            ids = np.where(y == c)[0]
            wrong = ids[pred[ids] != c]
            chosen = []
            if len(wrong):
                margin = base[wrong].max(1) - base[wrong, c]
                chosen = list(map(int, wrong[np.argsort(-margin)[:k]]))
            seen = set(chosen)
            if len(chosen) < k:
                far = [int(i) for i in ids[np.argsort(X[ids] @ self.c[c])] if int(i) not in seen]
                chosen += far[:k-len(chosen)]
            for i in chosen:
                E.append(X[i].copy()); Ey.append(c)
        self.E = np.stack(E) if E else np.zeros((0, X.shape[1]), np.float32)
        self.Ey = np.asarray(Ey, dtype=np.int64)
    def predict(self, q, gamma):
        s = q @ self.c.T
        if len(self.E):
            es = q @ self.E.T
            for c in range(len(self.c)):
                z = es[:, self.Ey == c]
                if z.size: s[:, c] = np.maximum(s[:, c], gamma * z.max(1))
        return s.argmax(1)
    def vectors(self):
        return len(self.c) + len(self.E)

def split_within_class(y):
    train, val = [], []
    for c in range(77):
        ids = np.where(y == c)[0]
        n = max(1, len(ids)//5)
        train.extend(ids[:-n]); val.extend(ids[-n:])
    return np.asarray(train), np.asarray(val)

def main():
    pairs, questions, gold, context_chars, label_records = load_official()
    texts = [x for x, _ in pairs]
    y = np.asarray([c for _, c in pairs], dtype=np.int64)

    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    E = enc.encode(texts + questions, batch_size=128, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    X, Q = E[:len(texts)], E[len(texts):]
    d, classes = X.shape[1], 77
    scores, memory, detail = {}, {}, {}

    sims = Q @ X.T
    scores['full_1nn'] = acc(y[sims.argmax(1)], gold)
    memory['full_1nn'] = int(X.nbytes)

    sums = np.zeros((classes, d), np.float64); counts = np.zeros(classes, np.int64)
    for x, c in zip(X, y): sums[c] += x; counts[c] += 1
    C = norm((sums / counts[:, None]).astype(np.float32))
    scores['stream_centroid'] = acc((Q @ C.T).argmax(1), gold)
    memory['stream_centroid'] = int(C.nbytes + counts.nbytes)

    for k in [2, 4, 8]:
        m = OnlineProto(classes, k)
        for x, c in zip(X, y): m.write(x, int(c))
        name = f'online_proto_K{k}'
        scores[name] = acc(m.predict(Q), gold)
        memory[name] = int(m.vectors() * d * 4)

    ti, vi = split_within_class(y)
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]

    configs = [
        ('logreg', [.1, .3, 1, 3, 10], lambda h: LogisticRegression(C=h, max_iter=2000, solver='lbfgs')),
        ('linear_svm', [.01, .03, .1, .3, 1], lambda h: LinearSVC(C=h, max_iter=7000)),
        ('ridge', [.1, .3, 1, 3, 10], lambda h: RidgeClassifier(alpha=h)),
    ]
    for name, grid, build in configs:
        best = (-1., None)
        for h in grid:
            m = build(h); m.fit(Xt, yt); a = acc(m.predict(Xv), yv)
            if a > best[0]: best = (a, h)
        m = build(best[1]); m.fit(X, y)
        scores[name] = acc(m.predict(Q), gold)
        coef = getattr(m, 'coef_', None)
        memory[name] = int(coef.nbytes) if coef is not None else None
        detail[name] = {'validation': best[0], 'hyper': best[1]}

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
    scores['shrinkage_lda'] = acc(lda.predict(Q), gold)
    memory['shrinkage_lda'] = None

    cut = int(len(X) * .8); all_classes = np.arange(classes); best = (-1., None, None)
    for alpha in [1e-6, 1e-5, 1e-4]:
        for eta in [.001, .003, .01, .03]:
            m = SGDClassifier(loss='log_loss', alpha=alpha, learning_rate='constant', eta0=eta, random_state=0)
            for i in range(cut):
                m.partial_fit(X[i:i+1], y[i:i+1], classes=all_classes if i == 0 else None)
            a = acc(m.predict(X[cut:]), y[cut:])
            if a > best[0]: best = (a, alpha, eta)
    m = SGDClassifier(loss='log_loss', alpha=best[1], learning_rate='constant', eta0=best[2], random_state=0)
    for i in range(len(X)):
        m.partial_fit(X[i:i+1], y[i:i+1], classes=all_classes if i == 0 else None)
    scores['online_sgd_onepass'] = acc(m.predict(Q), gold)
    memory['online_sgd_onepass'] = int(m.coef_.nbytes)
    detail['online_sgd_onepass'] = {'validation': best[0], 'alpha': best[1], 'eta': best[2]}

    A = Xt.T @ Xt; B = np.eye(classes, dtype=np.float32)[yt].T @ Xt; best = (-1., None)
    for lam in [.1, .3, 1, 3, 10, 30]:
        W = np.linalg.solve(A + lam*np.eye(d, dtype=np.float32), B.T).T
        a = acc((Xv @ W.T).argmax(1), yv)
        if a > best[0]: best = (a, lam)
    Af = X.T @ X; Bf = np.eye(classes, dtype=np.float32)[y].T @ X
    W = np.linalg.solve(Af + best[1]*np.eye(d, dtype=np.float32), Bf.T).T.astype(np.float32)
    scores['reversible_ridge_state'] = acc((Q @ W.T).argmax(1), gold)
    memory['reversible_ridge_state'] = int(Af.nbytes + Bf.nbytes)
    detail['reversible_ridge_state'] = {'validation': best[0], 'lambda': best[1]}

    for k in [1, 2, 4, 8]:
        hm_val = HardExceptionMemory(Xt, yt, classes, k); bestg = (-1., None)
        for gamma in [.8, .9, 1., 1.05, 1.1, 1.2]:
            a = acc(hm_val.predict(Xv, gamma), yv)
            if a > bestg[0]: bestg = (a, gamma)
        hm = HardExceptionMemory(X, y, classes, k)
        name = f'hard_exception_K{k}'
        scores[name] = acc(hm.predict(Q, bestg[1]), gold)
        memory[name] = int(hm.vectors() * d * 4)
        detail[name] = {'validation': bestg[0], 'gamma': bestg[1], 'vectors': hm.vectors()}

    out = {
        'benchmark': 'MemoryAgentBench official Test_Time_Learning / icl_banking77_5900shot_balance',
        'source': SOURCE, 'released_label_records': label_records,
        'context_chars': context_chars, 'demonstrations_consumed': len(X), 'questions': len(Q),
        'encoder': 'sentence-transformers/all-MiniLM-L6-v2', 'embedding_dim': d,
        'scores': scores, 'memory_bytes_float_state': memory, 'detail': detail,
        'best': max(scores.items(), key=lambda z: z[1]),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print('MAB_OFFICIAL_RESULT=' + json.dumps(out))

if __name__ == '__main__':
    main()
