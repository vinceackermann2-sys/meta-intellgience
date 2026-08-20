"""Performance-equivalent wrapper for gatea_source_selector.
Caches episode text embeddings by source-id/text fingerprint. No scoring/routing logic changes.
"""
import hashlib
import numpy as np
import episode_scoped_router as es
import gatea_source_selector as run

_ORIG=es.retrieve_episodes
_CACHE={}

def cached_retrieve(enc,eps,query_text,k=es.TOP_EPISODES):
    if not eps:return []
    fp=hashlib.sha1(('\n'.join(str(e.get('source_id',''))+'|'+e.get('text','')[:6000] for e in eps)).encode('utf-8')).hexdigest()
    if fp not in _CACHE:
        texts=[e['text'][:6000] for e in eps]
        _CACHE[fp]=enc.encode(texts,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    E=_CACHE[fp]
    q=enc.encode([query_text],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
    order=np.argsort(-(E@q))[:min(k,len(eps))]
    return [(rank,eps[int(i)],float(E[int(i)]@q)) for rank,i in enumerate(order)]

es.retrieve_episodes=cached_retrieve
run.OUT=run.Path(__file__).with_name('gatea_source_selector_cached_result.json')
run.main()
