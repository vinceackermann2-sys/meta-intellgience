from pathlib import Path
import json, math, re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import semantic_role_bridge_cv as rb

OUT=Path(__file__).with_name('semantic_role_bridge_weighted_cv_result.json')
ALPHAS=[1e-2,1e-1,1.0,10.0,100.0]

def info_weight(v):
    if isinstance(v,bool): return 0.08
    if isinstance(v,(int,float)):
        try:
            x=float(v)
            if x in (0.0,1.0):return 0.08
            if abs(x)<10:return 0.25
            return 0.65
        except: return 0.3
    s=str(v).strip()
    if not s:return 0.05
    if s.casefold() in {'true','false','yes','no','none','null','unknown','0','1'}:return 0.08
    if re.fullmatch(r'[A-Za-z]{2,3}',s):return 0.35
    if re.fullmatch(r'https?://\S+',s):return 1.4
    if re.fullmatch(r'0x[a-fA-F0-9]{8,}',s):return 1.4
    if re.fullmatch(r'[A-Za-z0-9_.-]{8,}',s):return 1.1
    return min(1.25,0.45+0.08*min(len(s),10))

def fit_map(slots,train_ids,alpha):
    xs=[];ys=[];ws=[];raw=0
    for s in slots:
        if s['qi'] not in train_ids:continue
        # Frequency of each executor payload inside this slot. Repeated low-information values are weak semantic supervision.
        freq={}
        for c in s['cands']:
            k=base.norm(c['value']);freq[k]=freq.get(k,0)+1
        pos=np.where(s['labels']>0)[0]
        for i in pos[:8]:
            c=s['cands'][int(i)];k=base.norm(c['value']);w=info_weight(c['value'])/math.sqrt(max(1,freq.get(k,1)))
            # Structured field correspondences are stronger than free-text entity extraction.
            if c.get('src')=='structured':w*=1.25
            if c.get('op')!='identity':w*=0.9
            xs.append(s['source_vecs'][int(i)]);ys.append(s['target_vec']);ws.append(max(0.02,w));raw+=1
    if not xs:return None,0,0.0
    X=np.asarray(xs,np.float32);Y=np.asarray(ys,np.float32);W=np.asarray(ws,np.float32)
    model=Ridge(alpha=alpha,fit_intercept=True).fit(X,Y,sample_weight=W)
    return model,raw,float(W.sum())

def main():
    enc=rb.CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'))
    slots,missing=rb.build(enc);idx=np.arange(rb.N);folds=list(KFold(n_splits=5,shuffle=True,random_state=20260820).split(idx));grid=[]
    for alpha in ALPHAS:
        for blend in (0.25,0.5,0.75,1.0):
            fr=[];pairs=[];weights=[]
            for tr,va in folds:
                model,n,w=fit_map(slots,set(idx[tr].tolist()),alpha);r,_=rb.score(slots,set(idx[va].tolist()),model,blend);fr.append(r);pairs.append(n);weights.append(w)
            obj=float(np.mean([rb.objective(r) for r in fr]));grid.append((obj,alpha,blend,fr,pairs,weights));print('WEIGHTED_ROLE_GRID',alpha,blend,obj,flush=True)
    obj,alpha,blend,fr,pairs,weights=max(grid,key=lambda z:z[0]);model,n,w=fit_map(slots,set(range(rb.N)),alpha);full,examples=rb.score(slots,set(range(rb.N)),model,blend)
    result={'stage':'Semantic World Model v0.1: informativeness-weighted cross-schema role alignment','split':'QA001-100 development-only 5-fold CV; QA101-400 gold remains sealed','change_from_v0':'Same candidates and value-masked semantic roles. Positive role correspondences are weighted by executor-payload informativeness, inverse within-slot value frequency, structured provenance, and transform penalty. Candidate values remain labels/payloads only, never semantic features.','selected':{'alpha':alpha,'blend':blend,'mean_cv_explicit_inferred_top1':obj,'folds':fr,'positive_pairs_per_fold':pairs,'effective_weight_per_fold':weights},'all_dev_refit':full,'positive_pairs_all':n,'effective_weight_all':w,'examples':examples,'missing_zero_based':missing,'embedding_cache_entries':len(enc.cache),'guardrail':'No QA101-400 gold is read. Weighting is generic and fixed before evaluation; it does not use grounding_info.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ROLE_BRIDGE_WEIGHTED='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
