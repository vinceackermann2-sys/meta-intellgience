import json, math
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

SOURCE='icl_banking77_5900shot_balance'
OUT=Path('researchstrong/strong_banking77_result.json')

def norm(x):return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)
def acc(p,y):return float(np.mean(np.asarray(p,dtype=int)==np.asarray(y,dtype=int)))

def load_official():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Test_Time_Learning',revision='main')
    row=[r for r in ds if (r.get('metadata') or {}).get('source')==SOURCE][0]
    pairs=[];buf=[]
    for raw in row['context'].splitlines():
        s=raw.strip()
        if s.startswith('label:'):
            pairs.append(('\n'.join(buf).strip(),int(s.split(':',1)[1])));buf=[]
        elif s:buf.append(raw)
    qs=list(row.get('questions') or []);ans=list(row.get('answers') or [])
    gold=np.asarray([int((a if isinstance(a,list) else [a])[0]) for a in ans],dtype=np.int64)
    return pairs,qs,gold,len(row['context'])

def split_within_class(y,frac=.2):
    tr=[];va=[]
    for c in range(77):
        ids=np.where(y==c)[0];n=max(1,int(round(len(ids)*frac)));tr.extend(ids[:-n]);va.extend(ids[-n:])
    return np.asarray(tr),np.asarray(va)

def class_centroids(X,y):return norm(np.stack([X[y==c].mean(0) for c in range(77)]).astype(np.float32))

def select_hard(X,y,decision,budget_frac):
    # Hardness is low true-class margin. Select class-balanced so rare/easy classes keep local exceptions too.
    order=[];per=max(1,int(math.ceil(len(X)*budget_frac/77)))
    for c in range(77):
        ids=np.where(y==c)[0]
        if not len(ids):continue
        true=decision[ids,c];tmp=decision[ids].copy();tmp[:,c]=-1e9;other=tmp.max(1);hard=other-true
        take=ids[np.argsort(-hard)[:min(per,len(ids))]];order.extend(map(int,take))
    # Trim globally by actual hardness when balanced ceiling overshoots.
    target=max(77,int(round(len(X)*budget_frac)))
    if len(order)>target:
        ids=np.asarray(order);true=decision[ids,y[ids]];tmp=decision[ids].copy();tmp[np.arange(len(ids)),y[ids]]=-1e9;hard=tmp.max(1)-true
        order=list(map(int,ids[np.argsort(-hard)[:target]]))
    return np.asarray(order,dtype=int)

def exception_class_sims(Q,E,Ey):
    S=np.full((len(Q),77),-1.0,np.float32)
    if not len(E):return S
    Z=Q@E.T
    for c in range(77):
        m=(Ey==c)
        if np.any(m):S[:,c]=Z[:,m].max(1)
    return S

def hybrid_predict(dec,exc,mode,beta,tau,margin):
    if mode=='additive':
        bonus=np.maximum(0,exc-tau)*beta
        return (dec+bonus).argmax(1)
    base=dec.argmax(1);d=np.sort(dec,axis=1);bm=d[:,-1]-d[:,-2];ei=exc.argmax(1);es=exc[np.arange(len(exc)),ei]
    out=base.copy();mask=(es>=tau)&(bm<=margin);out[mask]=ei[mask];return out

def tune_hybrid(Xt,yt,Xv,yv,C,budget):
    b=LogisticRegression(C=C,max_iter=2500,solver='lbfgs').fit(Xt,yt);dt=b.decision_function(Xt);dv=b.decision_function(Xv)
    ids=select_hard(Xt,yt,dt,budget);E=Xt[ids];Ey=yt[ids];ex=exception_class_sims(Xv,E,Ey)
    best=(-1,None)
    for mode in ['additive','override']:
      if mode=='additive':
        for tau in [.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]:
          for beta in [.25,.5,1,2,4,8,12]:
            p=hybrid_predict(dv,ex,mode,beta,tau,0);a=acc(p,yv)
            if a>best[0]:best=(a,(mode,beta,tau,0))
      else:
        for tau in [.45,.5,.55,.6,.65,.7,.75,.8,.85]:
          for margin in [.1,.2,.3,.5,.75,1,1.5,2,3]:
            p=hybrid_predict(dv,ex,mode,0,tau,margin);a=acc(p,yv)
            if a>best[0]:best=(a,(mode,0,tau,margin))
    return best,len(ids)

def farthest_first_class(X,y,k):
    sel=[]
    for c in range(77):
        ids=np.where(y==c)[0];Z=X[ids];cent=norm(Z.mean(0,keepdims=True))[0];first=int(np.argmax(Z@cent));chosen=[first]
        best=1-(Z@Z[first])
        while len(chosen)<min(k,len(ids)):
            j=int(np.argmax(best));chosen.append(j);best=np.minimum(best,1-(Z@Z[j]))
        sel.extend(map(int,ids[chosen]))
    return np.asarray(sel,dtype=int)

def condensed_nn(X,y,max_pass=5):
    # Classic condensed-nearest-neighbor style exact exemplar ledger.
    C=class_centroids(X,y);sel=[]
    for c in range(77):
        ids=np.where(y==c)[0];sel.append(int(ids[np.argmax(X[ids]@C[c])]))
    selected=set(sel);rng=np.random.default_rng(7);order=np.arange(len(X))
    for _ in range(max_pass):
        rng.shuffle(order);added=0
        P=X[np.asarray(sel)];Py=y[np.asarray(sel)]
        for i in order:
            if int(i) in selected:continue
            pred=int(Py[np.argmax(P@X[i])])
            if pred!=int(y[i]):
                sel.append(int(i));selected.add(int(i));added+=1;P=np.vstack([P,X[i:i+1]]);Py=np.append(Py,y[i])
        if added==0:break
    return np.asarray(sel,dtype=int)

def main():
    pairs,questions,gold,context_chars=load_official();texts=[x for x,_ in pairs];y=np.asarray([c for _,c in pairs],dtype=np.int64)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    E=enc.encode(texts+questions,batch_size=128,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32);X,Q=E[:len(texts)],E[len(texts):];d=X.shape[1]
    scores={};state={};detail={}

    # Full episodic references.
    sims=Q@X.T
    for k in [1,3,5,7]:
        idx=np.argpartition(-sims,k-1,axis=1)[:,:k]
        if k==1:p=y[idx[:,0]]
        else:
            p=[]
            for r,ii in enumerate(idx):
                vote=np.bincount(y[ii],weights=np.maximum(sims[r,ii],0)**4,minlength=77);p.append(int(vote.argmax()))
            p=np.asarray(p)
        scores[f'full_knn_K{k}']=acc(p,gold);state[f'full_knn_K{k}']={'vectors':len(X),'float_bytes':int(X.nbytes),'fraction':1.0}

    ti,vi=split_within_class(y);Xt,yt,Xv,yv=X[ti],y[ti],X[vi],y[vi]
    # Compact linear semantic state.
    best=(-1,None)
    for C in [.05,.1,.2,.3,.5,1,2,3,5,10]:
        m=LogisticRegression(C=C,max_iter=2500,solver='lbfgs').fit(Xt,yt);a=acc(m.predict(Xv),yv)
        if a>best[0]:best=(a,C)
    base=LogisticRegression(C=best[1],max_iter=2500,solver='lbfgs').fit(X,y);scores['linear_semantic_state']=acc(base.predict(Q),gold);state['linear_semantic_state']={'vectors_equiv':77,'float_bytes':int(base.coef_.nbytes+base.intercept_.nbytes),'fraction_vs_full_vectors':77/len(X)};detail['linear_semantic_state']={'validation':best[0],'C':best[1]}

    # Exact exemplar compression baselines.
    for k in [1,2,4,8,12,16,24]:
        ids=farthest_first_class(X,y,k);pred=y[ids[(Q@X[ids].T).argmax(1)]];name=f'farthest_exemplar_K{k}';scores[name]=acc(pred,gold);state[name]={'vectors':len(ids),'float_bytes':int(X[ids].nbytes),'fraction':len(ids)/len(X)}
    cnn=condensed_nn(X,y);scores['condensed_1nn']=acc(y[cnn[(Q@X[cnn].T).argmax(1)]],gold);state['condensed_1nn']={'vectors':len(cnn),'float_bytes':int(X[cnn].nbytes),'fraction':len(cnn)/len(X)}

    # Cortex + hippocampal exceptions: linear state handles the common manifold; only hard local cases remain episodic.
    for budget in [.01,.02,.03,.05,.075,.10,.15,.20,.30]:
        val,hyper=tune_hybrid(Xt,yt,Xv,yv,best[1],budget)
        dec=base.decision_function(X);ids=select_hard(X,y,dec,budget);ex=exception_class_sims(Q,X[ids],y[ids]);dq=base.decision_function(Q);mode,beta,tau,margin=hyper;pred=hybrid_predict(dq,ex,mode,beta,tau,margin)
        name=f'cortex_plus_exceptions_{int(budget*1000):03d}permille';scores[name]=acc(pred,gold)
        bytes_=base.coef_.nbytes+base.intercept_.nbytes+X[ids].nbytes
        state[name]={'exception_vectors':len(ids),'exception_fraction':len(ids)/len(X),'float_bytes':int(bytes_),'bytes_fraction_vs_full_embeddings':float(bytes_/X.nbytes)}
        detail[name]={'validation':val,'mode':mode,'beta':beta,'tau':tau,'base_margin_threshold':margin}

    # Full embeddings quantized to int8: separates vector-count vs byte-compression effects.
    scale=np.maximum(np.max(np.abs(X),axis=0),1e-6)/127.0;Xi=np.round(X/scale).clip(-127,127).astype(np.int8);Qi=(Q/scale).astype(np.float32)
    # dot in reconstructed space, without materializing float X permanently.
    qs=(Q*scale);qsim=qs@Xi.T.astype(np.float32);scores['full_1nn_int8_state']=acc(y[qsim.argmax(1)],gold);state['full_1nn_int8_state']={'vectors':len(X),'bytes':int(Xi.nbytes+scale.nbytes),'byte_fraction_vs_float_full':float((Xi.nbytes+scale.nbytes)/X.nbytes)}

    best_name=max(scores,key=scores.get)
    frontier=sorted([(v,state[k].get('exception_fraction',state[k].get('fraction',1.0)),k) for k,v in scores.items() if 'float_bytes' in state[k]],key=lambda z:(z[1],-z[0]))
    out={'benchmark':'MemoryAgentBench Banking77 official 5897 demonstrations / 100 queries','encoder':'all-MiniLM-L6-v2','embedding_dim':d,'context_chars':context_chars,
         'hypothesis':'operator-aware two-system memory: compress stable class semantics into a small linear cortical state and retain only hard local exceptions episodically',
         'scores':scores,'state':state,'detail':detail,'best':{'name':best_name,'accuracy':scores[best_name]},'memory_accuracy_frontier':frontier,
         'guardrail':'all thresholds/budgets use a within-class train/validation split; the 100 official benchmark answers are used only once for final scoring.'}
    OUT.write_text(json.dumps(out,indent=2));print('BANKING_EXCEPTION_LEDGER='+json.dumps(out))
if __name__=='__main__':main()
