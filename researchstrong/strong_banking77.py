import json, re, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SOURCE='icl_banking77_5900shot_balance'
HF_ROWS='https://datasets-server.huggingface.co/first-rows?dataset='+urllib.parse.quote('ai-hyz/MemoryAgentBench',safe='')+'&config=default&split=Test_Time_Learning'
OUT=Path('researchstrong/strong_banking77_result.json')

def norm(x):
    return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)

def acc(p,y):
    return float(np.mean(np.asarray(p,dtype=int)==np.asarray(y,dtype=int)))

def as_obj(v):
    if isinstance(v,dict): return v
    if isinstance(v,str):
        try: return json.loads(v)
        except Exception: return {'_raw':v}
    return {}

def as_list(v):
    if isinstance(v,list): return v
    if isinstance(v,str):
        try:
            z=json.loads(v)
            return z if isinstance(z,list) else [z]
        except Exception: return [v]
    return [] if v is None else [v]

def load_official():
    req=urllib.request.Request(HF_ROWS,headers={'User-Agent':'memory-gate/1.0'})
    with urllib.request.urlopen(req,timeout=120) as r:
        data=json.loads(r.read().decode('utf-8'))
    row=None
    for wrap in data.get('rows',[]):
        z=wrap.get('row',{}); meta=as_obj(z.get('metadata'))
        if meta.get('source')==SOURCE:
            row=z; break
    if row is None: raise RuntimeError('official Banking77 TTL row not found')
    context=row['context']
    # Each official demonstration is one utterance followed by `label: N`, separated by blank lines.
    pairs=[]
    for m in re.finditer(r'(?:\A|\n\n)(.*?)\nlabel:\s*(\d+)\s*(?=\n\n|\Z)',context,re.S):
        text=m.group(1).strip(); label=int(m.group(2))
        if text: pairs.append((text,label))
    questions=as_list(row.get('questions')); answers=as_list(row.get('answers'))
    y=[]
    for a in answers:
        aa=as_list(a)
        if not aa: raise RuntimeError('empty answer')
        y.append(int(aa[0]))
    if not (5800 <= len(pairs) <= 6000): raise RuntimeError(f'parsed {len(pairs)} demonstrations, expected ~5900')
    if len(questions)!=100 or len(y)!=100: raise RuntimeError((len(questions),len(y)))
    labels=sorted(set(v for _,v in pairs))
    if labels!=list(range(77)): raise RuntimeError(f'labels={labels}')
    return pairs,questions,np.asarray(y,dtype=np.int64),len(context)

class OnlineProto:
    def __init__(self,nc,d,K):
        self.K=K; self.P=[[] for _ in range(nc)]; self.N=[[] for _ in range(nc)]; self.d=d
    def write(self,x,y):
        p=self.P[y]
        if len(p)<self.K:
            p.append(x.copy()); self.N[y].append(1); return
        P=norm(np.stack(p)); j=int((P@x).argmax()); n=self.N[y][j]+1
        p[j]=p[j]+(x-p[j])/n; self.N[y][j]=n
    def pred(self,Q):
        S=np.full((len(Q),len(self.P)),-1e9,np.float32)
        for c,p in enumerate(self.P):
            if p: S[:,c]=(Q@norm(np.stack(p)).T).max(1)
        return S.argmax(1)
    def vectors(self): return sum(len(x) for x in self.P)

class HardExceptionMemory:
    """One semantic state per class plus a tiny boundary-exception cache.
    Exceptions are selected only from examples the semantic state confuses.
    """
    def __init__(self,X,y,nc,K):
        self.C=np.stack([norm(X[y==c].mean(0,keepdims=True))[0] for c in range(nc)])
        base=X@self.C.T; pred=base.argmax(1)
        self.E=[]; self.Ey=[]
        for c in range(nc):
            ids=np.where(y==c)[0]
            wrong=ids[pred[ids]!=c]
            # prioritize most confidently wrong points; fill with farthest-from-centroid points
            if len(wrong):
                margin=base[wrong].max(1)-base[wrong,c]
                chosen=list(wrong[np.argsort(-margin)[:K]])
            else: chosen=[]
            if len(chosen)<K:
                left=[i for i in ids[np.argsort(X[ids]@self.C[c])] if int(i) not in set(chosen)]
                chosen+=list(left[:K-len(chosen)])
            for i in chosen:
                self.E.append(X[int(i)].copy()); self.Ey.append(c)
        self.E=np.stack(self.E) if self.E else np.zeros((0,X.shape[1]),np.float32)
        self.Ey=np.asarray(self.Ey,dtype=np.int64)
    def pred(self,Q,gamma):
        S=Q@self.C.T
        if len(self.E):
            ES=Q@self.E.T
            for c in range(len(self.C)):
                z=ES[:,self.Ey==c]
                if z.size: S[:,c]=np.maximum(S[:,c],gamma*z.max(1))
        return S.argmax(1)
    def vectors(self): return len(self.C)+len(self.E)

def tune_split(y):
    # deterministic within-class 80/20 split, preserving the official shuffled stream as much as possible
    tr=[]; va=[]
    for c in range(77):
        ids=np.where(y==c)[0]
        n=max(1,len(ids)//5)
        va.extend(ids[-n:]); tr.extend(ids[:-n])
    return np.asarray(tr,dtype=int),np.asarray(va,dtype=int)

def main():
    pairs,questions,qy,context_chars=load_official()
    texts=[t for t,_ in pairs]; y=np.asarray([v for _,v in pairs],dtype=np.int64)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    E=enc.encode(texts+questions,batch_size=128,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
    X,Q=E[:len(texts)],E[len(texts):]; d=X.shape[1]; nc=77
    scores={}; memory={}; detail={}

    # Full episodic retrieval floors.
    S=Q@X.T; scores['full_1nn']=acc(y[S.argmax(1)],qy); memory['full_1nn']=int(X.nbytes)
    k=5; ix=np.argpartition(S,-k,axis=1)[:,-k:]; p=[]
    for r,ids in enumerate(ix):
        vote={}
        for j in ids: vote[int(y[j])]=vote.get(int(y[j]),0.0)+float(max(0.0,S[r,j]))
        p.append(max(vote,key=vote.get))
    scores['full_5nn']=acc(p,qy); memory['full_5nn']=int(X.nbytes)

    # Exact streaming sufficient statistic: one sum + count per class.
    sums=np.zeros((nc,d),np.float64); counts=np.zeros(nc,np.int64)
    for x,c in zip(X,y): sums[c]+=x; counts[c]+=1
    C=norm((sums/counts[:,None]).astype(np.float32)); scores['stream_centroid']=acc((Q@C.T).argmax(1),qy)
    memory['stream_centroid']=int(C.nbytes+counts.nbytes)

    # Online multi-prototype states: bounded constant memory per class.
    for K in [2,4,8]:
        m=OnlineProto(nc,d,K)
        for x,c in zip(X,y): m.write(x,int(c))
        name=f'online_proto_K{K}'; scores[name]=acc(m.pred(Q),qy); memory[name]=int(m.vectors()*d*4)

    # Conventional supervised floor; tune on labels inside the demonstration stream, then refit on all 5900.
    ti,vi=tune_split(y); Xt,yt,Xv,yv=X[ti],y[ti],X[vi],y[vi]
    for name,grid,builder in [
        ('logreg',[.03,.1,.3,1,3,10,30],lambda h:LogisticRegression(C=h,max_iter=3000,solver='lbfgs')),
        ('linear_svm',[.003,.01,.03,.1,.3,1,3],lambda h:LinearSVC(C=h,max_iter=10000)),
        ('ridge',[.03,.1,.3,1,3,10,30],lambda h:RidgeClassifier(alpha=h)),
    ]:
        best=(-1,None)
        for h in grid:
            m=builder(h); m.fit(Xt,yt); a=acc(m.predict(Xv),yv)
            if a>best[0]: best=(a,h)
        m=builder(best[1]); m.fit(X,y); scores[name]=acc(m.predict(Q),qy); detail[name]={'validation':best[0],'hyper':best[1]}
        coef=getattr(m,'coef_',None); memory[name]=int(coef.nbytes) if coef is not None else None

    lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'); lda.fit(X,y)
    scores['shrinkage_lda']=acc(lda.predict(Q),qy); memory['shrinkage_lda']=None

    # True one-pass online learner, tuned on a prefix then replayed once over the full official sequence.
    best=(-1,None,None)
    cut=int(len(X)*0.8); classes=np.arange(nc)
    for alpha in [1e-6,1e-5,1e-4,1e-3]:
        for eta in [.001,.003,.01,.03,.1]:
            m=SGDClassifier(loss='log_loss',alpha=alpha,learning_rate='constant',eta0=eta,random_state=0)
            for i in range(cut): m.partial_fit(X[i:i+1],y[i:i+1],classes=classes if i==0 else None)
            a=acc(m.predict(X[cut:]),y[cut:])
            if a>best[0]: best=(a,alpha,eta)
    m=SGDClassifier(loss='log_loss',alpha=best[1],learning_rate='constant',eta0=best[2],random_state=0)
    for i in range(len(X)): m.partial_fit(X[i:i+1],y[i:i+1],classes=classes if i==0 else None)
    scores['online_sgd_onepass']=acc(m.predict(Q),qy); memory['online_sgd_onepass']=int(m.coef_.nbytes); detail['online_sgd_onepass']={'validation':best[0],'alpha':best[1],'eta':best[2]}

    # Reversible ridge fast state from additive sufficient statistics: X^T X and Y^T X.
    A=Xt.T@Xt; B=np.eye(nc,dtype=np.float32)[yt].T@Xt
    best=(-1,None)
    for lam in [.01,.03,.1,.3,1,3,10,30,100]:
        W=np.linalg.solve(A+lam*np.eye(d,dtype=np.float32),B.T).T
        a=acc((Xv@W.T).argmax(1),yv)
        if a>best[0]: best=(a,lam)
    Af=X.T@X; Bf=np.eye(nc,dtype=np.float32)[y].T@X
    W=np.linalg.solve(Af+best[1]*np.eye(d,dtype=np.float32),Bf.T).T.astype(np.float32)
    scores['reversible_ridge_state']=acc((Q@W.T).argmax(1),qy)
    memory['reversible_ridge_state']=int(Af.nbytes+Bf.nbytes)
    detail['reversible_ridge_state']={'validation':best[0],'lambda':best[1]}

    # Hybrid semantic state + deliberately small hard-boundary exception cache.
    for K in [1,2,4,8]:
        hm=HardExceptionMemory(Xt,yt,nc,K)
        bestg=(-1,None)
        for g in [.8,.9,1.0,1.05,1.1,1.2]:
            a=acc(hm.pred(Xv,g),yv)
            if a>bestg[0]: bestg=(a,g)
        hm=HardExceptionMemory(X,y,nc,K)
        name=f'hard_exception_K{K}'
        scores[name]=acc(hm.pred(Q,bestg[1]),qy); memory[name]=int(hm.vectors()*d*4)
        detail[name]={'validation':bestg[0],'gamma':bestg[1],'vectors':hm.vectors()}

    out={
      'benchmark':'MemoryAgentBench official Test_Time_Learning / icl_banking77_5900shot_balance',
      'source':SOURCE,'context_chars':context_chars,'demonstrations':len(X),'questions':len(Q),
      'encoder':'sentence-transformers/all-MiniLM-L6-v2','embedding_dim':d,
      'scores':scores,'memory_bytes_float_state':memory,'detail':detail,
      'best':max(scores.items(),key=lambda z:z[1])
    }
    OUT.write_text(json.dumps(out,indent=2)); print('MAB_OFFICIAL_RESULT='+json.dumps(out))

if __name__=='__main__': main()
