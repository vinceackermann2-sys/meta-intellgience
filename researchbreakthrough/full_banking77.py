import csv, io, json, math, os, random, urllib.request
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

TRAIN_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv'
TEST_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'

def read_csv(url):
    with urllib.request.urlopen(url, timeout=60) as r: txt=r.read().decode('utf-8')
    rows=list(csv.DictReader(io.StringIO(txt)))
    by={}
    for z in rows: by.setdefault(z['category'],[]).append(z['text'])
    return by

def norm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)

class DenseFast:
    def __init__(self,nc,d,lr): self.W=np.zeros((nc,d),np.float32);self.lr=lr
    def write(self,x,y):
        z=self.W@x;z-=z.max();p=np.exp(z);p/=p.sum();p[y]-=1
        self.W-=self.lr*np.outer(p,x)
    def pred(self,Q):return (Q@self.W.T).argmax(1)

class Proto:
    def __init__(self,nc,K,th):self.ps=[[] for _ in range(nc)];self.ns=[[] for _ in range(nc)];self.K=K;self.th=th
    def write(self,x,y):
        ps=self.ps[y]
        if not ps:ps.append(x.copy());self.ns[y].append(1);return
        P=norm(np.stack(ps));s=P@x;j=int(s.argmax())
        if len(ps)<self.K and s[j]<self.th:ps.append(x.copy());self.ns[y].append(1)
        else:
            self.ns[y][j]+=1;eta=1/self.ns[y][j];ps[j]=(1-eta)*ps[j]+eta*x
    def pred(self,Q):
        out=np.full((len(Q),len(self.ps)),-1e9,np.float32)
        for c,ps in enumerate(self.ps):
            if ps:out[:,c]=(Q@norm(np.stack(ps)).T).max(1)
        return out.argmax(1)
    def count(self):return sum(len(x) for x in self.ps)

def ridge_fit(X,y,nc,lam):
    # reversible sufficient-statistic fast learner: W = Y^T X (X^T X + lam I)^-1
    d=X.shape[1];Y=np.eye(nc,dtype=np.float32)[y]
    A=X.T@X + lam*np.eye(d,dtype=np.float32)
    B=Y.T@X
    # solve A * Z = B.T, W=Z.T
    return np.linalg.solve(A,B.T).T.astype(np.float32)

def acc(p,y):return float(np.mean(np.asarray(p)==np.asarray(y)))

def main():
    train=read_csv(TRAIN_URL);test=read_csv(TEST_URL)
    labels=sorted(set(train)&set(test));L={x:i for i,x in enumerate(labels)};nc=len(labels)
    assert nc==77,(nc,labels)
    alltxt=[];spans={};k=0
    for split,name in [(train,'tr'),(test,'te')]:
        for lab in labels:
            a=k;alltxt.extend(split[lab]);k+=len(split[lab]);spans[(name,lab)]=(a,k)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    E=enc.encode(alltxt,batch_size=128,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
    d=E.shape[1]
    Q=np.concatenate([E[slice(*spans[('te',lab)])] for lab in labels])
    qy=np.concatenate([np.full(len(test[lab]),L[lab],np.int64) for lab in labels])
    seeds=[11,23,41,57,73]
    runs=[]
    for seed in seeds:
        rng=random.Random(seed);write=[];val=[]
        for lab in labels:
            a,b=spans[('tr',lab)];inds=list(range(a,b));rng.shuffle(inds);write+=inds[:4];val+=inds[4:6]
        wy=[];vy=[]
        for lab in labels:
            a,b=spans[('tr',lab)];s=set(range(a,b));
        # map global train embedding index to class
        g2y={i:L[lab] for lab in labels for i in range(*spans[('tr',lab)])}
        wy=np.array([g2y[i] for i in write]);vy=np.array([g2y[i] for i in val])
        X=E[write];V=E[val]
        # retrieval baselines
        S=Q@X.T; p1=wy[S.argmax(1)]
        k3=np.argpartition(S,-3,axis=1)[:,-3:]
        # similarity-weighted 3NN vote
        p3=[]
        for r,ix in enumerate(k3):
            sc={}
            for j in ix:sc[int(wy[j])]=sc.get(int(wy[j]),0.0)+float(max(0,S[r,j]))
            p3.append(max(sc,key=sc.get))
        # centroid
        C=np.stack([norm(X[wy==c].mean(0,keepdims=True))[0] for c in range(nc)]);pc=(Q@C.T).argmax(1)
        # dense fast tune on validation only
        dense_grid=[]
        for lr in [.03,.1,.3,1.,3.,10.]:
            m=DenseFast(nc,d,lr)
            order=list(range(len(X)));rng.shuffle(order)
            for j in order:m.write(X[j],int(wy[j]))
            dense_grid.append((acc(m.pred(V),vy),lr))
        dlr=max(dense_grid)[1];dm=DenseFast(nc,d,dlr)
        order=list(range(len(X)));random.Random(seed+999).shuffle(order)
        for j in order:dm.write(X[j],int(wy[j]))
        pd=dm.pred(Q)
        # adaptive prototypes tuned on validation
        pg=[]
        for K in [1,2,3,4]:
            for th in [.35,.5,.65,.75,.85]:
                m=Proto(nc,K,th)
                for j in order:m.write(X[j],int(wy[j]))
                pg.append((acc(m.pred(V),vy),-m.count(),K,th))
        _,_,pk,pth=max(pg);pm=Proto(nc,pk,pth)
        for j in order:pm.write(X[j],int(wy[j]))
        pp=pm.pred(Q)
        # ridge / orthogonal fast weights tune lambda on validation
        rg=[]
        for lam in [.01,.03,.1,.3,1.,3.,10.,30.]:
            W=ridge_fit(X,wy,nc,lam);rg.append((acc((V@W.T).argmax(1),vy),lam))
        rlam=max(rg)[1];W=ridge_fit(X,wy,nc,rlam);pr=(Q@W.T).argmax(1)
        runs.append({'seed':seed,'accuracy':{'1nn':acc(p1,qy),'3nn':acc(p3,qy),'centroid':acc(pc,qy),'dense_fast':acc(pd,qy),'adaptive_proto':acc(pp,qy),'reversible_ridge_fast':acc(pr,qy)},
                     'selected':{'dense_lr':dlr,'proto_K':pk,'proto_threshold':pth,'ridge_lambda':rlam},'prototype_vectors':pm.count()})
    methods=list(runs[0]['accuracy']);summary={}
    for m in methods:
        v=np.array([r['accuracy'][m] for r in runs]);summary[m]={'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max())}
    # float32 memory, ignoring small labels/metadata
    nwrite=77*4
    mem={'1nn_bytes':nwrite*d*4,'centroid_bytes':77*d*4,'dense_fast_bytes':77*d*4,'ridge_fast_W_bytes':77*d*4,'ridge_reversible_cov_bytes':d*d*4,
         'adaptive_proto_bytes_mean':float(np.mean([r['prototype_vectors'] for r in runs])*d*4)}
    out={'benchmark':'Banking77 full official test, 77 intents, 4-shot per intent','encoder':'sentence-transformers/all-MiniLM-L6-v2','embedding_dim':d,
         'train_examples':sum(map(len,train.values())),'test_examples':len(Q),'seeds':seeds,'summary':summary,'memory':mem,'runs':runs}
    Path('researchbreakthrough/full_banking77_result.json').write_text(json.dumps(out,indent=2))
    print('FINAL_RESULT='+json.dumps(out))
if __name__=='__main__':main()
