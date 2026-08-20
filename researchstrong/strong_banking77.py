import csv,io,json,random,urllib.request
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TRAIN_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv'
TEST_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'
def read(url):
    with urllib.request.urlopen(url,timeout=60) as r:rows=list(csv.DictReader(io.StringIO(r.read().decode())))
    by={}
    for z in rows:by.setdefault(z['category'],[]).append(z['text'])
    return by
def acc(p,y):return float(np.mean(np.asarray(p)==np.asarray(y)))
def main():
    tr,te=read(TRAIN_URL),read(TEST_URL);labs=sorted(set(tr)&set(te));L={x:i for i,x in enumerate(labs)}
    texts=[];sp={};k=0
    for D,n in [(tr,'tr'),(te,'te')]:
      for l in labs:
        a=k;texts+=D[l];k+=len(D[l]);sp[(n,l)]=(a,k)
    m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    E=m.encode(texts,batch_size=128,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
    Q=np.concatenate([E[slice(*sp[('te',l)])] for l in labs]);qy=np.concatenate([np.full(len(te[l]),L[l]) for l in labs])
    runs=[]
    for seed in [11,23,41,57,73]:
      rng=random.Random(seed);write=[];val=[];g2y={}
      for l in labs:
        a,b=sp[('tr',l)];inds=list(range(a,b));rng.shuffle(inds);write+=inds[:4];val+=inds[4:8]
        for i in range(a,b):g2y[i]=L[l]
      X,V=E[write],E[val];y=np.array([g2y[i] for i in write]);vy=np.array([g2y[i] for i in val])
      models={}
      # hyperparameter tune only on validation examples not used in fitting
      for name,grid,builder in [
        ('logreg',[.01,.03,.1,.3,1,3,10],lambda c:LogisticRegression(C=c,max_iter=3000,solver='lbfgs')),
        ('linear_svm',[.001,.003,.01,.03,.1,.3,1,3],lambda c:LinearSVC(C=c,max_iter=10000)),
        ('ridge',[.01,.03,.1,.3,1,3,10,30],lambda a:RidgeClassifier(alpha=a)),
      ]:
        best=None
        for h in grid:
          mm=builder(h);mm.fit(X,y);s=acc(mm.predict(V),vy)
          if best is None or (s,-float(h))>(best[0],-float(best[1])):best=(s,h,mm)
        models[name]=(best[1],best[2])
      # shrinkage LDA; no tune
      lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto');lda.fit(X,y);models['shrinkage_lda']=('auto',lda)
      # online SGD: one pass only, tune eta0/alpha; true streaming baseline
      order=list(range(len(X)));random.Random(seed+1000).shuffle(order)
      best=None
      for alpha in [1e-6,1e-5,1e-4,1e-3]:
        for eta in [.001,.003,.01,.03,.1]:
          mm=SGDClassifier(loss='log_loss',alpha=alpha,learning_rate='constant',eta0=eta,random_state=seed)
          first=True
          for j in order:
            mm.partial_fit(X[j:j+1],y[j:j+1],classes=np.arange(77) if first else None);first=False
          s=acc(mm.predict(V),vy)
          if best is None or s>best[0]:best=(s,alpha,eta,mm)
      models['online_sgd']=((best[1],best[2]),best[3])
      scores={n:acc(mm.predict(Q),qy) for n,(h,mm) in models.items()}
      runs.append({'seed':seed,'scores':scores,'hyper':{n:str(h) for n,(h,_) in models.items()}})
    methods=list(runs[0]['scores']);summary={}
    for n in methods:
      v=np.array([r['scores'][n] for r in runs]);summary[n]={'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max())}
    out={'benchmark':'Banking77 official test, 77 intents, 4-shot, strong supervised baselines','encoder':'all-MiniLM-L6-v2','runs':runs,'summary':summary}
    Path('researchstrong/strong_banking77_result.json').write_text(json.dumps(out,indent=2));print('STRONG_RESULT='+json.dumps(out))
if __name__=='__main__':main()
