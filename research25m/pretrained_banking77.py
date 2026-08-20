from __future__ import annotations
import argparse, csv, io, json, math, statistics
from pathlib import Path
import requests
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

TRAIN_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv'
TEST_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'
MODEL_NAME='EleutherAI/pythia-160m'


def get_csv(url):
    r=requests.get(url,timeout=60); r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def make_slice(train,test,nclasses=10,nwrite=8,nval=2,ntest=20):
    labels=[]
    for row in train:
        if row['category'] not in labels: labels.append(row['category'])
        if len(labels)>=nclasses: break
    tr_by={k:[] for k in labels}; te_by={k:[] for k in labels}
    for row in train:
        if row['category'] in tr_by and len(tr_by[row['category']])<nwrite+nval: tr_by[row['category']].append(row['text'])
    for row in test:
        if row['category'] in te_by and len(te_by[row['category']])<ntest: te_by[row['category']].append(row['text'])
    assert all(len(v)==nwrite+nval for v in tr_by.values())
    assert all(len(v)==ntest for v in te_by.values())
    return labels,tr_by,te_by


@torch.no_grad()
def embed(model,tok,texts,batch=8,max_length=96):
    out=[]
    for i in range(0,len(texts),batch):
        enc=tok(texts[i:i+batch],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
        z=model(**enc).last_hidden_state
        idx=enc['attention_mask'].sum(-1)-1
        h=z[torch.arange(z.shape[0]),idx]
        out.append(F.normalize(h.float(),dim=-1).cpu())
    return torch.cat(out,0)


class DenseFast:
    def __init__(self,d,nc,lr):
        self.W=torch.zeros(nc,d); self.b=torch.zeros(nc); self.lr=lr
    def write(self,h,y):
        p=torch.softmax(self.W@h+self.b,0); e=p.clone(); e[y]-=1
        self.W.add_(torch.outer(e,h),alpha=-self.lr); self.b.add_(e,alpha=-self.lr)
    def pred(self,h): return int((self.W@h+self.b).argmax())


class RoutedFast:
    # Fixed LSH router + bank-local plastic classifier. Router/projections are fixed state,
    # fast trainable values are W only. Choose rank so fast W count ~= dense W count.
    def __init__(self,d,nc,lr,seed=42,banks=16):
        self.banks=banks; self.lr=lr; self.nc=nc
        rank=max(1,(nc*d)//(banks*nc)) # d/banks, so banks*nc*rank ~= nc*d
        self.rank=rank
        g=torch.Generator().manual_seed(seed)
        bits=int(math.log2(banks)); assert 2**bits==banks
        self.hyper=F.normalize(torch.randn(bits,d,generator=g),dim=-1)
        self.proj=F.normalize(torch.randn(banks,rank,d,generator=g),dim=-1)
        self.W=torch.zeros(banks,nc,rank); self.b=torch.zeros(banks,nc)
    def route(self,h):
        signs=(self.hyper@h>0).to(torch.int64); idx=0
        for i,b in enumerate(signs): idx |= int(b)<<i
        return idx
    def feat(self,h,b): return F.normalize(self.proj[b]@h,dim=0)
    def write(self,h,y):
        b=self.route(h); f=self.feat(h,b); p=torch.softmax(self.W[b]@f+self.b[b],0); e=p.clone();e[y]-=1
        self.W[b].add_(torch.outer(e,f),alpha=-self.lr);self.b[b].add_(e,alpha=-self.lr)
    def pred(self,h):
        b=self.route(h);f=self.feat(h,b);return int((self.W[b]@f+self.b[b]).argmax())
    def fast_params(self): return self.W.numel()+self.b.numel()


def accuracy(pred,y): return sum(int(a==b) for a,b in zip(pred,y))/len(y)


def main(a):
    torch.set_num_threads(4)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train,test=get_csv(TRAIN_URL),get_csv(TEST_URL)
    labels,tr_by,te_by=make_slice(train,test,a.classes,a.write,a.val,a.test)
    l2i={x:i for i,x in enumerate(labels)}
    write_text=[];write_y=[];val_text=[];val_y=[];test_text=[];test_y=[]
    for lab in labels:
        write_text+=tr_by[lab][:a.write];write_y += [l2i[lab]]*a.write
        val_text+=tr_by[lab][a.write:];val_y += [l2i[lab]]*a.val
        test_text+=te_by[lab];test_y += [l2i[lab]]*a.test
    print(json.dumps({'model':MODEL_NAME,'classes':labels,'n_write':len(write_text),'n_val':len(val_text),'n_test':len(test_text)}),flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(MODEL_NAME);model.eval()
    d=model.config.hidden_size
    Hwrite=embed(model,tok,write_text,a.batch);Hval=embed(model,tok,val_text,a.batch);Htest=embed(model,tok,test_text,a.batch)
    del model

    # Retrieval / representation baselines.
    sim=Htest@Hwrite.T; knn=[write_y[int(j)] for j in sim.argmax(-1)]
    cents=torch.stack([F.normalize(Hwrite[[i for i,y in enumerate(write_y) if y==c]].mean(0),dim=0) for c in range(len(labels))])
    centroid=(Htest@cents.T).argmax(-1).tolist()
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1)
    X=vec.fit_transform(write_text);Q=vec.transform(test_text);nn=np.asarray(cosine_similarity(Q,X).argmax(1)).ravel()
    tfidf=[write_y[int(j)] for j in nn]

    # Tune online update LR on held-out real validation examples only.
    lrs=[.03,.1,.3,.7,1.5,3.0]
    dval={}
    for lr in lrs:
        m=DenseFast(d,len(labels),lr)
        for h,y in zip(Hwrite,write_y):m.write(h,y)
        dval[str(lr)]=accuracy([m.pred(h) for h in Hval],val_y)
    dlr=float(max(dval,key=lambda x:(dval[x],-float(x))))

    routed_seeds=[41,42,43]
    rval={}
    for seed in routed_seeds:
        rval[str(seed)]={}
        for lr in lrs:
            m=RoutedFast(d,len(labels),lr,seed)
            for h,y in zip(Hwrite,write_y):m.write(h,y)
            rval[str(seed)][str(lr)]=accuracy([m.pred(h) for h in Hval],val_y)
    # One shared LR selected by mean validation across router seeds.
    rmeans={str(lr):statistics.mean(rval[str(s)][str(lr)] for s in routed_seeds) for lr in lrs}
    rlr=float(max(rmeans,key=lambda x:(rmeans[x],-float(x))))

    # Balanced streaming order: one example per class each round.
    curves={'dense':{},'routed_mean':{},'routed_by_seed':{str(s):{} for s in routed_seeds}}
    for k in [1,2,4,a.write]:
        idx=[]
        for j in range(k):
            for c in range(len(labels)):idx.append(c*a.write+j)
        dm=DenseFast(d,len(labels),dlr)
        for i in idx:dm.write(Hwrite[i],write_y[i])
        curves['dense'][str(k)]=accuracy([dm.pred(h) for h in Htest],test_y)
        vals=[]
        for seed in routed_seeds:
            rm=RoutedFast(d,len(labels),rlr,seed)
            for i in idx:rm.write(Hwrite[i],write_y[i])
            ac=accuracy([rm.pred(h) for h in Htest],test_y);vals.append(ac);curves['routed_by_seed'][str(seed)][str(k)]=ac
        curves['routed_mean'][str(k)]=statistics.mean(vals)

    probe=RoutedFast(d,len(labels),rlr,42)
    result={
      'dataset':'Banking77','real_data':True,'model':MODEL_NAME,'hidden_size':d,'classes':labels,
      'counts':{'write':len(write_text),'validation':len(val_text),'test':len(test_text)},
      'baselines':{'hidden_1nn':accuracy(knn,test_y),'hidden_centroid':accuracy(centroid,test_y),'tfidf_char_1nn':accuracy(tfidf,test_y)},
      'lr_validation':{'dense':dval,'routed_by_seed':rval,'routed_mean':rmeans,'selected_dense':dlr,'selected_routed':rlr},
      'test_curve_examples_per_class':curves,
      'fast_params':{'dense':len(labels)*d+len(labels),'routed':probe.fast_params(),'routed_rank':probe.rank,'routed_banks':probe.banks},
      'gate':{'chance':1/len(labels),'passes_semantic_generalization':curves['routed_mean'][str(a.write)]>accuracy(tfidf,test_y)}
    }
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/pretrained-banking77');p.add_argument('--classes',type=int,default=10);p.add_argument('--write',type=int,default=8);p.add_argument('--val',type=int,default=2);p.add_argument('--test',type=int,default=20);p.add_argument('--batch',type=int,default=8);main(p.parse_args())
