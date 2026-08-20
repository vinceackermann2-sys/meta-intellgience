import argparse, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducible procedural corpus: compact enough for free CPU runners, but tests
# language, code-like structure, instruction following, and continual memory.
VOCAB = 260
PAD, BOS, EOS, SEP = 256, 257, 258, 259

def enc(s): return [BOS] + list(s.encode('utf-8', errors='ignore')) + [EOS]
def dec(xs): return bytes([x for x in xs if x < 256]).decode('utf-8', errors='ignore')

def make_example(rng, kind):
    if kind == 'code':
        a,b = rng.randint(1,99), rng.randint(1,99)
        op = rng.choice([('+','add'),('-','subtract'),('*','multiply')])
        if op[0]=='+': ans=a+b
        elif op[0]=='-': ans=a-b
        else: ans=a*b
        return f"# Task: {op[1]} {a} and {b}\ndef solve():\n    return {a} {op[0]} {b}\n# result {ans}\n"
    if kind == 'chat':
        name=rng.choice(['Ava','Noah','Mia','Leo','Iris']); n=rng.randint(2,12)
        return f"User: My name is {name}. Count to {n}.\nAssistant: " + ', '.join(str(i) for i in range(1,n+1)) + f". Nice to meet you, {name}.\n"
    if kind == 'logic':
        a,b,c=[rng.randint(0,20) for _ in range(3)]
        return f"Question: If x={a}, y={b}, z={c}, what is x+y+z?\nAnswer: {a+b+c}\n"
    # Long-horizon episodic memory pattern.
    who=rng.choice(['Ada','Turing','Grace','Linus','Hedy']); color=rng.choice(['amber','violet','teal','silver','coral'])
    distract=' '.join(rng.choice(['river','stone','cloud','forest','signal','orbit']) for _ in range(40))
    return f"Memory: {who}'s key is {color}. Context: {distract}. Query: What is {who}'s key? Answer: {color}.\n"

def batch(rng, bs, seqlen, device):
    xs=[]
    kinds=['code','chat','logic','memory']
    for _ in range(bs):
        s=''
        while len(enc(s)) < seqlen+1:
            s += make_example(rng, rng.choice(kinds))
        ids=enc(s)[:seqlen+1]
        if len(ids)<seqlen+1: ids += [PAD]*(seqlen+1-len(ids))
        xs.append(ids)
    t=torch.tensor(xs,dtype=torch.long,device=device)
    return t[:,:-1],t[:,1:]

class RMSNorm(nn.Module):
    def __init__(self,d): super().__init__(); self.w=nn.Parameter(torch.ones(d))
    def forward(self,x): return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6) * self.w

class FFN(nn.Module):
    def __init__(self,d,mult=4):
        super().__init__(); h=int(d*mult*2/3); self.a=nn.Linear(d,h,bias=False); self.b=nn.Linear(d,h,bias=False); self.o=nn.Linear(h,d,bias=False)
    def forward(self,x): return self.o(F.silu(self.a(x))*self.b(x))

class CausalAttn(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.h=h; self.dh=d//h; self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
    def forward(self,x):
        B,T,D=x.shape; q,k,v=self.qkv(x).chunk(3,-1)
        q=q.view(B,T,self.h,self.dh).transpose(1,2); k=k.view(B,T,self.h,self.dh).transpose(1,2); v=v.view(B,T,self.h,self.dh).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,D))

class TransformerBlock(nn.Module):
    def __init__(self,d,h): super().__init__(); self.n1=RMSNorm(d); self.a=CausalAttn(d,h); self.n2=RMSNorm(d); self.f=FFN(d)
    def forward(self,x): x=x+self.a(self.n1(x)); return x+self.f(self.n2(x))

class PlasticMemory(nn.Module):
    """Surprise-gated online associative memory.
    Slow projections are learned by SGD. Fast K/V slots are state, updated online
    by a local rule and can persist across calls. No inference-time backprop.
    """
    def __init__(self,d,slots=32,dk=64):
        super().__init__(); self.slots=slots; self.dk=dk
        self.q=nn.Linear(d,dk,bias=False); self.k=nn.Linear(d,dk,bias=False); self.v=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(d,1); self.register_buffer('fast_k',torch.zeros(1,slots,dk),persistent=False); self.register_buffer('fast_v',torch.zeros(1,slots,d),persistent=False); self.register_buffer('age',torch.zeros(1,slots),persistent=False)
    def reset(self,batch,device):
        self.fast_k=torch.zeros(batch,self.slots,self.dk,device=device); self.fast_v=torch.zeros(batch,self.slots,self.o.in_features,device=device); self.age=torch.zeros(batch,self.slots,device=device)
    def forward(self,x,plastic=True):
        B,T,D=x.shape
        if self.fast_k.shape[0]!=B or self.fast_k.device!=x.device: self.reset(B,x.device)
        outs=[]; surprise=[]
        # Sequential on purpose: memory available at t only contains the past.
        for t in range(T):
            z=x[:,t]; q=F.normalize(self.q(z),dim=-1)
            sim=torch.einsum('bd,bsd->bs',q,F.normalize(self.fast_k+1e-6,dim=-1)); w=F.softmax(sim*6.0,dim=-1)
            r=torch.einsum('bs,bsd->bd',w,self.fast_v); pred=self.o(r)
            err=(z-pred).pow(2).mean(-1,keepdim=True); surprise.append(err)
            g=torch.sigmoid(self.gate(z)) * (err/(err.detach().mean()+1e-6)).clamp(0,2)
            outs.append(z + g*pred)
            if plastic:
                # Replace the least-recently useful slot; detach implements a true local fast update.
                with torch.no_grad():
                    idx=self.age.argmax(-1)
                    self.age.add_(1)
                    kk=self.k(z).detach(); vv=self.v(z).detach()
                    for b in range(B):
                        j=int(idx[b]); eta=float(g[b].clamp(.02,.5));
                        self.fast_k[b,j].mul_(1-eta).add_(kk[b],alpha=eta)
                        self.fast_v[b,j].mul_(1-eta).add_(vv[b],alpha=eta)
                        self.age[b,j]=0
        return torch.stack(outs,1), torch.stack(surprise,1).mean()

class Model(nn.Module):
    def __init__(self,arch='transformer',d=512,layers=8,heads=8):
        super().__init__(); self.arch=arch; self.emb=nn.Embedding(VOCAB,d); self.blocks=nn.ModuleList([TransformerBlock(d,heads) for _ in range(layers)])
        self.mem=PlasticMemory(d) if arch=='plastic' else None
        self.norm=RMSNorm(d); self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.emb.weight
        self.future=nn.Linear(d,d,bias=False)
    def reset_memory(self,batch,device):
        if self.mem: self.mem.reset(batch,device)
    def forward(self,ids,plastic=True,return_hidden=False):
        x=self.emb(ids)
        for i,b in enumerate(self.blocks):
            x=b(x)
            if self.mem is not None and i==len(self.blocks)//2: x,surp=self.mem(x,plastic=plastic)
        x=self.norm(x); logits=self.head(x)
        return (logits,x) if return_hidden else logits

def params(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def evaluate(m,device,seqlen=128,batches=8,seed=999):
    m.eval(); rng=random.Random(seed); losses=[]
    for _ in range(batches):
        x,y=batch(rng,4,seqlen,device); m.reset_memory(x.size(0),device); logits=m(x,plastic=True); losses.append(F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)).item())
    return {'loss':sum(losses)/len(losses),'ppl':math.exp(min(20,sum(losses)/len(losses)))}

@torch.no_grad()
def memory_eval(m,device,distances=(8,32,64,96)):
    m.eval(); out={}
    for dist in distances:
        correct=0; total=20
        for i in range(total):
            rng=random.Random(10000+dist*100+i); who=f"P{i}"; val=rng.choice(['amber','violet','teal','silver'])
            filler=' '.join(rng.choice(['aa','bb','cc','dd','ee']) for _ in range(dist))
            prompt=f"Memory: {who} key {val}. {filler} Query: {who} key "
            target=val.encode()[0]
            ids=enc(prompt)[:-1][-127:]; x=torch.tensor([ids],device=device); m.reset_memory(1,device); logits=m(x,plastic=True)
            correct += int(logits[0,-1].argmax().item()==target)
        out[str(dist)]=correct/total
    return out

def sft_batch(rng,bs,seqlen,device):
    # Instruction-heavy mixture for post-training.
    xs=[]
    for _ in range(bs):
        s=''
        while len(enc(s))<seqlen+1: s += make_example(rng,rng.choice(['chat','code','logic']))
        ids=enc(s)[:seqlen+1]; ids += [PAD]*max(0,seqlen+1-len(ids)); xs.append(ids)
    t=torch.tensor(xs,device=device); return t[:,:-1],t[:,1:]

def preference_step(m,opt,rng,device,seqlen=96,beta=.1):
    # DPO-style preference optimization: chosen answer is correct arithmetic,
    # rejected answer is deliberately incorrect. This is preference training,
    # not claimed as human-feedback RLHF.
    a,b=rng.randint(1,30),rng.randint(1,30); good=a+b; bad=good+rng.choice([-3,-2,-1,1,2,3])
    p=f"User: What is {a}+{b}?\nAssistant: "
    ch=enc(p+str(good))[:seqlen]; rj=enc(p+str(bad))[:seqlen]
    def score(ids):
        t=torch.tensor([ids],device=device); logits=m(t[:,:-1],plastic=False); y=t[:,1:]; lp=F.log_softmax(logits,-1).gather(-1,y.unsqueeze(-1)).squeeze(-1); return lp.mean()
    sc,sr=score(ch),score(rj); loss=-F.logsigmoid(beta*(sc-sr)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); return float(loss)

def train(args):
    torch.set_num_threads(max(1,os.cpu_count() or 1)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(args.seed); rng=random.Random(args.seed)
    m=Model(args.arch,d=args.d,layers=args.layers,heads=args.heads).to(device)
    n=params(m); print(json.dumps({'arch':args.arch,'params':n,'device':str(device)}),flush=True)
    if not 23_000_000 <= n <= 29_000_000: print('WARNING parameter count outside 25M target band')
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=.1,betas=(.9,.95))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time(); best=1e9; step0=0
    ck=out/'last.pt'
    if args.resume and ck.exists():
        z=torch.load(ck,map_location=device); m.load_state_dict(z['model']); opt.load_state_dict(z['opt']); step0=z['step']+1; best=z.get('best',best); print('resumed',step0)
    # Phase 1 pretraining: autoregressive + latent future-prediction objective.
    for step in range(step0,args.steps):
        m.train(); x,y=batch(rng,args.bs,args.seqlen,device); m.reset_memory(args.bs,device); logits,h=m(x,plastic=True,return_hidden=True)
        ce=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)); pred=F.normalize(m.future(h[:,:-1]),dim=-1); tgt=F.normalize(h[:,1:].detach(),dim=-1); future=(1-(pred*tgt).sum(-1)).mean(); loss=ce+args.future_weight*future
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%args.log_every==0: print(json.dumps({'phase':'pretrain','step':step,'loss':float(loss),'ce':float(ce),'future':float(future),'sec':time.time()-start}),flush=True)
        if (step+1)%args.eval_every==0 or step==args.steps-1:
            ev=evaluate(m,device,args.seqlen,4); print(json.dumps({'phase':'eval','step':step,**ev}),flush=True); best=min(best,ev['loss']); torch.save({'model':m.state_dict(),'opt':opt.state_dict(),'step':step,'best':best,'args':vars(args)},ck)
    # Phase 2 SFT.
    for s in range(args.sft_steps):
        m.train(); x,y=sft_batch(rng,args.bs,args.seqlen,device); m.reset_memory(args.bs,device); logits=m(x,plastic=True); loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if s%max(1,args.log_every)==0: print(json.dumps({'phase':'sft','step':s,'loss':float(loss)}),flush=True)
    # Phase 3 preference post-training.
    for s in range(args.pref_steps):
        pl=preference_step(m,opt,rng,device)
        if s%max(1,args.log_every)==0: print(json.dumps({'phase':'preference','step':s,'loss':pl}),flush=True)
    final=evaluate(m,device,args.seqlen,8); mem=memory_eval(m,device); result={'arch':args.arch,'params':n,'final':final,'memory':mem,'elapsed_sec':time.time()-start}
    print('RESULT '+json.dumps(result),flush=True); (out/'result.json').write_text(json.dumps(result,indent=2)); torch.save({'model':m.state_dict(),'args':vars(args),'result':result},out/'final.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--arch',choices=['transformer','plastic'],default='plastic'); p.add_argument('--d',type=int,default=512); p.add_argument('--layers',type=int,default=8); p.add_argument('--heads',type=int,default=8); p.add_argument('--steps',type=int,default=220); p.add_argument('--sft-steps',type=int,default=40); p.add_argument('--pref-steps',type=int,default=20); p.add_argument('--bs',type=int,default=2); p.add_argument('--seqlen',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--future-weight',type=float,default=.1); p.add_argument('--eval-every',type=int,default=50); p.add_argument('--log-every',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--out',default='runs/run'); p.add_argument('--resume',action='store_true'); train(p.parse_args())
