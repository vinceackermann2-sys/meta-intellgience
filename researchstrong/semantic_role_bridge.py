from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile, random
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af

OUT = Path(__file__).with_name('semantic_role_bridge_result.json')
DEV_END = 100
TOP_EPISODES = 3
RANK = 32
EPOCHS = 120
NEG_PER_POS = 24
SEED = 20260820


def canon(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).casefold()).strip()


def leaf(key):
    x = str(key).split('.')[-1]
    return re.sub(r'\[\d+\]$', '', x)


def target_role(qa, p, d):
    schema = qa.get('target_tool_schema') or {}
    return (
        f"target action {schema.get('name','')}; target semantic role {p}; "
        f"meaning {d.get('description','')}; type {d.get('type','')}; "
        f"current intent {qa.get('query','')}"
    )


def source_role(c):
    # IMPORTANT: candidate value is intentionally excluded.
    key = str(c.get('key',''))
    addr = str(c.get('address',''))
    tool = str(c.get('tool',''))
    kind = str(c.get('kind',''))
    role = 'unknown'
    if addr.startswith('role user'): role = 'user_request'
    elif addr.startswith('role assistant'): role = 'assistant_action'
    elif addr.startswith('role tool'): role = 'tool_result'
    return (
        f"historical {role}; source action {tool}; source semantic role {leaf(key)}; "
        f"field path {key}; record context {addr[:1000]}; kind {kind}"
    )


def norm_eq(a,b):
    return base.norm(a) == base.norm(b)


def build_slots(enc):
    td = Path(tempfile.gettempdir())/'semantic_role_bridge'; td.mkdir(exist_ok=True)
    qp, cp = td/'qa.jsonl', td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl', qp)
    base.fetch(base.BASE+'toolmem_conversation.jsonl', cp)
    qas = list(base.load_jsonl(qp))[:DEV_END]
    sessions, by = base.build_session_map(cp)
    cache = {}
    slots = []
    missing = []
    for qi, qa in enumerate(qas):
        ses = base.find_session(qa, sessions, by)
        if ses is None:
            missing.append(qi); continue
        eps = es.episodes(ses)
        schema = qa.get('target_tool_schema') or {}
        tool = str(schema.get('name',''))
        props = ((schema.get('parameters') or {}).get('properties') or {})
        gold = ((qa.get('tool_call') or {}).get('arguments') or {})
        grounding = ((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            d = props.get(p) or {}
            ttext = target_role(qa,p,d)
            picked = es.retrieve_episodes(enc, eps, ttext, TOP_EPISODES)
            cands=[]
            for erank, ep, esim in picked:
                for c in af.occurrences(ep, erank, esim):
                    if c.get('kind') == 'text_span':
                        continue
                    # preserve addresses; values exist only as executor payload / label check
                    cands.append(c)
            # de-dupe addresses, not values
            seen=set(); uniq=[]
            for c in cands:
                sig=(str(c.get('key','')),str(c.get('tool','')),int(c.get('turn',-1)),str(c.get('address','')))
                if sig in seen: continue
                seen.add(sig); uniq.append(c)
            cands=uniq
            if not cands:
                slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'tool':tool,'p':p,'grounding':str((grounding.get(p) or {}).get('type','unknown')),'gold':g,'target_text':ttext,'cands':[],'labels':[]})
                continue
            texts=[ttext]+[source_role(c) for c in cands]
            # global text cache for exact repeated strings
            missing_text=[x for x in texts if x not in cache]
            if missing_text:
                E=enc.encode(missing_text,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
                for x,e in zip(missing_text,E): cache[x]=e
            te=cache[ttext]
            se=np.stack([cache[source_role(c)] for c in cands]).astype(np.float32)
            labels=[int(norm_eq(c.get('value'),g)) for c in cands]
            struct=[]
            for c in cands:
                k=str(c.get('key','')); ct=str(c.get('tool',''))
                struct.append([
                    float(canon(leaf(k))==canon(p) and bool(leaf(k))),
                    float(canon(ct)==canon(tool) and bool(ct)),
                    float(c.get('rank',9)==0),
                    float(c.get('rank',9)==1),
                    float(c.get('rank',9)==2),
                    float(str(c.get('address','')).startswith('role tool')),
                    float(str(c.get('address','')).startswith('role assistant')),
                    float(str(c.get('address','')).startswith('role user')),
                ])
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'tool':tool,'p':p,'grounding':str((grounding.get(p) or {}).get('type','unknown')),'gold':g,'target_text':ttext,'te':te,'se':se,'struct':np.asarray(struct,np.float32),'cands':cands,'labels':labels})
        if qi%10==0: print('ROLE_BRIDGE_BUILD',qi,'slots',len(slots),'cache',len(cache),flush=True)
    return slots,missing


class RoleBridge(nn.Module):
    def __init__(self, dim, rank, sdim):
        super().__init__()
        self.tproj=nn.Linear(dim,rank,bias=False)
        self.sproj=nn.Linear(dim,rank,bias=False)
        self.struct=nn.Linear(sdim,1,bias=False)
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,t,s,st):
        # low-rank bilinear role compatibility + small structural prior
        a=self.tproj(t); b=self.sproj(s)
        return (a*b).sum(-1)/(a.shape[-1]**0.5) + self.struct(st).squeeze(-1)+self.bias


def make_pairs(slots, train_idx, rng):
    rows=[]
    for si in train_idx:
        s=slots[si]
        pos=[i for i,y in enumerate(s.get('labels',[])) if y]
        neg=[i for i,y in enumerate(s.get('labels',[])) if not y]
        if not pos or not neg: continue
        # each positive versus sampled negatives; pairwise logistic loss
        for pi in pos[:3]:
            sample=neg if len(neg)<=NEG_PER_POS else rng.sample(neg,NEG_PER_POS)
            for ni in sample:
                rows.append((si,pi,ni))
    return rows


def train_fold(slots, train_idx, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dim=len(slots[next(i for i in train_idx if 'te' in slots[i])]['te'])
    sdim=slots[next(i for i in train_idx if 'te' in slots[i])]['struct'].shape[1]
    model=RoleBridge(dim,RANK,sdim)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-3)
    rng=random.Random(seed)
    pairs=make_pairs(slots,train_idx,rng)
    if not pairs: return model,0
    for ep in range(EPOCHS):
        rng.shuffle(pairs)
        total=0.0
        for off in range(0,len(pairs),256):
            batch=pairs[off:off+256]
            tp=[];sp=[];stp=[];tn=[];sn=[];stn=[]
            for si,pi,ni in batch:
                s=slots[si]
                tp.append(s['te']); sp.append(s['se'][pi]); stp.append(s['struct'][pi])
                tn.append(s['te']); sn.append(s['se'][ni]); stn.append(s['struct'][ni])
            tp=torch.tensor(np.asarray(tp)); sp=torch.tensor(np.asarray(sp)); stp=torch.tensor(np.asarray(stp))
            tn=torch.tensor(np.asarray(tn)); sn=torch.tensor(np.asarray(sn)); stn=torch.tensor(np.asarray(stn))
            margin=model(tp,sp,stp)-model(tn,sn,stn)
            loss=torch.nn.functional.softplus(-margin).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(batch)
    return model,len(pairs)


def evaluate(slots, test_idx, model):
    by=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0))
    errors=[]
    model.eval()
    with torch.no_grad():
        for si in test_idx:
            s=slots[si]; typ=s['grounding']; m=by[typ]; m['n']+=1
            if 'te' not in s or len(s['cands'])==0: continue
            pos=[i for i,y in enumerate(s['labels']) if y]
            m['covered']+=int(bool(pos))
            t=np.repeat(s['te'][None,:],len(s['cands']),axis=0)
            score=model(torch.tensor(t),torch.tensor(s['se']),torch.tensor(s['struct'])).numpy()
            order=np.argsort(-score)
            if pos:
                m['top1']+=int(int(order[0]) in pos)
                m['top3']+=int(any(int(i) in pos for i in order[:3]))
                m['top5']+=int(any(int(i) in pos for i in order[:5]))
                if typ=='explicit' and int(order[0]) not in pos and len(errors)<12:
                    errors.append({'qa_id':s['qa_id'],'tool':s['tool'],'parameter':s['p'],'top_fields':[{'key':s['cands'][int(i)].get('key'),'tool':s['cands'][int(i)].get('tool'),'score':float(score[int(i)])} for i in order[:4]],'positive_fields':[{'key':s['cands'][i].get('key'),'tool':s['cands'][i].get('tool')} for i in pos[:4]]})
    out={}
    for typ,c in by.items():
        n=max(1,c['n']); cov=max(1,c['covered'])
        out[typ]={'n':c['n'],'coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov,'top3_given_covered':c['top3']/cov}
    return out,errors


def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    slots,missing=build_slots(enc)
    eligible=[i for i,s in enumerate(slots) if 'te' in s]
    groups=np.array([slots[i]['tool'] or '__none__' for i in eligible])
    X=np.zeros((len(eligible),1)); y=np.zeros(len(eligible))
    # strict unseen-tool folds; reduce splits if needed
    n_groups=len(set(groups.tolist())); n_splits=min(5,max(2,n_groups))
    gkf=GroupKFold(n_splits=n_splits)
    folds=[]; agg=defaultdict(list); pair_counts=[]
    all_errors=[]
    for fi,(tr0,te0) in enumerate(gkf.split(X,y,groups)):
        tr=[eligible[int(j)] for j in tr0]; te=[eligible[int(j)] for j in te0]
        model,npairs=train_fold(slots,tr,SEED+fi); pair_counts.append(npairs)
        metrics,errs=evaluate(slots,te,model); folds.append({'fold':fi,'train_slots':len(tr),'test_slots':len(te),'train_tool_groups':len(set(slots[i]['tool'] for i in tr)),'test_tool_groups':len(set(slots[i]['tool'] for i in te)),'pairs':npairs,'metrics':metrics})
        for typ,m in metrics.items():
            for k,v in m.items():
                if k!='n': agg[(typ,k)].append(v)
        all_errors.extend(errs[:3])
        print('ROLE_BRIDGE_FOLD',fi,json.dumps(metrics),flush=True)
    mean={}
    for (typ,k),vals in agg.items(): mean.setdefault(typ,{})[k]=float(np.mean(vals))
    # baseline cosine on same eligible slots, grouped evaluation unnecessary because no training
    baseline=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0))
    for i in eligible:
        s=slots[i]; typ=s['grounding']; m=baseline[typ]; m['n']+=1
        pos=[j for j,z in enumerate(s['labels']) if z]; m['covered']+=int(bool(pos))
        score=s['se']@s['te']; order=np.argsort(-score)
        if pos:
            m['top1']+=int(int(order[0]) in pos); m['top3']+=int(any(int(j) in pos for j in order[:3]))
    bpack={}
    for typ,c in baseline.items():
        n=max(1,c['n']); bpack[typ]={'n':c['n'],'coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n}
    result={
      'stage':'Semantic world-model role bridge v1',
      'protocol':'QA001-100 development only; GroupKFold by target tool name so held-out schemas are unseen during bridge training; QA101-400 gold remains sealed',
      'architecture':'value-masked target role embedding <-> value-masked historical field/record role embedding via learned rank-32 bilinear bridge; exact candidate values are labels/executor payloads only',
      'baseline_cosine':bpack,'mean_group_cv':mean,'folds':folds,'pair_counts':pair_counts,'missing_zero_based':missing,'sample_errors':all_errors,
      'pass_signal':'Useful only if unseen-tool top1 materially exceeds cosine/previous ~28-33% explicit selection without reducing coverage. This is not a breakthrough gate by itself.',
      'guardrail':'No candidate value text is used by the bridge. Gold QA001-100 values label positive source addresses only. No QA101-400 gold is read.'
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('SEMANTIC_ROLE_BRIDGE='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
