import json,re,string
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT=Path('researchbreakthrough/full_banking77_result.json')
PATTERNS=[
('P19','place of birth',re.compile(r'^(.+?) was born in the city of (.+?)\.$')),('P20','place of death',re.compile(r'^(.+?) died in the city of (.+?)\.$')),
('P413','position played on team',re.compile(r'^(.+?) plays the position of (.+?)\.$')),('P641','sport',re.compile(r'^(.+?) is associated with the sport of (.+?)\.$')),
('P30','continent',re.compile(r'^(.+?) is located in the continent of (.+?)\.$')),('P937','work location',re.compile(r'^(.+?) worked in the city of (.+?)\.$')),
('P26','spouse',re.compile(r'^(.+?) is married to (.+?)\.$')),('P27','country of citizenship',re.compile(r'^(.+?) is a citizen of (.+?)\.$')),
('P175','performer',re.compile(r'^(.+?) was performed by (.+?)\.$')),('P108','employer',re.compile(r'^(.+?) is employed by (.+?)\.$')),
('P112','founder',re.compile(r'^(.+?) was founded by (.+?)\.$')),('P170','creator',re.compile(r'^(.+?) was created by (.+?)\.$')),
('P178','developer',re.compile(r'^(.+?) was developed by (.+?)\.$')),('P800','notable work',re.compile(r'^(.+?) is famous for (.+?)\.$')),
('P1412','spoken language',re.compile(r'^(.+?) speaks the language of (.+?)\.$')),('P407','language of work',re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
('P140','religion',re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),('P106','occupation',re.compile(r'^(.+?) works in the field of (.+?)\.$')),
('P495','country of origin',re.compile(r'^(.+?) was created in the country of (.+?)\.$')),('P740','place founded',re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
('P740','place founded',re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),('P40','child',re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
('P40','child',re.compile(r"^(.+?)'s child is (.+?)\.$")),('P36','capital',re.compile(r'^The capital of (.+?) is (.+?)\.$')),
('P159','headquarters location',re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),('P50','author',re.compile(r'^The author of (.+?) is (.+?)\.$')),
('P69','educational institution',re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),('P1037','director or manager',re.compile(r'^The director of (.+?) is (.+?)\.$')),
('P488','chairperson',re.compile(r'^The chairperson of (.+?) is (.+?)\.$')),('P169','chief executive officer',re.compile(r'^The chief executive officer of (.+?) is (.+?)\.$')),
('P449','original broadcaster',re.compile(r'^The (?:origianl|original) broadcaster of (.+?) is (.+?)\.$')),('P176','producer',re.compile(r'^The company that produced (.+?) is (.+?)\.$')),
('P136','music genre',re.compile(r'^The type of music that (.+?) plays is (.+?)\.$')),('P37','official language',re.compile(r'^The official language of (.+?) is (.+?)\.$')),
('P364','original language',re.compile(r'^The original language of (.+?) is (.+?)\.$')),('P35','head of state',re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
('P35','head of state',re.compile(r'^The name of the current head of state in (.+?) is (.+?)\.$')),('P6','head of government',re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
('P286','coach',re.compile(r'^The coach of (.+?) is (.+?)\.$')),('P286','coach',re.compile(r'^The head coach of (.+?) is (.+?)\.$'))]
OFFICE=re.compile(r'^The (.+?) is (.+?)\.$')

def parse_fact(t):
    for rid,name,p in PATTERNS:
        m=p.match(t)
        if m:return m.group(1).strip(),rid,m.group(2).strip()
    m=OFFICE.match(t); return (m.group(1).strip(),'P1308',m.group(2).strip()) if m else None

def norm(s):
    s=str(s).lower();s=''.join(c for c in s if c not in string.punctuation);s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())
def golds(a):
    if isinstance(a,str):return [a]
    if isinstance(a,list) and a and isinstance(a[0],list):return [x for z in a for x in z]
    return list(a or [])
def exact(x,a):return any(norm(x)==norm(g) for g in golds(a))

def compile_row(row,K=2):
    reg=defaultdict(dict);nf=0
    for line in row['context'].splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if not m:continue
        z=parse_fact(m.group(2))
        if not z:continue
        nf+=1;ser=int(m.group(1));s,r,o=z;reg[(s,r)][o]=max(ser,reg[(s,r)].get(o,-1))
    adj=defaultdict(list)
    for (s,r),vals in reg.items():
        vs=sorted(((ser,o) for o,ser in vals.items()),reverse=True)[:K]
        for rank,(ser,o) in enumerate(vs):adj[s].append((r,o,ser,rank))
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26':adj[o].append((r,s,ser,rank))
    return adj,nf,len(reg),sum(min(K,len(v)) for v in reg.values())

def anchor(q,adj):
    qf=q.casefold();c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in adj if norm(e) and norm(e) in nq];return max(c,key=len) if c else None

def template(q,st):
    x=q.casefold()
    if st:x=re.sub(re.escape(st.casefold()),'<ent>',x)
    return re.sub(r'\s+',' ',x).strip()

def paths(st,adj,maxh=4,cap=50000):
    if not st:return []
    out=[];stack=[(st,[],{st})]
    while stack and len(out)<cap:
        node,es,seen=stack.pop()
        if es:out.append(es)
        if len(es)>=maxh:continue
        for r,o,ser,rank in adj.get(node,[]):
            if o not in seen:stack.append((o,es+[(node,r,o,ser,rank)],seen|{o}))
    return out

def seq(p):return tuple(e[1] for e in p)
def ranks(p):return tuple(e[4] for e in p)

def build(row):
    adj,nf,nreg,nedge=compile_row(row,2);E=[]
    for q,a in zip(row['questions'],row['answers']):
        st=anchor(q,adj);ps=paths(st,adj);gp=[p for p in ps if exact(p[-1][2],a)]
        gp.sort(key=lambda p:(len(p),sum(ranks(p)),max(e[3] for e in p)-min(e[3] for e in p)))
        E.append({'q':q,'a':a,'st':st,'tpl':template(q,st),'paths':ps,'gold':gp,'nf':nf})
    return {'facts':nf,'registers':nreg,'edges':nedge,'examples':E}

def train_program(dev):
    tr=[]
    for e in dev:
        if e['gold']:tr.append((e['tpl'],'|'.join(seq(e['gold'][0]))))
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,6),sublinear_tf=True)
    X=vec.fit_transform([x[0] for x in tr]);return tr,vec,X

def programs(tpl,tr,vec,X,k=30):
    sim=cosine_similarity(vec.transform([tpl]),X).ravel();idx=np.argsort(-sim)[:min(k,len(sim))];vote=defaultdict(float)
    for j,i in enumerate(idx):vote[tr[i][1]]+=(max(0,float(sim[i]))**3)+1e-8/(j+1)
    return [tuple(z.split('|')) for z,_ in sorted(vote.items(),key=lambda kv:kv[1],reverse=True)[:5]],float(sim[idx[0]]) if len(idx) else 0.0

def choose(ps,policy,nf):
    if not ps:return None
    def serials(p):return [e[3] for e in p]
    def current(p):return sum(e[4] for e in p)
    if policy=='all_current':
        return min(ps,key=lambda p:(current(p),max(serials(p))-min(serials(p))))
    if policy=='first_current_minspan':
        z=[p for p in ps if p[0][4]==0] or ps
        return min(z,key=lambda p:(max(serials(p))-min(serials(p)),current(p)))
    if policy=='first_current_shadow_rest':
        z=[p for p in ps if p[0][4]==0] or ps
        return min(z,key=lambda p:(-sum(e[4] for e in p[1:]),max(serials(p))-min(serials(p))))
    if policy=='snapshot_first':
        z=[p for p in ps if p[0][4]==0] or ps
        def key(p):
            s0=p[0][3];down=serials(p)[1:]
            viol=sum(s>s0 for s in down)
            age=sum((s0-s) for s in down if s<=s0)
            future=sum((s-s0) for s in down if s>s0)
            return (viol,future,age,current(p))
        return min(z,key=key)
    if policy=='snapshot_monotone':
        z=[p for p in ps if p[0][4]==0] or ps
        def key(p):
            ss=serials(p);inc=sum(max(0,ss[i]-ss[i-1]) for i in range(1,len(ss)));gap=sum(abs(ss[i]-ss[i-1]) for i in range(1,len(ss)))
            return (inc,gap,current(p))
        return min(z,key=key)
    if policy=='min_span':
        return min(ps,key=lambda p:(max(serials(p))-min(serials(p)),current(p)))
    raise ValueError(policy)

POLICIES=['all_current','first_current_minspan','first_current_shadow_rest','snapshot_first','snapshot_monotone','min_span']

def version_score(dev,policy,length=None,program=None):
    ok=n=0
    for e in dev:
        if not e['gold']:continue
        gs={seq(p) for p in e['gold']}
        if length is not None and len(next(iter(gs)))!=length:continue
        if program is not None and program not in gs:continue
        # Gold-program diagnostic: policy only chooses versions, not relation syntax.
        hit=0
        for g in gs:
            p=choose([x for x in e['paths'] if seq(x)==g],policy,e['nf'])
            if p is not None and exact(p[-1][2],e['a']):hit=1;break
        ok+=hit;n+=1
    return ok/max(1,n),n

def select_policies(dev):
    global_best=max(POLICIES,key=lambda p:version_score(dev,p)[0])
    bylen={}
    for L in [1,2,3,4]:
        scores=[(version_score(dev,p,length=L)[0],p,version_score(dev,p,length=L)[1]) for p in POLICIES]
        bylen[L]=max(scores)[1]
    counts=Counter(seq(e['gold'][0]) for e in dev if e['gold'])
    byprog={}
    for g,c in counts.items():
        if c>=4:byprog[g]=max(POLICIES,key=lambda p:version_score(dev,p,program=g)[0])
    return global_best,bylen,byprog,{p:version_score(dev,p)[0] for p in POLICIES}

def evaluate(D,tr,vec,X,global_policy,bylen,byprog):
    out={};n=len(D['examples'])
    for mode in ['global','by_length','by_program']:
        ok=proghit=voracle=pathoracle=0;sims=[]
        for e in D['examples']:
            gs={seq(p) for p in e['gold']};pathoracle+=int(bool(e['gold']))
            pg,s=programs(e['tpl'],tr,vec,X);sims.append(s);proghit+=int(any(x in gs for x in pg))
            pred=None
            for g in pg:
                ps=[p for p in e['paths'] if seq(p)==g]
                if not ps:continue
                pol=global_policy
                if mode in ('by_length','by_program'):pol=bylen.get(len(g),global_policy)
                if mode=='by_program':pol=byprog.get(g,pol)
                pred=choose(ps,pol,e['nf']);
                if pred is not None:break
            ok+=int(pred is not None and exact(pred[-1][2],e['a']))
            # Version-only diagnostic with gold program; policy selection remains dev-only.
            vh=0
            for g in gs:
                pol=global_policy
                if mode in ('by_length','by_program'):pol=bylen.get(len(g),global_policy)
                if mode=='by_program':pol=byprog.get(g,pol)
                p=choose([x for x in e['paths'] if seq(x)==g],pol,e['nf'])
                if p is not None and exact(p[-1][2],e['a']):vh=1;break
            voracle+=vh
        out[mode]={'accuracy':ok/max(1,n),'program_top5_recall':proghit/max(1,n),'version_accuracy_given_gold_program':voracle/max(1,n),'K2_path_oracle':pathoracle/max(1,n),'mean_template_similarity':float(np.mean(sims))}
    return out

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    D={s:build(rows[s]) for s in srcs}
    held={norm(e['q']) for s in srcs[2:] for e in D[s]['examples']}
    dev=[e for s in srcs[:2] for e in D[s]['examples'] if norm(e['q']) not in held]
    overlap=sum(norm(e['q']) in held for s in srcs[:2] for e in D[s]['examples'])
    tr,vec,X=train_program(dev);gb,bl,bp,devpol=select_policies(dev)
    res={s:evaluate(D[s],tr,vec,X,gb,bl,bp) for s in srcs}
    out={'stage':'transaction-snapshot K2 compiler','hypothesis':'first-hop edit defines a transaction snapshot; downstream facts should be resolved as-of that serial instead of by global latest-write',
         'training_protocol':'ordered relation programs and policy choice use 6K+32K only; exact held-out question overlaps removed; 64K+262K answers only score frozen selectors',
         'dev_examples':len(dev),'exact_overlap_removed':overlap,'global_policy':gb,'policy_by_length':{str(k):v for k,v in bl.items()},'program_specific_policy_count':len(bp),'dev_version_policy_accuracy':devpol,
         'policies':POLICIES,'results':res,'state':{s:{'facts':D[s]['facts'],'registers':D[s]['registers'],'stored_edges':D[s]['edges']} for s in srcs},
         'guardrail':'K2 path oracle and gold-program version diagnostic are evaluation-only; no held-out labels are used in compiler or policy fitting.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('SNAPSHOT_COMPILER_K2='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
