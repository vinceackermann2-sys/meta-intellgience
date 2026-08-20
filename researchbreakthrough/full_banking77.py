import json,re,string,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

OUT=Path('researchbreakthrough/full_banking77_result.json')

PATTERNS=[
('P19','place of birth',re.compile(r'^(.+?) was born in the city of (.+?)\.$')),
('P20','place of death',re.compile(r'^(.+?) died in the city of (.+?)\.$')),
('P413','position played on team',re.compile(r'^(.+?) plays the position of (.+?)\.$')),
('P641','sport',re.compile(r'^(.+?) is associated with the sport of (.+?)\.$')),
('P30','continent',re.compile(r'^(.+?) is located in the continent of (.+?)\.$')),
('P937','work location',re.compile(r'^(.+?) worked in the city of (.+?)\.$')),
('P26','spouse',re.compile(r'^(.+?) is married to (.+?)\.$')),
('P27','country of citizenship',re.compile(r'^(.+?) is a citizen of (.+?)\.$')),
('P175','performer',re.compile(r'^(.+?) was performed by (.+?)\.$')),
('P108','employer',re.compile(r'^(.+?) is employed by (.+?)\.$')),
('P112','founder',re.compile(r'^(.+?) was founded by (.+?)\.$')),
('P170','creator',re.compile(r'^(.+?) was created by (.+?)\.$')),
('P178','developer',re.compile(r'^(.+?) was developed by (.+?)\.$')),
('P800','notable work',re.compile(r'^(.+?) is famous for (.+?)\.$')),
('P1412','spoken language',re.compile(r'^(.+?) speaks the language of (.+?)\.$')),
('P407','language of work',re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
('P140','religion',re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
('P106','occupation',re.compile(r'^(.+?) works in the field of (.+?)\.$')),
('P495','country of origin',re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
('P740','place founded',re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
('P740','place founded',re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
('P40','child',re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
('P40','child',re.compile(r"^(.+?)'s child is (.+?)\.$")),
('P36','capital',re.compile(r'^The capital of (.+?) is (.+?)\.$')),
('P159','headquarters location',re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),
('P50','author',re.compile(r'^The author of (.+?) is (.+?)\.$')),
('P69','educational institution',re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),
('P1037','director or manager',re.compile(r'^The director of (.+?) is (.+?)\.$')),
('P488','chairperson',re.compile(r'^The chairperson of (.+?) is (.+?)\.$')),
('P169','chief executive officer',re.compile(r'^The chief executive officer of (.+?) is (.+?)\.$')),
('P449','original broadcaster',re.compile(r'^The (?:origianl|original) broadcaster of (.+?) is (.+?)\.$')),
('P176','producer',re.compile(r'^The company that produced (.+?) is (.+?)\.$')),
('P136','music genre',re.compile(r'^The type of music that (.+?) plays is (.+?)\.$')),
('P37','official language',re.compile(r'^The official language of (.+?) is (.+?)\.$')),
('P364','original language',re.compile(r'^The original language of (.+?) is (.+?)\.$')),
('P35','head of state',re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
('P35','head of state',re.compile(r'^The name of the current head of state in (.+?) is (.+?)\.$')),
('P6','head of government',re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
('P286','coach',re.compile(r'^The coach of (.+?) is (.+?)\.$')),
('P286','coach',re.compile(r'^The head coach of (.+?) is (.+?)\.$')),
]
OFFICE=re.compile(r'^The (.+?) is (.+?)\.$')
RELNAME={rid:name for rid,name,_ in PATTERNS}; RELNAME['P1308']='officeholder'

def parse_fact(t):
    for rid,name,p in PATTERNS:
        m=p.match(t)
        if m:return m.group(1).strip(),rid,m.group(2).strip()
    m=OFFICE.match(t)
    return (m.group(1).strip(),'P1308',m.group(2).strip()) if m else None

def norm(s):
    s=str(s).lower(); s=''.join(c for c in s if c not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())

def golds(a):
    if isinstance(a,str): return [a]
    if isinstance(a,list) and a and isinstance(a[0],list): return [x for z in a for x in z]
    return list(a or [])

def exact(x,a): return any(norm(x)==norm(g) for g in golds(a))

def compile_row(row,K=2):
    reg=defaultdict(dict); nf=0
    for line in row['context'].splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if not m: continue
        z=parse_fact(m.group(2))
        if not z: continue
        nf+=1; ser=int(m.group(1)); s,r,o=z
        reg[(s,r)][o]=max(ser,reg[(s,r)].get(o,-1))
    vers={k:sorted(((ser,o) for o,ser in v.items()),reverse=True) for k,v in reg.items()}
    adj=defaultdict(list)
    for (s,r),vs in vers.items():
        for rank,(ser,o) in enumerate(vs[:K]): adj[s].append((r,o,ser,rank))
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26': adj[o].append((r,s,ser,rank))
    return adj,nf,len(reg),sum(min(K,len(v)) for v in vers.values())

def anchor(q,adj):
    qf=q.casefold(); c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q); c=[e for e in adj if norm(e) and norm(e) in nq]
    return max(c,key=len) if c else None

def template(q,st):
    s=q.casefold()
    if st: s=re.sub(re.escape(st.casefold()),'<ent>',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s

def paths(st,adj,maxh=4,cap=50000):
    if not st:return []
    out=[]; stack=[(st,[],{st})]
    while stack and len(out)<cap:
        node,es,seen=stack.pop()
        if es: out.append(es)
        if len(es)>=maxh: continue
        for r,o,ser,rank in adj.get(node,[]):
            if o not in seen: stack.append((o,es+[(node,r,o,ser,rank)],seen|{o}))
    return out

def seq(p): return tuple(e[1] for e in p)
def ranks(p): return tuple(e[4] for e in p)

def path_features(p,nf):
    ss=np.array([e[3] for e in p],dtype=float); rr=np.array([e[4] for e in p],dtype=float)
    gaps=np.abs(np.diff(ss)) if len(ss)>1 else np.array([0.0])
    return [
        len(p), rr.sum(), rr.mean(), 1.0-rr.mean(), rr[0], rr[-1],
        ss.min()/max(1,nf), ss.max()/max(1,nf), ss.mean()/max(1,nf), ss.std()/max(1,nf),
        (ss.max()-ss.min())/max(1,nf), gaps.mean()/max(1,nf), gaps.max()/max(1,nf),
        float(np.all(np.diff(ss)>=0)) if len(ss)>1 else 1.0,
        float(np.all(np.diff(ss)<=0)) if len(ss)>1 else 1.0,
    ]

def build_examples(row):
    adj,nf,nreg,nedge=compile_row(row,2); ex=[]
    for q,a in zip(row['questions'],row['answers']):
        st=anchor(q,adj); ps=paths(st,adj); gp=[p for p in ps if exact(p[-1][2],a)]
        gp.sort(key=lambda p:(len(p),sum(e[4] for e in p),max(e[3] for e in p)-min(e[3] for e in p)))
        ex.append({'q':q,'a':a,'st':st,'tpl':template(q,st),'paths':ps,'gold_paths':gp,'nf':nf})
    return {'source':(row.get('metadata') or {}).get('source',''),'facts':nf,'registers':nreg,'stored_edges':nedge,'examples':ex}

def label_of(p): return '|'.join(seq(p))+'#'+''.join(map(str,ranks(p)))
def seq_label(p): return '|'.join(seq(p))

def parse_label(z):
    a,b=z.split('#'); return tuple(a.split('|')),tuple(int(x) for x in b)

def candidate_for_label(paths_,lab):
    ss,rr=parse_label(lab)
    exacts=[p for p in paths_ if seq(p)==ss and ranks(p)==rr]
    if exacts:return exacts
    # Preserve program if exact version pattern is absent; rank by Hamming distance to requested pattern.
    same=[p for p in paths_ if seq(p)==ss]
    if same:
        return sorted(same,key=lambda p:(sum(abs(a-b) for a,b in zip(ranks(p),rr)),sum(ranks(p))))
    return []

def relation_oracle(ex):
    gs={seq(p) for p in ex['gold_paths']}
    return gs

def train_program_compiler(dev):
    train=[]
    for e in dev:
        if not e['gold_paths']: continue
        # Multiple gold paths are possible; shortest/newest representative is deterministic.
        p=e['gold_paths'][0]
        train.append((e['tpl'],seq_label(p),label_of(p)))
    texts=[x[0] for x in train]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,6),min_df=1,sublinear_tf=True)
    X=vec.fit_transform(texts)
    return train,vec,X

def predict_labels(tpl,train,vec,X,k=25):
    qv=vec.transform([tpl]); sims=cosine_similarity(qv,X).ravel(); idx=np.argsort(-sims)[:min(k,len(sims))]
    # First vote relation program, then version pattern inside the winning programs.
    prog=defaultdict(float); full=defaultdict(float)
    for rank_i,i in enumerate(idx):
        w=float(max(0,sims[i]))**3 + 1e-6/(rank_i+1)
        prog[train[i][1]]+=w; full[train[i][2]]+=w
    top_prog=[x for x,_ in sorted(prog.items(),key=lambda kv:kv[1],reverse=True)[:5]]
    labs=[]
    for z,_ in sorted(full.items(),key=lambda kv:kv[1],reverse=True):
        if z.split('#')[0] in top_prog: labs.append(z)
        if len(labs)>=12:break
    return labs,top_prog,float(sims[idx[0]]) if len(idx) else 0.0

def train_structural_ranker(dev):
    X=[];y=[]
    for e in dev:
        if not e['paths'] or not e['gold_paths']: continue
        goldset={(seq(p),ranks(p),p[-1][2]) for p in e['gold_paths']}
        goldseq={seq(p) for p in e['gold_paths']}
        # Train version chooser only on paths having a gold relation program.
        for p in e['paths']:
            if seq(p) not in goldseq: continue
            X.append(path_features(p,e['nf']))
            y.append(int((seq(p),ranks(p),p[-1][2]) in goldset))
    if len(set(y))<2:return None
    clf=RandomForestClassifier(n_estimators=400,max_depth=7,min_samples_leaf=2,class_weight='balanced_subsample',random_state=7,n_jobs=-1)
    clf.fit(np.asarray(X),np.asarray(y))
    return clf

def choose_structural(paths_,program,nf,clf):
    ps=[p for p in paths_ if seq(p)==program]
    if not ps:return None
    if clf is None:return min(ps,key=lambda p:(sum(ranks(p)),max(e[3] for e in p)-min(e[3] for e in p)))
    pr=clf.predict_proba(np.asarray([path_features(p,nf) for p in ps]))[:,1]
    return ps[int(np.argmax(pr))]

def evaluate(data,train,vec,X,ranker):
    ok=0; relation_ok=0; version_given_goldprog=0; compiler_prog_ok=0; n=0; details=[]
    for e in data['examples']:
        n+=1; goldseq=relation_oracle(e)
        if goldseq: relation_ok+=1
        labs,progs,sim=predict_labels(e['tpl'],train,vec,X)
        pred_prog=[tuple(z.split('|')) for z in progs]
        prog_hit=any(p in goldseq for p in pred_prog); compiler_prog_ok+=int(prog_hit)
        # Main prediction: full template label vote; first executable label wins.
        pred=None
        for lab in labs:
            ps=candidate_for_label(e['paths'],lab)
            if ps:
                # Among duplicates use structural selector for that program, but exact requested rank pattern gets priority.
                ss,rr=parse_label(lab); exactps=[p for p in ps if ranks(p)==rr]
                pred=exactps[0] if exactps else choose_structural(ps,ss,e['nf'],ranker)
                if pred:break
        # Fallback: top predicted program + structural version chooser.
        if pred is None:
            for pg in pred_prog:
                pred=choose_structural(e['paths'],pg,e['nf'],ranker)
                if pred:break
        if pred is None and e['paths']:
            pred=min(e['paths'],key=lambda p:(len(p),sum(ranks(p))))
        hit=int(pred is not None and exact(pred[-1][2],e['a'])); ok+=hit
        # Diagnostic ceiling: assume correct program known, ask structural ranker to choose version.
        vh=0
        for pg in goldseq:
            p=choose_structural(e['paths'],pg,e['nf'],ranker)
            if p is not None and exact(p[-1][2],e['a']): vh=1;break
        version_given_goldprog+=vh
        details.append({'hit':hit,'program_hit':int(prog_hit),'version_hit_given_gold_program':vh,'template_similarity':sim})
    return {'accuracy':ok/max(1,n),'program_top5_recall':compiler_prog_ok/max(1,n),
            'K2_path_oracle':relation_ok/max(1,n),'version_accuracy_given_gold_program':version_given_goldprog/max(1,n),
            'mean_nearest_template_similarity':float(np.mean([d['template_similarity'] for d in details]))}

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    D={s:build_examples(rows[s]) for s in srcs}
    held_exact={norm(e['q']) for s in srcs[2:] for e in D[s]['examples']}
    dev=[e for s in srcs[:2] for e in D[s]['examples'] if norm(e['q']) not in held_exact]
    overlap=sum(norm(e['q']) in held_exact for s in srcs[:2] for e in D[s]['examples'])
    train,vec,X=train_program_compiler(dev); ranker=train_structural_ranker(dev)
    results={s:evaluate(D[s],train,vec,X,ranker) for s in srcs}
    # Template coverage diagnostics: no answers involved.
    dev_tpl={e['tpl'] for e in dev}
    tpl_cov={s:sum(e['tpl'] in dev_tpl for e in D[s]['examples'])/len(D[s]['examples']) for s in srcs}
    out={'stage':'template-supervised ordered relation-program compiler + structural K2 version chooser',
         'training_protocol':'program/version labels learned only from 6K+32K gold paths; exact held-out question overlaps removed; 64K+262K answers used only after predictions for evaluation',
         'dev_examples':len(dev),'heldout_exact_overlap_removed':overlap,'training_labels':len(train),
         'architecture':{'canonical_state':'K=2 current+shadow registers','query':'anchor -> normalized question grammar -> ordered relation program -> local version execution',
                         'program_model':'character n-gram TF-IDF nearest-template voting','version_model':'random-forest structural chooser trained only within gold programs on dev'},
         'template_exact_coverage':tpl_cov,'results':results,
         'guardrail':'K2_path_oracle and version_accuracy_given_gold_program are diagnostics only; held-out gold never enters compiler or ranker fitting.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)); print('PROGRAM_COMPILER_K2='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
