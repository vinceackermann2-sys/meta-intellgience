import json,re,string,itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT=Path('researchbreakthrough/full_banking77_result.json')
PATTERNS=[
('P19','place of birth',re.compile(r'^(.+?) was born in the city of (.+?)\.$')),
('P20','place of death',re.compile(r'^(.+?) died in the city of (.+?)\.$')),
('P413','position',re.compile(r'^(.+?) plays the position of (.+?)\.$')),
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
('P407','written language',re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
('P140','religion',re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
('P106','occupation',re.compile(r'^(.+?) works in the field of (.+?)\.$')),
('P495','country of origin',re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
('P740','place founded',re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
('P740','place founded',re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
('P40','child',re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
('P40','child',re.compile(r"^(.+?)'s child is (.+?)\.$")),
('P36','capital',re.compile(r'^The capital of (.+?) is (.+?)\.$')),
('P159','headquarters',re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),
('P50','author',re.compile(r'^The author of (.+?) is (.+?)\.$')),
('P69','education',re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),
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
RELNAME={rid:name for rid,name,_ in PATTERNS};RELNAME['P1308']='officeholder'
CUES={
'P19':['birth','born'],'P20':['death','died','pass away'],'P413':['position','speciality','specialty'],
'P641':['sport'],'P30':['continent'],'P937':['work location','worked'],'P26':['spouse','partner','married'],
'P27':['citizenship','citizen','nationality'],'P175':['performer','performed'],'P108':['employer','employed'],
'P112':['founder','founded','established'],'P170':['creator','created'],'P178':['developer','developed'],
'P800':['famous for','known for','notable work'],'P1412':['speak','communicate'],'P407':['written','wrote'],
'P140':['religion','faith'],'P106':['occupation','field'],'P495':['country of origin','origin','hails','came from'],
'P740':['founded','established'],'P40':['child'],'P36':['capital'],'P159':['headquarters','hq'],
'P50':['author'],'P69':['university','school','educated','education'],'P1037':['director','manager'],
'P488':['chairperson'],'P169':['ceo','chief executive'],'P449':['broadcaster','aired'],'P176':['producer','manufacturer'],
'P136':['genre','music'],'P37':['official language','officially spoken','official documents'],'P364':['original language'],
'P35':['head of state','chief public representative'],'P6':['head of government'],'P286':['coach'],'P1308':['president','prime minister','governor','mayor','pope','officeholder']}

def parse_fact(t):
    for rid,name,p in PATTERNS:
        m=p.match(t)
        if m:return m.group(1).strip(),rid,m.group(2).strip()
    m=OFFICE.match(t)
    return (m.group(1).strip(),'P1308',m.group(2).strip()) if m else None

def norm(s):
    s=s.lower();s=''.join(c for c in s if c not in string.punctuation);s=re.sub(r'\b(a|an|the)\b',' ',s)
    return ' '.join(s.split())
def golds(a):
    if isinstance(a,str):return [a]
    if a and isinstance(a[0],list):return [x for z in a for x in z]
    return list(a or [])
def exact(x,a):return int(any(norm(x)==norm(g) for g in golds(a)))

def compile_row(row,K=2):
    reg=defaultdict(dict);nf=0
    for line in row['context'].splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if not m:continue
        z=parse_fact(m.group(2))
        if not z:continue
        nf+=1;serial=int(m.group(1));s,r,o=z;reg[(s,r)][o]=max(serial,reg[(s,r)].get(o,-1))
    vers={k:sorted(((ser,o) for o,ser in v.items()),reverse=True) for k,v in reg.items()}
    adj=defaultdict(list); latest={k:v[0][0] for k,v in vers.items()}
    for (s,r),vs in vers.items():
        for rank,(ser,o) in enumerate(vs[:K]):adj[s].append((r,o,ser,rank))
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26':adj[o].append((r,s,ser,rank))
    return adj,latest,nf,len(reg),sum(min(K,len(v)) for v in vers.values())

def anchor(q,adj):
    qf=q.casefold();c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in adj if norm(e) and norm(e) in nq]
    return max(c,key=len) if c else None

def paths(st,adj,maxh=4,cap=4000):
    out=[]
    if not st:return out
    stack=[(st,[],{st})]
    while stack and len(out)<cap:
        node,es,seen=stack.pop()
        if es:out.append(es)
        if len(es)>=maxh:continue
        for r,o,ser,rank in adj.get(node,[]):
            if o not in seen:stack.append((o,es+[(node,r,o,ser,rank)],seen|{o}))
    return out

def rel_desc(st,p):
    x=st
    for _s,r,_o,_ser,_rank in p:x=f'the {RELNAME.get(r,r)} of {x}'
    return x

def full_desc(st,p):
    bits=[st]
    for s,r,o,ser,rank in p:bits.append(f'{s} --{RELNAME.get(r,r)}--> {o}')
    return '. '.join(bits)

def cue(q,p):
    q=q.casefold();rels=[e[1] for e in p]
    hits=sum(any(t in q for t in CUES.get(r,[RELNAME.get(r,r)])) for r in rels)
    return hits/max(1,len(rels))

def row_candidates(row,enc,K=2):
    adj,latest,nf,nreg,nedge=compile_row(row,K);qs=list(row['questions']);ans=list(row['answers'])
    qemb=enc.encode(qs,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    data=[];anchor_ok=[]
    for i,(q,a) in enumerate(zip(qs,ans)):
        st=anchor(q,adj);anchor_ok.append(st is not None);ps=paths(st,adj)
        if not ps:data.append({'q':q,'gold':a,'paths':[]});continue
        rd=[rel_desc(st,p) for p in ps];fd=[full_desc(st,p) for p in ps]
        R=enc.encode(rd,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        F=enc.encode(fd,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        items=[]
        for j,p in enumerate(ps):
            ranks=[e[4] for e in p]
            items.append({'end':p[-1][2],'rel_sem':float(R[j]@qemb[i]),'full_sem':float(F[j]@qemb[i]),
                          'cue':cue(q,p),'newest_frac':1-sum(ranks)/len(ranks),'first_newest':1-ranks[0],
                          'last_newest':1-ranks[-1],'len':len(p),'gold':exact(p[-1][2],a)})
        data.append({'q':q,'gold':a,'paths':items})
    return {'source':(row.get('metadata') or {}).get('source',''),'facts':nf,'registers':nreg,'stored_edges':nedge,
            'anchor_coverage':sum(anchor_ok)/len(anchor_ok),'questions':data}

def accuracy(data,w):
    ok=0;preds=[]
    a,b,c,d,e,f=w
    for z in data:
        if not z['paths']:preds.append(None);continue
        def score(p):return p['rel_sem']+a*p['full_sem']+b*p['cue']+c*p['newest_frac']+d*p['first_newest']+e*p['last_newest']-f*(p['len']-1)
        p=max(z['paths'],key=score);ok+=p['gold'];preds.append(p['end'])
    return ok/len(data),preds

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    rows={ (r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    D={s:row_candidates(rows[s],enc,2) for s in srcs}
    # Exact normalized question overlap audit.  Any held-out question is removed from dev if duplicated.
    held=set(norm(z['q']) for s in srcs[2:] for z in D[s]['questions'])
    dev=[z for s in srcs[:2] for z in D[s]['questions'] if norm(z['q']) not in held]
    overlap=sum(norm(z['q']) in held for s in srcs[:2] for z in D[s]['questions'])
    grid=[];best=(-1,None)
    for w in itertools.product([0,.25,.5,1.0],[0,.1,.2],[ -.15,-.05,0,.05,.15],[0,.03],[0,.03],[0,.01]):
        ac,_=accuracy(dev,w)
        if ac>best[0]:best=(ac,w)
    results={}
    for s in srcs:
        ac,_=accuracy(D[s]['questions'],best[1]);results[s]={'accuracy':ac,'stored_edges':D[s]['stored_edges'],'facts':D[s]['facts'],'anchor_coverage':D[s]['anchor_coverage']}
    # Untuned transparent controls on held-out lanes.
    controls={}
    for name,w in {'relation_semantic_only':(0,0,0,0,0,0),'relation_plus_full':(.5,0,0,0,0,0),'prefer_newest':(.5,.1,.15,.03,.03,.01),'prefer_shadow':(.5,.1,-.10,0,0,.01)}.items():
        controls[name]={s:accuracy(D[s]['questions'],w)[0] for s in srcs}
    out={'stage':'answer-blind K=2 dual-view path planner','training_protocol':'weights selected on 6K+32K only; exact-overlap dev questions removed; 64K+262K held out','question_overlap_removed_from_dev':overlap,
         'dev_questions':len(dev),'selected_dev_accuracy':best[0],'selected_weights':best[1],'results':results,'controls':controls,
         'note':'No 64K/262K answers used for weight selection. K=2 preserves current and one shadow value per register.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('DUAL_VIEW_PLANNER='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
