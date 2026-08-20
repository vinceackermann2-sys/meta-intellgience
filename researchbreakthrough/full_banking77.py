import json,re,string,itertools,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

OUT=Path('researchbreakthrough/full_banking77_result.json')

# Published MQuAKE relation inventory + the surface forms used by MemoryAgentBench.
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
RELNAME={rid:name for rid,name,_ in PATTERNS};RELNAME['P1308']='officeholder'
CUES={
'P19':['birth','born'],'P20':['death','died','pass away'],'P413':['position','speciality','specialty'],
'P641':['sport'],'P30':['continent'],'P937':['worked','work location'],'P26':['spouse','partner','married'],
'P27':['citizenship','citizen','nationality'],'P175':['performer','performed'],'P108':['employer','employed'],
'P112':['founder','founded','established'],'P170':['creator','created'],'P178':['developer','developed'],
'P800':['famous for','known for','notable work'],'P1412':['speak','communicate'],'P407':['written','wrote'],
'P140':['religion','faith'],'P106':['occupation','field'],'P495':['country of origin','origin','hail','came from'],
'P740':['founded','established'],'P40':['child'],'P36':['capital'],'P159':['headquarters'],
'P50':['author'],'P69':['university','school','educated','education'],'P1037':['director','manager'],
'P488':['chairperson'],'P169':['ceo','chief executive'],'P449':['broadcaster','aired'],'P176':['producer','manufacturer'],
'P136':['genre','music'],'P37':['official language','officially spoken','official documents'],'P364':['original language'],
'P35':['head of state','chief public representative'],'P6':['head of government'],'P286':['coach'],
'P1308':['president','prime minister','governor','mayor','pope','officeholder']}

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
    if isinstance(a,list) and a and isinstance(a[0],list):return [x for z in a for x in z]
    return list(a or [])
def exact(x,a):return int(any(norm(x)==norm(g) for g in golds(a)))

def compile_row(row,K=2):
    reg=defaultdict(dict);nf=0
    for line in row['context'].splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if not m:continue
        z=parse_fact(m.group(2))
        if not z:continue
        nf+=1;ser=int(m.group(1));s,r,o=z;reg[(s,r)][o]=max(ser,reg[(s,r)].get(o,-1))
    vers={k:sorted(((ser,o) for o,ser in v.items()),reverse=True) for k,v in reg.items()}
    adj=defaultdict(list)
    for (s,r),vs in vers.items():
        for rank,(ser,o) in enumerate(vs[:K]):adj[s].append((r,o,ser,rank))
    # P26 is symmetric; inverse is a derived index and costs no extra stored fact.
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26':adj[o].append((r,s,ser,rank))
    return adj,nf,len(reg),sum(min(K,len(v)) for v in vers.values())

def anchor(q,adj):
    qf=q.casefold();c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in adj if norm(e) and norm(e) in nq]
    return max(c,key=len) if c else None

def paths(st,adj,maxh=4,cap=12000):
    if not st:return []
    out=[];stack=[(st,[],{st})]
    while stack and len(out)<cap:
        node,es,seen=stack.pop()
        if es:out.append(es)
        if len(es)>=maxh:continue
        for r,o,ser,rank in adj.get(node,[]):
            if o not in seen:stack.append((o,es+[(node,r,o,ser,rank)],seen|{o}))
    return out

def relation_text(st,p):
    x=st
    for _s,r,_o,_ser,_rank in p:x=f'the {RELNAME.get(r,r)} of {x}'
    return x

def chain_text(st,p):
    return '; '.join([f'start entity: {st}']+[f'{s} -- {RELNAME.get(r,r)} --> {o}' for s,r,o,ser,rank in p])

def cue_score(q,p):
    q=q.casefold();rels=[e[1] for e in p]
    return sum(any(t in q for t in CUES.get(r,[RELNAME.get(r,r)])) for r in rels)/max(1,len(rels))

def zscore(x):
    x=np.asarray(x,dtype=np.float32)
    if len(x)<2:return np.zeros_like(x)
    return (x-x.mean())/(x.std()+1e-6)

def generate(row,bi,cross,top_sequences=8):
    adj,nf,nreg,nedge=compile_row(row,2);qs=list(row['questions']);ans=list(row['answers'])
    qemb=bi.encode(qs,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    output=[];anchor_ok=[]
    for qi,(q,a) in enumerate(zip(qs,ans)):
        st=anchor(q,adj);anchor_ok.append(st is not None);ps=paths(st,adj)
        if not ps:output.append({'q':q,'gold':a,'cands':[],'all_oracle':0,'prefilter_oracle':0});continue
        # Rank UNIQUE relation programs first, then keep every K=2 version branch for the best programs.
        byseq=defaultdict(list)
        for p in ps:byseq[tuple(e[1] for e in p)].append(p)
        seqs=list(byseq);seqtxt=[]
        for seq in seqs:
            x=st
            for r in seq:x=f'the {RELNAME.get(r,r)} of {x}'
            seqtxt.append(x)
        S=bi.encode(seqtxt,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)@qemb[qi]
        scored=[]
        for j,seq in enumerate(seqs):
            p0=byseq[seq][0];cheap=float(S[j])+0.16*cue_score(q,p0)-0.004*(len(seq)-1)
            scored.append((cheap,seq))
        chosen=set(seq for _,seq in sorted(scored,reverse=True)[:top_sequences])
        candpaths=[p for seq in chosen for p in byseq[seq]]
        # Hard cap after preserving relation-program diversity.  Cheap per-path recency only breaks huge branch explosions.
        if len(candpaths)>160:
            candpaths=sorted(candpaths,key=lambda p:(cue_score(q,p)+0.03*(1-sum(e[4] for e in p)/len(p))),reverse=True)[:160]
        pairtexts=[(q,chain_text(st,p)) for p in candpaths]
        cs=np.asarray(cross.predict(pairtexts,batch_size=64,show_progress_bar=False),dtype=np.float32).reshape(-1)
        cz=zscore(cs)
        rt=[relation_text(st,p) for p in candpaths]
        re=bi.encode(rt,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        sem=re@qemb[qi];sz=zscore(sem)
        items=[]
        for j,p in enumerate(candpaths):
            ranks=[e[4] for e in p];serials=[e[3] for e in p]
            items.append({'end':p[-1][2],'cross_z':float(cz[j]),'relation_z':float(sz[j]),'cue':cue_score(q,p),
                          'newest_frac':1-sum(ranks)/len(ranks),'first_newest':1-ranks[0],'last_newest':1-ranks[-1],
                          'shadow_count':sum(ranks),'len':len(p),'serial_span':(max(serials)-min(serials))/max(1,nf),
                          'gold':exact(p[-1][2],a)})
        output.append({'q':q,'gold':a,'cands':items,'all_oracle':int(any(exact(p[-1][2],a) for p in ps)),
                       'prefilter_oracle':int(any(x['gold'] for x in items))})
    return {'source':(row.get('metadata') or {}).get('source',''),'facts':nf,'registers':nreg,'stored_edges':nedge,
            'anchor_coverage':sum(anchor_ok)/len(anchor_ok),'questions':output}

def score_item(p,w):
    # cross, relation, cue, newest_frac, first_newest, last_newest, shadow_count, len_penalty, serial_span
    return (w[0]*p['cross_z']+w[1]*p['relation_z']+w[2]*p['cue']+w[3]*p['newest_frac']+
            w[4]*p['first_newest']+w[5]*p['last_newest']+w[6]*p['shadow_count']-
            w[7]*(p['len']-1)+w[8]*p['serial_span'])

def accuracy(rows,w):
    ok=0;n=0
    for z in rows:
        n+=1
        if not z['cands']:continue
        p=max(z['cands'],key=lambda x:score_item(x,w));ok+=p['gold']
    return ok/max(1,n)

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    bi=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    cross=CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2',device='cpu',max_length=256)
    D={s:generate(rows[s],bi,cross,8) for s in srcs}
    held=set(norm(z['q']) for s in srcs[2:] for z in D[s]['questions'])
    dev=[z for s in srcs[:2] for z in D[s]['questions'] if norm(z['q']) not in held]
    overlap=sum(norm(z['q']) in held for s in srcs[:2] for z in D[s]['questions'])

    # Coarse search, then refine around the best answer-blind feature mixture. Held-out labels never enter selection.
    best=(-1,None)
    for w in itertools.product([.5,1,1.5],[0,.25,.5],[0,.1,.2],[-.1,0,.1,.2],[0,.05],[0,.05],[-.08,-.03,0],[0,.01],[0]):
        a=accuracy(dev,w)
        if a>best[0]:best=(a,w)
    controls={
      'cross_only':(1,0,0,0,0,0,0,0,0),
      'cross_plus_relation':(1,.25,.1,0,0,0,0,.01,0),
      'cross_prefer_current':(1,.25,.1,.15,.05,.05,0,.01,0),
      'cross_allow_shadow':(1,.25,.1,.05,0,0,-.03,.01,0),
    }
    result={}
    for s in srcs:
        q=D[s]['questions'];result[s]={
          'accuracy_selected':accuracy(q,best[1]),'all_K2_path_oracle':float(np.mean([z['all_oracle'] for z in q])),
          'prefilter_path_oracle':float(np.mean([z['prefilter_oracle'] for z in q])),
          'stored_edges':D[s]['stored_edges'],'history_facts':D[s]['facts'],'anchor_coverage':D[s]['anchor_coverage'],
          'controls':{k:accuracy(q,w) for k,w in controls.items()}}
    out={'stage':'held-out K2 cross-encoder version/path selector','selector':'ms-marco-MiniLM-L6-v2 over explicit typed path + MiniLM relation-program score + structural version features',
         'training_protocol':'weights selected only on 6K+32K; exact-overlap questions with 64K/262K removed; 64K+262K answers never used for selection',
         'question_overlap_removed_from_dev':overlap,'dev_questions':len(dev),'selected_dev_accuracy':best[0],
         'selected_weights':list(best[1]),'top_relation_programs':8,'results':result,
         'scientific_guardrail':'gold is used only after candidate generation for evaluation/oracle reporting; it is not used to select held-out paths or weights'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('CROSS_ENCODER_K2='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
