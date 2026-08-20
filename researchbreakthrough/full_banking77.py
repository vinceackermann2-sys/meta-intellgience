import json,re,string,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
RELNAME={r:n for r,n,_ in PATTERNS};RELNAME['P1308']='officeholder'
# Fixed relation cues used before held-out question inspection; no 64K/262K answer or phrase-specific additions.
CUES={
'P19':['place of birth','birthplace','born'],'P20':['place of death','death','died','pass away'],'P413':['position played on team','position','speciality','specialty'],
'P641':['sport'],'P30':['continent'],'P937':['work location','place of work','worked'],'P26':['spouse','partner','married'],
'P27':['country of citizenship','citizenship','citizen','nationality'],'P175':['performer','performed'],'P108':['employer','employed'],
'P112':['founder','founded','established'],'P170':['creator','created'],'P178':['developer','developed'],'P800':['notable work','famous for','known for'],
'P1412':['spoken language','speak','communicate'],'P407':['language of work','written','wrote'],'P140':['religion','faith'],'P106':['occupation','field'],
'P495':['country of origin','origin','hail','came from'],'P740':['place founded','founded','established'],'P40':['child'],'P36':['capital'],
'P159':['headquarters location','headquarters'],'P50':['author'],'P69':['educational institution','university','school','educated','education'],
'P1037':['director or manager','director','manager'],'P488':['chairperson'],'P169':['chief executive officer','ceo','chief executive'],
'P449':['original broadcaster','broadcaster','aired'],'P176':['producer','manufacturer'],'P136':['music genre','genre','music'],
'P37':['official language','officially spoken','official documents'],'P364':['original language'],'P35':['head of state','chief public representative'],
'P6':['head of government'],'P286':['coach'],'P1308':['president','prime minister','governor','mayor','pope','officeholder']}

def parse_fact(t):
    for rid,name,p in PATTERNS:
        m=p.match(t)
        if m:return m.group(1).strip(),rid,m.group(2).strip()
    m=OFFICE.match(t);return (m.group(1).strip(),'P1308',m.group(2).strip()) if m else None

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
        for rank,(ser,o) in enumerate(sorted(((ser,o) for o,ser in vals.items()),reverse=True)[:K]):adj[s].append((r,o,ser,rank))
    for s,es in list(adj.items()):
        for r,o,ser,rank in list(es):
            if r=='P26':adj[o].append((r,s,ser,rank))
    return adj,nf,len(reg),sum(min(K,len(v)) for v in reg.values())

def anchor(q,adj):
    qf=q.casefold();c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in adj if norm(e) and norm(e) in nq];return max(c,key=len) if c else None

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

def qmask(q,st):
    x=q.casefold()
    if st:x=re.sub(re.escape(st.casefold()),'<entity>',x)
    return re.sub(r'\s+',' ',x).strip()

def canonical_nested(s):
    x='<entity>'
    for r in s:x=f'the {RELNAME.get(r,r)} of {x}'
    return 'what is '+x+'?'
def canonical_steps(s):return 'starting from the entity, follow these relations in order: '+', then '.join(RELNAME.get(r,r) for r in s)+'.'

def alias_hits(q,r):
    q=q.casefold();hits=[]
    for a in CUES.get(r,[RELNAME.get(r,r)]):
        start=0
        while True:
            i=q.find(a,start)
            if i<0:break
            hits.append((i,len(a.split()),len(a)));start=i+1
    return hits

def lexical_features(q,s):
    # Text usually states the outer/answer relation before nested inner relations, so execution order is approximately reverse textual cue order.
    pos=[];covered=0;strength=0.0
    for r in s:
        hs=alias_hits(q,r)
        if hs:
            covered+=1;best=max(hs,key=lambda z:(z[1],z[2]));pos.append(best[0]);strength+=min(1.0,best[1]/3.0)
        else:pos.append(None)
    known=[(i,p) for i,p in enumerate(pos) if p is not None]
    inv_score=0.5
    if len(known)>=2:
        good=tot=0
        for a in range(len(known)):
            for b in range(a+1,len(known)):
                # relation a executes before b; we expect its cue later in text than b's cue.
                tot+=1;good+=int(known[a][1]>=known[b][1])
        inv_score=good/max(1,tot)
    # Last relation is the requested answer operator; reward its cue near question edges.
    last_hits=alias_hits(q,s[-1]);edge=0.0
    if last_hits:
        L=max(1,len(q));edge=max(max(1-h[0]/L,h[0]/L) for h in last_hits)
    return covered/len(s),strength/len(s),inv_score,edge

def target_zones(q):
    toks=re.findall(r"[\w'-]+",q.casefold())
    head=' '.join(toks[:14]);tail=' '.join(toks[-14:])
    return head,tail

def type_map(adj):
    typ=defaultdict(lambda:defaultdict(float))
    roles={
      'P19':('person','location'),'P20':('person','location'),'P413':('person','position'),'P641':('entity','sport'),'P30':('country','continent'),
      'P937':('person','location'),'P26':('person','person'),'P27':('person','country'),'P175':('work','person'),'P108':('person','organization'),
      'P112':('entity','person'),'P170':('work','person'),'P178':('product','organization'),'P800':('person','work'),'P1412':('person','language'),
      'P407':('work','language'),'P140':('person','religion'),'P106':('person','occupation'),'P495':('entity','country'),'P740':('organization','location'),
      'P40':('person','person'),'P36':('country','location'),'P159':('organization','location'),'P50':('work','person'),'P69':('person','organization'),
      'P1037':('entity','person'),'P488':('organization','person'),'P169':('organization','person'),'P449':('work','organization'),'P176':('product','organization'),
      'P136':('entity','genre'),'P37':('country','language'),'P364':('work','language'),'P35':('country','person'),'P6':('country','person'),
      'P286':('entity','person'),'P1308':('office','person')}
    for s,es in adj.items():
        for r,o,ser,rank in es:
            a,b=roles.get(r,('entity','entity'));typ[s][a]+=1;typ[o][b]+=1
    return typ,roles

def coherence_for_path(p,typ,roles):
    vals=[]
    for s,r,o,ser,rank in p:
        a,b=roles.get(r,('entity','entity'))
        os=sum(typ[o].values())+1e-6;ss=sum(typ[s].values())+1e-6
        vals.append(0.5*(typ[o].get(b,0)/os)+0.5*(typ[s].get(a,0)/ss))
    return float(np.mean(vals)) if vals else 0.0

def choose_path(ps,typ,roles,coh_weight=0.0):
    if not ps:return None
    def score(p):
        shadow=sum(e[4] for e in p);span=(max(e[3] for e in p)-min(e[3] for e in p)) if len(p)>1 else 0
        coh=coherence_for_path(p,typ,roles)
        return -shadow-0.00001*span+coh_weight*coh
    return max(ps,key=score)

def build(row,model):
    adj,nf,nreg,nedge=compile_row(row,2);typ,roles=type_map(adj);E=[]
    rel_ids=list(RELNAME);rel_text=[RELNAME[r]+'. '+', '.join(CUES.get(r,[])) for r in rel_ids]
    rel_emb=model.encode(rel_text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    for q,a in zip(row['questions'],row['answers']):
        st=anchor(q,adj);ps=paths(st,adj);by=defaultdict(list)
        for p in ps:by[seq(p)].append(p)
        programs=list(by);qm=qmask(q,st);head,tail=target_zones(qm)
        texts=[]
        for s in programs:texts.extend([canonical_nested(s),canonical_steps(s)])
        pe=model.encode(texts,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if texts else np.zeros((0,384),dtype=np.float32)
        qe=model.encode([qm,head,tail],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        feats=[]
        for j,s in enumerate(programs):
            cov,strength,order,edge=lexical_features(qm,s)
            nested=float(pe[2*j]@qe[0]);steps=float(pe[2*j+1]@qe[0])
            li=rel_ids.index(s[-1]);target_sem=max(float(rel_emb[li]@qe[1]),float(rel_emb[li]@qe[2]))
            # relation-set semantic coverage: each relation gets best of whole/head/tail similarity.
            rsem=float(np.mean([max(float(rel_emb[rel_ids.index(r)]@qe[k]) for k in range(3)) for r in s]))
            cur=max((coherence_for_path(p,typ,roles) for p in by[s]),default=0.0)
            feats.append([nested,steps,cov,strength,order,edge,target_sem,rsem,len(s),cur])
        gp={seq(p) for p in ps if exact(p[-1][2],a)}
        E.append({'q':q,'a':a,'st':st,'programs':programs,'paths_by':by,'features':feats,'gold_programs':gp,'nf':nf,'typ':typ,'roles':roles})
    return {'facts':nf,'registers':nreg,'stored_edges':nedge,'examples':E}

def prefilter(e,n=12):
    if not e['programs']:return []
    # Fixed semantic proposal score; the previous experiment showed broad semantic prefilters can retain the gold program.
    vals=[]
    for i,f in enumerate(e['features']):
        nested,steps,cov,strength,order,edge,target_sem,rsem,L,cur=f
        v=0.55*nested+0.2*steps+0.12*cov+0.06*order+0.05*target_sem+0.02*rsem-0.012*(L-1)
        vals.append((v,i))
    return [i for _,i in sorted(vals,reverse=True)[:n]]

def train_ranker(dev):
    X=[];y=[]
    for e in dev:
        idx=prefilter(e,20)
        for i in idx:
            X.append(e['features'][i]);y.append(int(e['programs'][i] in e['gold_programs']))
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=3000,random_state=7))
    clf.fit(np.asarray(X),np.asarray(y));return clf

def tune_coherence(dev):
    best=(0,None)
    for w in [0,0.25,0.5,1,2,4]:
        ok=n=0
        for e in dev:
            for g in e['gold_programs']:
                p=choose_path(e['paths_by'].get(g,[]),e['typ'],e['roles'],w);n+=1
                if p is not None and exact(p[-1][2],e['a']):ok+=1
                break
        a=ok/max(1,n)
        if a>best[0]:best=(a,w)
    return best

def evaluate(D,clf,coh_w):
    ok=0;top1prog=0;top5prog=0;pref_oracle=0;path_oracle=0;version=0;n=0
    for e in D['examples']:
        n+=1;path_oracle+=int(bool(e['gold_programs']))
        idx=prefilter(e,12);pref_oracle+=int(any(e['programs'][i] in e['gold_programs'] for i in idx))
        if not idx:continue
        probs=clf.predict_proba(np.asarray([e['features'][i] for i in idx]))[:,1]
        ranked=[idx[j] for j in np.argsort(-probs)]
        top1prog+=int(e['programs'][ranked[0]] in e['gold_programs']);top5prog+=int(any(e['programs'][i] in e['gold_programs'] for i in ranked[:5]))
        pred=None
        for i in ranked:
            pred=choose_path(e['paths_by'][e['programs'][i]],e['typ'],e['roles'],coh_w)
            if pred is not None:break
        ok+=int(pred is not None and exact(pred[-1][2],e['a']))
        vh=0
        for g in e['gold_programs']:
            p=choose_path(e['paths_by'].get(g,[]),e['typ'],e['roles'],coh_w)
            if p is not None and exact(p[-1][2],e['a']):vh=1;break
        version+=vh
    return {'accuracy':ok/max(1,n),'program_top1_accuracy':top1prog/max(1,n),'program_top5_recall':top5prog/max(1,n),
            'semantic_prefilter_gold_recall':pref_oracle/max(1,n),'K2_path_oracle':path_oracle/max(1,n),'version_accuracy_given_gold_program':version/max(1,n)}

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k']
    rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    D={s:build(rows[s],model) for s in srcs}
    held_exact={norm(e['q']) for s in srcs[2:] for e in D[s]['examples']}
    dev=[e for s in srcs[:2] for e in D[s]['examples'] if norm(e['q']) not in held_exact]
    overlap=sum(norm(e['q']) in held_exact for s in srcs[:2] for e in D[s]['examples'])
    clf=train_ranker(dev);dev_version,coh_w=tune_coherence(dev)
    results={s:evaluate(D[s],clf,coh_w) for s in srcs}
    out={'stage':'compositional semantic program discriminator + typed K2 execution',
         'protocol':'ranker and coherence weight fitted on 6K+32K only; exact held-out overlaps removed. Relation labels/cues were fixed before held-out question inspection. 262K questions have since been inspected without answers, so 262K is exploratory evidence rather than a pristine blind claim.',
         'dev_examples':len(dev),'overlap_removed':overlap,'coherence_weight':coh_w,'dev_gold_program_version_accuracy':dev_version,
         'program_features':['nested_semantic','step_semantic','lexical_relation_coverage','lexical_strength','reverse_nesting_order','answer_relation_edge_position','answer_relation_semantics','relation_set_semantics','program_length','graph_type_coherence'],
         'results':results,'state':{s:{'facts':D[s]['facts'],'registers':D[s]['registers'],'stored_edges':D[s]['stored_edges']} for s in srcs},
         'guardrail':'gold answers are used only to fit 6K/32K program discriminator and to evaluate diagnostics; 64K/262K answers are not used during fitting.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('COMPOSITIONAL_PROGRAM_K2='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
