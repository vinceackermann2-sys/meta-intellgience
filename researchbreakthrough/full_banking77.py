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
LABEL2PID={'author':'P50','capital':'P36','chairperson':'P488','chief executive officer':'P169','child':'P40','continent':'P30',
'country of citizenship':'P27','country of origin':'P495','creator':'P170','developer':'P178','director / manager':'P1037','educated at':'P69',
'employer':'P108','founded by':'P112','genre':'P136','head coach':'P286','head of government':'P6','head of state':'P35','headquarters location':'P159',
'language of work or name':'P407','languages spoken, written or signed':'P1412','location of formation':'P740','manufacturer':'P176','notable work':'P800',
'occupation':'P106','officeholder':'P1308','official language':'P37','original broadcaster':'P449','performer':'P175','place of birth':'P19','place of death':'P20',
'position played on team / speciality':'P413','religion or worldview':'P140','sport':'P641','spouse':'P26','work location':'P937'}

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
    pos=[];covered=0;strength=0.0
    for r in s:
        hs=alias_hits(q,r)
        if hs:
            covered+=1;best=max(hs,key=lambda z:(z[1],z[2]));pos.append(best[0]);strength+=min(1.0,best[1]/3.0)
        else:pos.append(None)
    known=[(i,p) for i,p in enumerate(pos) if p is not None];inv_score=0.5
    if len(known)>=2:
        good=tot=0
        for a in range(len(known)):
            for b in range(a+1,len(known)):
                tot+=1;good+=int(known[a][1]>=known[b][1])
        inv_score=good/max(1,tot)
    last_hits=alias_hits(q,s[-1]);edge=0.0
    if last_hits:
        L=max(1,len(q));edge=max(max(1-h[0]/L,h[0]/L) for h in last_hits)
    return covered/len(s),strength/len(s),inv_score,edge

def target_zones(q):
    toks=re.findall(r"[\w'-]+",q.casefold());return ' '.join(toks[:14]),' '.join(toks[-14:])

def type_map(adj):
    typ=defaultdict(lambda:defaultdict(float))
    roles={'P19':('person','location'),'P20':('person','location'),'P413':('person','position'),'P641':('entity','sport'),'P30':('country','continent'),
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
        a,b=roles.get(r,('entity','entity'));os=sum(typ[o].values())+1e-6;ss=sum(typ[s].values())+1e-6
        vals.append(0.5*(typ[o].get(b,0)/os)+0.5*(typ[s].get(a,0)/ss))
    return float(np.mean(vals)) if vals else 0.0

def choose_path(ps,typ,roles,coh_weight=0.0):
    if not ps:return None
    def score(p):
        shadow=sum(e[4] for e in p);span=(max(e[3] for e in p)-min(e[3] for e in p)) if len(p)>1 else 0;coh=coherence_for_path(p,typ,roles)
        return -shadow-0.00001*span+coh_weight*coh
    return max(ps,key=score)

def feature_rows(q,st,programs,by,model,rel_emb,rel_ids,typ,roles):
    qm=qmask(q,st);head,tail=target_zones(qm);texts=[]
    for s in programs:texts.extend([canonical_nested(s),canonical_steps(s)])
    pe=model.encode(texts,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if texts else np.zeros((0,384),dtype=np.float32)
    qe=model.encode([qm,head,tail],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);feats=[]
    for j,s in enumerate(programs):
        cov,strength,order,edge=lexical_features(qm,s);nested=float(pe[2*j]@qe[0]);steps=float(pe[2*j+1]@qe[0]);li=rel_ids.index(s[-1])
        target_sem=max(float(rel_emb[li]@qe[1]),float(rel_emb[li]@qe[2]));rsem=float(np.mean([max(float(rel_emb[rel_ids.index(r)]@qe[k]) for k in range(3)) for r in s]));cur=max((coherence_for_path(p,typ,roles) for p in by[s]),default=0.0)
        feats.append([nested,steps,cov,strength,order,edge,target_sem,rsem,len(s),cur])
    return feats

def build(row,model,rel_emb,rel_ids):
    adj,nf,nreg,nedge=compile_row(row,2);typ,roles=type_map(adj);E=[]
    for q,a in zip(row['questions'],row['answers']):
        st=anchor(q,adj);ps=paths(st,adj);by=defaultdict(list)
        for p in ps:by[seq(p)].append(p)
        programs=list(by);feats=feature_rows(q,st,programs,by,model,rel_emb,rel_ids,typ,roles);gp={seq(p) for p in ps if exact(p[-1][2],a)}
        E.append({'q':q,'a':a,'programs':programs,'paths_by':by,'features':feats,'gold_programs':gp,'typ':typ,'roles':roles})
    return {'facts':nf,'registers':nreg,'stored_edges':nedge,'examples':E}

def prefilter(e,n=12):
    vals=[]
    for i,f in enumerate(e['features']):
        nested,steps,cov,strength,order,edge,target_sem,rsem,L,cur=f;v=0.55*nested+0.2*steps+0.12*cov+0.06*order+0.05*target_sem+0.02*rsem-0.012*(L-1);vals.append((v,i))
    return [i for _,i in sorted(vals,reverse=True)[:n]]

def train_ranker(dev):
    X=[];y=[]
    for e in dev:
        for i in prefilter(e,20):X.append(e['features'][i]);y.append(int(e['programs'][i] in e['gold_programs']))
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=3000,random_state=7));clf.fit(np.asarray(X),np.asarray(y));return clf

def tune_coherence(dev):
    best=(0,0)
    for w in [0,0.25,0.5,1,2,4]:
        ok=n=0
        for e in dev:
            for g in e['gold_programs']:
                p=choose_path(e['paths_by'].get(g,[]),e['typ'],e['roles'],w);n+=1;ok+=int(p is not None and exact(p[-1][2],e['a']));break
        if ok/max(1,n)>best[0]:best=(ok/max(1,n),w)
    return best

def evaluate(D,clf,coh_w):
    ok=top1=top5=prefo=patho=version=0;n=0
    for e in D['examples']:
        n+=1;patho+=int(bool(e['gold_programs']));idx=prefilter(e,12);prefo+=int(any(e['programs'][i] in e['gold_programs'] for i in idx))
        if not idx:continue
        probs=clf.predict_proba(np.asarray([e['features'][i] for i in idx]))[:,1];ranked=[idx[j] for j in np.argsort(-probs)];top1+=int(e['programs'][ranked[0]] in e['gold_programs']);top5+=int(any(e['programs'][i] in e['gold_programs'] for i in ranked[:5]))
        pred=choose_path(e['paths_by'][e['programs'][ranked[0]]],e['typ'],e['roles'],coh_w);ok+=int(pred is not None and exact(pred[-1][2],e['a']))
        vh=0
        for g in e['gold_programs']:
            p=choose_path(e['paths_by'].get(g,[]),e['typ'],e['roles'],coh_w)
            if p is not None and exact(p[-1][2],e['a']):vh=1;break
        version+=vh
    return {'accuracy':ok/max(1,n),'program_top1_accuracy':top1/max(1,n),'program_top5_recall':top5/max(1,n),'semantic_prefilter_gold_recall':prefo/max(1,n),'K2_path_oracle':patho/max(1,n),'version_accuracy_given_gold_program':version/max(1,n)}

def remastered_answers(row):
    a=[];x=row.get('new_answer')
    if isinstance(x,str):a.append(x)
    elif isinstance(x,list):a.extend(str(z) for z in x)
    al=row.get('new_answer_alias') or []
    if isinstance(al,str):a.append(al)
    elif isinstance(al,list):a.extend(str(z) for z in al)
    return [z for z in a if z]

def evaluate_remastered(model,rel_emb,rel_ids,clf):
    ds=load_dataset('henryzhongsc/MQuAKE-Remastered',split='CF3k');agg=defaultdict(lambda:{'questions':0,'answer_exact':0,'anchor_coverage':0,'gold_endpoint_path':0,'program_selected_gold_endpoint':0})
    for case_i,row in enumerate(ds):
        ts=list(row.get('new_triples_labeled') or []);adj=defaultdict(list);mapped=True
        for k,t in enumerate(ts):
            if not isinstance(t,(list,tuple)) or len(t)<3 or str(t[1]) not in LABEL2PID:mapped=False;break
            s,rlabel,o=map(str,t[:3]);adj[s].append((LABEL2PID[rlabel],o,k,0))
        if not mapped:continue
        for s,es in list(adj.items()):
            for r,o,ser,rank in list(es):
                if r=='P26':adj[o].append((r,s,ser,rank))
        typ,roles=type_map(adj);answers=remastered_answers(row);qs=row.get('questions') or []
        if isinstance(qs,str):qs=[qs]
        bucket=str(len(ts))
        for q in qs:
            A=agg[bucket];A['questions']+=1;st=anchor(q,adj);A['anchor_coverage']+=int(st is not None);ps=paths(st,adj);by=defaultdict(list)
            for p in ps:by[seq(p)].append(p)
            A['gold_endpoint_path']+=int(any(exact(p[-1][2],answers) for p in ps))
            programs=list(by)
            if not programs:continue
            feats=feature_rows(q,st,programs,by,model,rel_emb,rel_ids,typ,roles);e={'programs':programs,'features':feats};idx=prefilter(e,12)
            if not idx:continue
            probs=clf.predict_proba(np.asarray([feats[i] for i in idx]))[:,1];best=idx[int(np.argmax(probs))];pred=choose_path(by[programs[best]],typ,roles,0)
            hit=int(pred is not None and exact(pred[-1][2],answers));A['answer_exact']+=hit;A['program_selected_gold_endpoint']+=hit
    total={k:sum(v[k] for v in agg.values()) for k in ['questions','answer_exact','anchor_coverage','gold_endpoint_path','program_selected_gold_endpoint']}
    def fmt(v):
        n=max(1,v['questions']);return {'questions':v['questions'],'exact_match':v['answer_exact']/n,'anchor_coverage':v['anchor_coverage']/n,'gold_endpoint_path_oracle':v['gold_endpoint_path']/n}
    return {'by_chain_length':{k:fmt(v) for k,v in sorted(agg.items())},'overall':fmt(total)}

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main');srcs=['factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k'];rows={(r.get('metadata') or {}).get('source'):r for r in ds if (r.get('metadata') or {}).get('source') in srcs}
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');rel_ids=list(RELNAME);rel_text=[RELNAME[r]+'. '+', '.join(CUES.get(r,[])) for r in rel_ids];rel_emb=model.encode(rel_text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    D={s:build(rows[s],model,rel_emb,rel_ids) for s in srcs};held={norm(e['q']) for s in srcs[2:] for e in D[s]['examples']};dev=[e for s in srcs[:2] for e in D[s]['examples'] if norm(e['q']) not in held];clf=train_ranker(dev);dev_version,coh_w=tune_coherence(dev)
    mab={s:evaluate(D[s],clf,coh_w) for s in srcs}
    # FIRST-SHOT VALIDATION: this call is the first time this frozen parser reads CF3k question/answer fields; only aggregate metrics are emitted.
    rem=evaluate_remastered(model,rel_emb,rel_ids,clf)
    out={'stage':'frozen compositional parser: exploratory MAB + first-shot MQuAKE-Remastered transfer','protocol':{'fit':'MAB 6K+32K only','MAB_262K':'exploratory because questions were inspected answer-free before this run','MQuAKE_Remastered_CF3k':'first-shot blind question/answer transfer; relation inventory and chain lengths only were inspected before freezing'},'dev_examples':len(dev),'coherence_weight':coh_w,'mab_results':mab,'fresh_remastered':rem,'guardrail':'No CF3k question or answer was used to change relation cues, features, ranker, or weights before the fresh transfer metric was computed.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('FROZEN_FRESH_TRANSFER='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
