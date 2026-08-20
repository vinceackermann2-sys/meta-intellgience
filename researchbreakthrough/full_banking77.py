import json, re, string
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

OUT=Path('researchbreakthrough/full_banking77_result.json')

PATTERNS=[
('P19',re.compile(r'^(.+?) was born in the city of (.+?)\.$')),
('P20',re.compile(r'^(.+?) died in the city of (.+?)\.$')),
('P413',re.compile(r'^(.+?) plays the position of (.+?)\.$')),
('P641',re.compile(r'^(.+?) is associated with the sport of (.+?)\.$')),
('P30',re.compile(r'^(.+?) is located in the continent of (.+?)\.$')),
('P937',re.compile(r'^(.+?) worked in the city of (.+?)\.$')),
('P26',re.compile(r'^(.+?) is married to (.+?)\.$')),
('P27',re.compile(r'^(.+?) is a citizen of (.+?)\.$')),
('P175',re.compile(r'^(.+?) was performed by (.+?)\.$')),
('P108',re.compile(r'^(.+?) is employed by (.+?)\.$')),
('P112',re.compile(r'^(.+?) was founded by (.+?)\.$')),
('P170',re.compile(r'^(.+?) was created by (.+?)\.$')),
('P178',re.compile(r'^(.+?) was developed by (.+?)\.$')),
('P800',re.compile(r'^(.+?) is famous for (.+?)\.$')),
('P1412',re.compile(r'^(.+?) speaks the language of (.+?)\.$')),
('P407',re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
('P140',re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
('P106',re.compile(r'^(.+?) works in the field of (.+?)\.$')),
('P495',re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
('P740',re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
('P740',re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
('P40',re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
('P40',re.compile(r"^(.+?)'s child is (.+?)\.$")),
('P36',re.compile(r'^The capital of (.+?) is (.+?)\.$')),
('P159',re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),
('P50',re.compile(r'^The author of (.+?) is (.+?)\.$')),
('P69',re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),
('P1037',re.compile(r'^The director of (.+?) is (.+?)\.$')),
('P488',re.compile(r'^The chairperson of (.+?) is (.+?)\.$')),
('P169',re.compile(r'^The chief executive officer of (.+?) is (.+?)\.$')),
('P449',re.compile(r'^The (?:origianl|original) broadcaster of (.+?) is (.+?)\.$')),
('P176',re.compile(r'^The company that produced (.+?) is (.+?)\.$')),
('P136',re.compile(r'^The type of music that (.+?) plays is (.+?)\.$')),
('P37',re.compile(r'^The official language of (.+?) is (.+?)\.$')),
('P364',re.compile(r'^The original language of (.+?) is (.+?)\.$')),
('P35',re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
('P35',re.compile(r'^The name of the current head of state in (.+?) is (.+?)\.$')),
('P6',re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
('P286',re.compile(r'^The coach of (.+?) is (.+?)\.$')),
('P286',re.compile(r'^The head coach of (.+?) is (.+?)\.$')),
]
OFFICEHOLDER=re.compile(r'^The (.+?) is (.+?)\.$')

def parse_fact(text):
    for rid,pat in PATTERNS:
        m=pat.match(text)
        if m:return m.group(1).strip(),rid,m.group(2).strip()
    m=OFFICEHOLDER.match(text)
    return (m.group(1).strip(),'P1308',m.group(2).strip()) if m else None

def facts(context):
    out=[]
    for line in context.splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if m:
            z=parse_fact(m.group(2))
            if z: out.append((int(m.group(1)),*z,m.group(2)))
    return out

def norm(s):
    s=s.lower();s=''.join(c for c in s if c not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())

def golds(a):
    if isinstance(a,str):return [a]
    if a and isinstance(a[0],list):return [x for xs in a for x in xs]
    return list(a or [])

def is_gold(x,a):return any(norm(x)==norm(g) for g in golds(a))

def build(context):
    fs=facts(context); latest={}; all_adj=defaultdict(list)
    for serial,s,r,o,text in fs:
        all_adj[s].append((r,o,serial,text))
        k=(s,r)
        if k not in latest or serial>latest[k][0]:latest[k]=(serial,o,text)
    # spouse inverse history; inverse carries source serial and only supplements traversal.
    for serial,s,r,o,text in list(fs):
        if r=='P26':all_adj[o].append((r,s,serial,'[inverse] '+text))
    current=defaultdict(list)
    for (s,r),(serial,o,text) in latest.items():current[s].append((r,o,serial,text))
    for (s,r),(serial,o,text) in list(latest.items()):
        if r=='P26' and (o,'P26') not in latest:current[o].append((r,s,serial,'[inverse] '+text))
    return fs,latest,all_adj,current

def anchor(q,adj):
    qf=q.casefold();c=[e for e in adj if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in adj if norm(e) and norm(e) in nq]
    return max(c,key=len) if c else None

def paths(start,adj,max_hops=4,cap=200000):
    out=[];stack=[(start,[],{start})]
    while stack and len(out)<cap:
        node,edges,seen=stack.pop()
        if edges:out.append(edges)
        if len(edges)>=max_hops:continue
        for r,o,serial,text in adj.get(node,[]):
            if o in seen:continue
            stack.append((o,edges+[(node,r,o,serial,text)],seen|{o}))
    return out

def endpoint(p):return p[-1][2]

def forensic(row):
    fs,latest,all_adj,current=build(row['context'])
    qs=list(row['questions']);ans=list(row['answers'])
    failed=[]; cur_oracle=[]; hist_oracle=[]
    for q,a in zip(qs,ans):
        st=anchor(q,all_adj)
        cp=paths(st,current) if st else []
        hp=paths(st,all_adj) if st else []
        co=any(is_gold(endpoint(p),a) for p in cp); ho=any(is_gold(endpoint(p),a) for p in hp)
        cur_oracle.append(int(co));hist_oracle.append(int(ho))
        if co:continue
        gps=[p for p in hp if is_gold(endpoint(p),a)]
        # Prefer shortest gold path; then smallest serial span, then newest minimum serial.
        gps.sort(key=lambda p:(len(p), max(e[3] for e in p)-min(e[3] for e in p), -min(e[3] for e in p)))
        sample=[]
        for p in gps[:8]:
            edgeinfo=[]
            for s,r,o,serial,text in p:
                ls=latest.get((s,r),(None,None,None))[0]
                lo=latest.get((s,r),(None,None,None))[1]
                edgeinfo.append({'s':s,'r':r,'o':o,'serial':serial,'latest_serial':ls,'latest_object':lo,'is_local_latest':serial==ls})
            ser=[e[3] for e in p]
            sample.append({'endpoint':endpoint(p),'hops':len(p),'serials':ser,'serial_span':max(ser)-min(ser),
                           'min_serial':min(ser),'max_serial':max(ser),'all_local_latest':all(e['is_local_latest'] for e in edgeinfo),
                           'edges':edgeinfo})
        failed.append({'q':q,'gold':golds(a),'anchor':st,'history_gold_path_exists':ho,
                       'history_candidates':len(hp),'gold_path_count':len(gps),'gold_paths':sample})
    return {'source':(row.get('metadata') or {}).get('source',''),'history_facts':len(fs),'latest_registers':len(latest),
            'current_path_oracle':sum(cur_oracle)/len(cur_oracle),'full_history_path_oracle':sum(hist_oracle)/len(hist_oracle),
            'missing_current':len(failed),'failures':failed}

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    targets=[r for r in ds if (r.get('metadata') or {}).get('source') in {'factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k'}]
    result={'stage':'forensic comparison: independent latest registers vs full version-history gold-path reachability',
            'note':'gold is used only to diagnose state-loss after the answer-blind benchmark; not a prediction method',
            'rows':[forensic(r) for r in targets]}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('VERSION_CHAIN_FORENSICS='+json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
