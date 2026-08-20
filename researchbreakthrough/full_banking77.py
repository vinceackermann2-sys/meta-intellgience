import json, re, string
from collections import defaultdict, Counter
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

def parse_context(context):
    fs=[]
    for line in context.splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if m:
            z=parse_fact(m.group(2))
            if z:fs.append((int(m.group(1)),*z))
    return fs

def norm(s):
    s=s.lower();s=''.join(c for c in s if c not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())

def golds(a):
    if isinstance(a,str):return [a]
    if a and isinstance(a[0],list):return [x for xs in a for x in xs]
    return list(a or [])

def is_gold(x,a):return any(norm(x)==norm(g) for g in golds(a))

def registers(fs):
    reg=defaultdict(dict)
    for serial,s,r,o in fs:
        # For each distinct object retain only its newest occurrence.  K therefore
        # measures distinct semantic versions, not duplicate copies of one fact.
        reg[(s,r)][o]=max(serial,reg[(s,r)].get(o,-1))
    return {k:sorted(((serial,o) for o,serial in vals.items()),reverse=True) for k,vals in reg.items()}

def make_adj(reg,k):
    adj=defaultdict(list); kept=0
    for (s,r),versions in reg.items():
        take=versions if k is None else versions[:k]
        for serial,o in take:
            adj[s].append((r,o,serial));kept+=1
    # P26 is symmetric for traversal.  Add inverse traversals without counting
    # them as stored state; they are a derived index over the same version edge.
    for s,edges in list(adj.items()):
        for r,o,serial in list(edges):
            if r=='P26':adj[o].append((r,s,serial))
    return adj,kept

def anchor(q,subjects):
    qf=q.casefold();c=[e for e in subjects if e.casefold() in qf]
    if c:return max(c,key=len)
    nq=norm(q);c=[e for e in subjects if norm(e) and norm(e) in nq]
    return max(c,key=len) if c else None

def reachable_gold(start,adj,a,max_hops=4):
    if not start:return False
    stack=[(start,0,{start})]
    while stack:
        node,h,seen=stack.pop()
        if h>=max_hops:continue
        for _r,o,_serial in adj.get(node,[]):
            if is_gold(o,a):return True
            if o not in seen:stack.append((o,h+1,seen|{o}))
    return False

def eval_row(row):
    fs=parse_context(row['context']);reg=registers(fs)
    qs=list(row['questions']);ans=list(row['answers'])
    subjects=set(s for s,_r in reg)
    ks=[1,2,3,4,8,None]; per={}; reached_by_q={i:[] for i in range(len(qs))}
    for k in ks:
        adj,kept=make_adj(reg,k); hit=[]
        for i,(q,a) in enumerate(zip(qs,ans)):
            st=anchor(q,subjects);ok=reachable_gold(st,adj,a);hit.append(int(ok))
            if ok:reached_by_q[i].append(k if k is not None else 999)
        name='all' if k is None else str(k)
        per[name]={'path_oracle':sum(hit)/len(hit),'stored_version_edges':kept,
                   'edge_multiplier_vs_K1':None}
    base=per['1']['stored_version_edges']
    for z in per.values():z['edge_multiplier_vs_K1']=z['stored_version_edges']/base
    min_k=[]
    for i in range(len(qs)):
        vals=reached_by_q[i]
        min_k.append(min(vals) if vals else None)
    dist=Counter(('unreachable' if x is None else ('all_only' if x==999 else str(x))) for x in min_k)
    # Register version-depth statistics explain the state cost independent of QA.
    depths=[len(v) for v in reg.values()]
    return {'source':(row.get('metadata') or {}).get('source',''),'history_facts':len(fs),
            'logical_registers':len(reg),'distinct_version_edges':sum(depths),
            'register_depth_mean':sum(depths)/len(depths),'register_depth_max':max(depths),
            'registers_with_conflicts':sum(d>1 for d in depths),'versions':per,
            'minimum_K_to_recover_gold_path_distribution':dict(dist)}

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    wanted={'factconsolidation_mh_6k','factconsolidation_mh_32k','factconsolidation_mh_64k','factconsolidation_mh_262k'}
    rows=[eval_row(r) for r in ds if (r.get('metadata') or {}).get('source') in wanted]
    result={'stage':'bounded distinct-version stack oracle audit','gold_usage':'diagnostic reachability only; no gold used for prediction',
            'hypothesis':'small K-version stacks recover paths destroyed by K=1 newest-only consolidation','rows':rows}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('VERSION_STACK_AUDIT='+json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
