import json, re, string
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path('researchbreakthrough/full_banking77_result.json')

PATTERNS = [
    ('place of birth', re.compile(r'^(.+?) was born in the city of (.+?)\.$')),
    ('place of death', re.compile(r'^(.+?) died in the city of (.+?)\.$')),
    ('position played on team', re.compile(r'^(.+?) plays the position of (.+?)\.$')),
    ('sport', re.compile(r'^(.+?) is associated with the sport of (.+?)\.$')),
    ('continent of', re.compile(r'^(.+?) is located in the continent of (.+?)\.$')),
    ('worked in', re.compile(r'^(.+?) worked in the city of (.+?)\.$')),
    ('spouse', re.compile(r'^(.+?) is married to (.+?)\.$')),
    ('country of citizenship', re.compile(r'^(.+?) is a citizen of (.+?)\.$')),
    ('performed by', re.compile(r'^(.+?) was performed by (.+?)\.$')),
    ('employed by', re.compile(r'^(.+?) is employed by (.+?)\.$')),
    ('founded by', re.compile(r'^(.+?) was founded by (.+?)\.$')),
    ('created by', re.compile(r'^(.+?) was created by (.+?)\.$')),
    ('developed by', re.compile(r'^(.+?) was developed by (.+?)\.$')),
    ('famous for', re.compile(r'^(.+?) is famous for (.+?)\.$')),
    ('language', re.compile(r'^(.+?) speaks the language of (.+?)\.$')),
    ('language', re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
    ('religion', re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
    ('occupation', re.compile(r'^(.+?) works in the field of (.+?)\.$')),
    ('country of origin', re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
    ('founded in', re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
    ('founded in', re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
    ('coached by', re.compile(r'^(.+?) is coached by (.+?)\.$')),
    ('child', re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
    ('child', re.compile(r"^(.+?)'s child is (.+?)\.$")),
    ('capital of', re.compile(r'^The capital of (.+?) is (.+?)\.$')),
    ('headquarters location', re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),
    ('author of', re.compile(r'^The author of (.+?) is (.+?)\.$')),
    ('alma mater', re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),
    ('director', re.compile(r'^The director of (.+?) is (.+?)\.$')),
    ('chairperson', re.compile(r'^The chairperson of (.+?) is (.+?)\.$')),
    ('chief executive officier', re.compile(r'^The chief executive officer of (.+?) is (.+?)\.$')),
    ('broadcaster', re.compile(r'^The (?:origianl|original) broadcaster of (.+?) is (.+?)\.$')),
    ('produced by', re.compile(r'^The company that produced (.+?) is (.+?)\.$')),
    ('genre', re.compile(r'^The type of music that (.+?) plays is (.+?)\.$')),
    ('language', re.compile(r'^The official language of (.+?) is (.+?)\.$')),
    ('language', re.compile(r'^The original language of (.+?) is (.+?)\.$')),
    ('head of state', re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
    ('head of state', re.compile(r'^The name of the current head of state in (.+?) is (.+?)\.$')),
    ('head of government', re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
    ('head of government', re.compile(r'^The Prime Minister of (.+?) is (.+?)\.$')),
    ('head of state', re.compile(r'^The President of (.+?) is (.+?)\.$')),
    ('head of government', re.compile(r'^The Governor of (.+?) is (.+?)\.$')),
    ('head of government', re.compile(r'^The Mayor of (.+?) is (.+?)\.$')),
    ('coached by', re.compile(r'^The coach of (.+?) is (.+?)\.$')),
    ('coached by', re.compile(r'^The head coach of (.+?) is (.+?)\.$')),
    ('officeholder', re.compile(r'^The (.+?) is (.+?)\.$')),
]

REL_PHRASE = {
    'place of birth':'place of birth', 'place of death':'place of death',
    'position played on team':'position played on the team', 'sport':'sport',
    'continent of':'continent', 'worked in':'work location', 'spouse':'spouse',
    'country of citizenship':'country of citizenship', 'performed by':'performer',
    'employed by':'employer', 'founded by':'founder', 'created by':'creator',
    'developed by':'developer', 'famous for':'famous work', 'language':'language',
    'religion':'religion', 'occupation':'occupation', 'country of origin':'country of origin',
    'founded in':'place founded', 'coached by':'coach', 'child':'child',
    'capital of':'capital', 'headquarters location':'headquarters location',
    'author of':'author', 'alma mater':'educational institution', 'director':'director or manager',
    'chairperson':'chairperson', 'chief executive officier':'chief executive officer',
    'broadcaster':'original broadcaster', 'produced by':'producer', 'genre':'genre of music',
    'head of state':'head of state', 'head of government':'head of government',
    'officeholder':'officeholder',
}

REL_TERMS = {
    'place of birth':['birthplace','place of birth','born'],
    'place of death':['place of death','pass away','passed away','died','death'],
    'position played on team':['position','position on the team'],
    'sport':['sport'], 'continent of':['continent'], 'worked in':['worked','work location'],
    'spouse':['spouse','partner','married'], 'country of citizenship':['citizenship','citizen'],
    'performed by':['performer','performed'], 'employed by':['employer','employed'],
    'founded by':['founder','founded','established by'], 'created by':['creator','created by'],
    'developed by':['developer','developed by'], 'famous for':['famous for','known for','notable work'],
    'language':['language','written in','communicate'], 'religion':['religion','believed in','affiliated'],
    'occupation':['occupation','field of work'], 'country of origin':['country of origin','origin','came from','created in','birthplace of the sport'],
    'founded in':['founded in','established','come into existence'], 'coached by':['coach'],
    'child':['child'], 'capital of':['capital'], 'headquarters location':['headquarters'],
    'author of':['author','wrote','written by'], 'alma mater':['university','educated','education','educational institution'],
    'director':['director','manager'], 'chairperson':['chairperson'],
    'chief executive officier':['chief executive officer','ceo'], 'broadcaster':['broadcaster','aired'],
    'produced by':['producer','produced'], 'genre':['genre','type of music'],
    'head of state':['head of state','chief public representative','president'],
    'head of government':['head of government','prime minister','governor','mayor'],
    'officeholder':['officeholder','holds the title'],
}

def parse_fact(text):
    for rel, pat in PATTERNS:
        m=pat.match(text)
        if m: return m.group(1).strip(), rel, m.group(2).strip()
    return None

def fact_lines(context):
    out=[]
    for line in context.splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if m: out.append((int(m.group(1)),m.group(2)))
    return out

def norm_text(s):
    s=s.lower()
    s=''.join(ch for ch in s if ch not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s)
    return ' '.join(s.split())

def answer_list(a):
    if isinstance(a,str): return [a]
    if isinstance(a,list) and a and isinstance(a[0],list): return [x for xs in a for x in xs]
    return list(a or [])

def exact(pred,gold):
    p=norm_text(pred)
    return int(any(p==norm_text(g) for g in answer_list(gold)))

def substring(pred,gold):
    p=norm_text(pred)
    return int(any(norm_text(g) in p for g in answer_list(gold)))

def compile_graph(context):
    facts=fact_lines(context); latest={}; unmatched=[]; parsed=0
    for serial,text in facts:
        z=parse_fact(text)
        if z is None:
            unmatched.append((serial,text)); continue
        parsed+=1; s,r,o=z; key=(s,r)
        if key not in latest or serial>latest[key][0]: latest[key]=(serial,o)
    adj=defaultdict(list); entities=set()
    for (s,r),(serial,o) in latest.items():
        adj[s].append((r,o,serial)); entities.add(s); entities.add(o)
    return facts,latest,adj,entities,unmatched,parsed

def find_anchor(question,adj,entities):
    q=question.casefold()
    candidates=[e for e in adj if e.casefold() in q]
    if candidates: return max(candidates,key=len)
    nq=norm_text(question)
    candidates=[e for e in adj if norm_text(e) and norm_text(e) in nq]
    return max(candidates,key=len) if candidates else None

def enumerate_paths(anchor,adj,max_hops=4):
    out=[]
    stack=[(anchor,[],[],{anchor})]
    while stack:
        node,rels,nodes,seen=stack.pop()
        if rels: out.append((tuple(rels),node,tuple(nodes)))
        if len(rels)>=max_hops: continue
        for r,o,_serial in adj.get(node,[]):
            if o in seen: continue
            stack.append((o,rels+[r],nodes+[o],seen|{o}))
    return out

def describe(anchor,rels):
    d=anchor
    for r in rels:
        d=f"the {REL_PHRASE.get(r,r)} of {d}"
    return d

def cue_score(question,rels):
    q=question.casefold(); hits=0
    for r in rels:
        if any(t in q for t in REL_TERMS.get(r,[r])): hits+=1
    return hits/max(1,len(rels)), hits

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    all_rows=[]
    for row in ds:
        source=(row.get('metadata') or {}).get('source','')
        facts,latest,adj,entities,unmatched,parsed=compile_graph(row['context'])
        questions=list(row.get('questions') or []); answers=list(row.get('answers') or [])
        q_emb=encoder.encode(questions,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        exacts=[]; subs=[]; oracle=[]; anchors=[]; candidate_counts=[]; details=[]
        for qi,(q,gold) in enumerate(zip(questions,answers)):
            anchor=find_anchor(q,adj,entities); anchors.append(int(anchor is not None))
            if anchor is None:
                exacts.append(0); subs.append(0); oracle.append(0); candidate_counts.append(0)
                details.append({'q':q,'gold':answer_list(gold),'anchor':None,'pred':'','rels':[]}); continue
            paths=enumerate_paths(anchor,adj,4); candidate_counts.append(len(paths))
            if not paths:
                exacts.append(0); subs.append(0); oracle.append(0)
                details.append({'q':q,'gold':answer_list(gold),'anchor':anchor,'pred':'','rels':[]}); continue
            oracle.append(int(any(exact(endpoint,gold) for _rels,endpoint,_nodes in paths)))
            desc=[describe(anchor,rels) for rels,_endpoint,_nodes in paths]
            d_emb=encoder.encode(desc,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
            sem=d_emb@q_emb[qi]
            combined=[]
            for j,(rels,_endpoint,_nodes) in enumerate(paths):
                coverage,hits=cue_score(q,rels)
                # Fixed, answer-blind planner: semantic similarity dominates; explicit relation cues break ties.
                combined.append(float(sem[j])+0.10*coverage+0.015*hits-0.003*max(0,len(rels)-1))
            j=int(np.argmax(combined)); rels,pred,nodes=paths[j]
            exacts.append(exact(pred,gold)); subs.append(substring(pred,gold))
            details.append({'q':q,'gold':answer_list(gold),'anchor':anchor,'pred':pred,'rels':list(rels),
                            'score':combined[j],'semantic':float(sem[j]),'oracle_path_exists':oracle[-1]})
        all_rows.append({
            'source':source,'context_chars':len(row['context']),'history_facts':len(facts),
            'parsed_facts':parsed,'parser_coverage':parsed/max(1,len(facts)),
            'unmatched':len(unmatched),'compiled_current_edges':len(latest),
            'history_to_current_edge_ratio':len(facts)/max(1,len(latest)),
            'questions':len(questions),'anchor_coverage':float(np.mean(anchors)),
            'path_exists_oracle':float(np.mean(oracle)),'exact_match':float(np.mean(exacts)),
            'substring_exact_match':float(np.mean(subs)),'mean_candidates':float(np.mean(candidate_counts)),
            'examples':details[:20],
        })
    result={
        'stage':'compiled versioned graph + answer-blind semantic query planner',
        'history_read_at_query_time_chars':0,
        'planner':'longest exact current-graph entity anchor; enumerate 1-4 hops; MiniLM path-description similarity + fixed relation-cue tie break',
        'rows':all_rows,
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('GRAPH_QA_RESULT='+json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
