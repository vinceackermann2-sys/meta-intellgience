import json, re, string
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path('researchbreakthrough/full_banking77_result.json')

# Preserve the underlying MQuAKE/Wikidata relation identity.  This matters for
# last-write-wins: P1412 (spoken language), P407 (language of work), P37
# (official language), and P364 (original language) are NOT the same predicate,
# even though a surface-only schema can call all four "language".  Likewise,
# P1308 officeholder facts are distinct from P35/P6 state/government heads.
PATTERNS = [
    ('P19',  'place of birth', re.compile(r'^(.+?) was born in the city of (.+?)\.$')),
    ('P20',  'place of death', re.compile(r'^(.+?) died in the city of (.+?)\.$')),
    ('P413', 'position played on team', re.compile(r'^(.+?) plays the position of (.+?)\.$')),
    ('P641', 'sport', re.compile(r'^(.+?) is associated with the sport of (.+?)\.$')),
    ('P30',  'continent of', re.compile(r'^(.+?) is located in the continent of (.+?)\.$')),
    ('P937', 'worked in', re.compile(r'^(.+?) worked in the city of (.+?)\.$')),
    ('P26',  'spouse', re.compile(r'^(.+?) is married to (.+?)\.$')),
    ('P27',  'country of citizenship', re.compile(r'^(.+?) is a citizen of (.+?)\.$')),
    ('P175', 'performed by', re.compile(r'^(.+?) was performed by (.+?)\.$')),
    ('P108', 'employed by', re.compile(r'^(.+?) is employed by (.+?)\.$')),
    ('P112', 'founded by', re.compile(r'^(.+?) was founded by (.+?)\.$')),
    ('P170', 'created by', re.compile(r'^(.+?) was created by (.+?)\.$')),
    ('P178', 'developed by', re.compile(r'^(.+?) was developed by (.+?)\.$')),
    ('P800', 'famous for', re.compile(r'^(.+?) is famous for (.+?)\.$')),
    ('P1412','spoken language', re.compile(r'^(.+?) speaks the language of (.+?)\.$')),
    ('P407', 'language of work', re.compile(r'^(.+?) was written in the language of (.+?)\.$')),
    ('P140', 'religion', re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
    ('P106', 'occupation', re.compile(r'^(.+?) works in the field of (.+?)\.$')),
    ('P495', 'country of origin', re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
    ('P740', 'founded in', re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
    ('P740', 'founded in', re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
    ('P40',  'child', re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),
    ('P40',  'child', re.compile(r"^(.+?)'s child is (.+?)\.$")),
    ('P36',  'capital of', re.compile(r'^The capital of (.+?) is (.+?)\.$')),
    ('P159', 'headquarters location', re.compile(r'^The headquarters of (.+?) is located in the city of (.+?)\.$')),
    ('P50',  'author of', re.compile(r'^The author of (.+?) is (.+?)\.$')),
    ('P69',  'alma mater', re.compile(r'^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.$')),
    ('P1037','director', re.compile(r'^The director of (.+?) is (.+?)\.$')),
    ('P488', 'chairperson', re.compile(r'^The chairperson of (.+?) is (.+?)\.$')),
    ('P169', 'chief executive officer', re.compile(r'^The chief executive officer of (.+?) is (.+?)\.$')),
    ('P449', 'broadcaster', re.compile(r'^The (?:origianl|original) broadcaster of (.+?) is (.+?)\.$')),
    ('P176', 'produced by', re.compile(r'^The company that produced (.+?) is (.+?)\.$')),
    ('P136', 'genre', re.compile(r'^The type of music that (.+?) plays is (.+?)\.$')),
    ('P37',  'official language', re.compile(r'^The official language of (.+?) is (.+?)\.$')),
    ('P364', 'original language', re.compile(r'^The original language of (.+?) is (.+?)\.$')),
    ('P35',  'head of state', re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
    ('P35',  'head of state', re.compile(r'^The name of the current head of state in (.+?) is (.+?)\.$')),
    ('P6',   'head of government', re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
    ('P286', 'coached by', re.compile(r'^The coach of (.+?) is (.+?)\.$')),
    ('P286', 'coached by', re.compile(r'^The head coach of (.+?) is (.+?)\.$')),
]

# P1308 (officeholder) has surface forms such as "The Prime Minister of X is Y",
# "The Governor of X is Y", "The pope is Y", etc.  Keeping the complete office
# title as the subject prevents these from overwriting P6/P35 facts about X.
OFFICEHOLDER = re.compile(r'^The (.+?) is (.+?)\.$')

REL_NAME = {
    'P19':'place of birth','P20':'place of death','P413':'position played on the team',
    'P641':'sport','P30':'continent','P937':'work location','P26':'spouse',
    'P27':'country of citizenship','P175':'performer','P108':'employer','P112':'founder',
    'P170':'creator','P178':'developer','P800':'notable or famous work',
    'P1412':'language spoken','P407':'language in which the work was written',
    'P140':'religion','P106':'occupation','P495':'country of origin','P740':'place founded',
    'P40':'child','P36':'capital','P159':'headquarters location','P50':'author',
    'P69':'educational institution','P1037':'director or manager','P488':'chairperson',
    'P169':'chief executive officer','P449':'original broadcaster','P176':'producer',
    'P136':'genre of music','P37':'official language','P364':'original language',
    'P35':'head of state','P6':'head of government','P286':'coach','P1308':'officeholder',
}

REL_TERMS = {
    'P19':['birthplace','place of birth','born'],
    'P20':['place of death','pass away','passed away','died','death'],
    'P413':['position','position on the team','speciality','specialty'],
    'P641':['sport','sports discipline'], 'P30':['continent'],
    'P937':['worked','work location','location of work'],
    'P26':['spouse','partner','married'], 'P27':['citizenship','citizen','nationality'],
    'P175':['performer','performed'], 'P108':['employer','employed'],
    'P112':['founder','founded','established by'], 'P170':['creator','created by'],
    'P178':['developer','developed by'], 'P800':['famous for','known for','notable work','significant creation'],
    'P1412':['speak','speaks','communicate'],
    'P407':['written in','wrote','produced their notable work','produced his notable work','produced her notable work'],
    'P140':['religion','faith','believed in','affiliated'], 'P106':['occupation','field of work','job title'],
    'P495':['country of origin','origin','came from','created in','birthplace of the sport','originate'],
    'P740':['founded in','established','come into existence'], 'P40':['child'],
    'P36':['capital'], 'P159':['headquarters','head office'], 'P50':['author','wrote','written by'],
    'P69':['university','educated','education','educational institution','school'],
    'P1037':['director','manager'], 'P488':['chairperson'], 'P169':['chief executive officer','ceo'],
    'P449':['broadcaster','aired','broadcast'], 'P176':['producer','produced','manufacturer'],
    'P136':['genre','type of music'], 'P37':['official language','officially spoken','official documents'],
    'P364':['original language'], 'P35':['head of state','chief public representative'],
    'P6':['head of government'], 'P286':['coach'], 'P1308':['prime minister','president','governor','mayor','pope','officeholder','holds the title'],
}

def parse_fact(text):
    for rid, name, pat in PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).strip(), rid, name, m.group(2).strip()
    m = OFFICEHOLDER.match(text)
    if m:
        return m.group(1).strip(), 'P1308', 'officeholder', m.group(2).strip()
    return None

def fact_lines(context):
    out=[]
    for line in context.splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$',line)
        if m: out.append((int(m.group(1)),m.group(2)))
    return out

def norm_text(s):
    s=s.lower(); s=''.join(ch for ch in s if ch not in string.punctuation)
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

def compile_graph(context, spouse_inverse=True):
    facts=fact_lines(context); latest={}; unmatched=[]; parsed=0
    for serial,text in facts:
        z=parse_fact(text)
        if z is None:
            unmatched.append((serial,text)); continue
        parsed+=1; s,rid,name,o=z; key=(s,rid)
        if key not in latest or serial>latest[key][0]: latest[key]=(serial,o,name)
    adj=defaultdict(list); entities=set()
    for (s,rid),(serial,o,name) in latest.items():
        adj[s].append((rid,o,serial)); entities.add(s); entities.add(o)
    # P26 is semantically symmetric.  Add the inverse only when the benchmark
    # does not provide an explicit current P26 value for that object.
    if spouse_inverse:
        for (s,rid),(serial,o,name) in list(latest.items()):
            if rid=='P26' and (o,'P26') not in latest:
                adj[o].append(('P26',s,serial)); entities.add(o); entities.add(s)
    return facts,latest,adj,entities,unmatched,parsed

def find_anchor(question,adj):
    q=question.casefold()
    candidates=[e for e in adj if e.casefold() in q]
    if candidates: return max(candidates,key=len)
    nq=norm_text(question)
    candidates=[e for e in adj if norm_text(e) and norm_text(e) in nq]
    return max(candidates,key=len) if candidates else None

def enumerate_paths(anchor,adj,max_hops=4):
    out=[]; stack=[(anchor,[],[],{anchor})]
    while stack:
        node,rels,nodes,seen=stack.pop()
        if rels: out.append((tuple(rels),node,tuple(nodes)))
        if len(rels)>=max_hops: continue
        for rid,o,_serial in adj.get(node,[]):
            if o in seen: continue
            stack.append((o,rels+[rid],nodes+[o],seen|{o}))
    return out

def describe(anchor,rels):
    d=anchor
    for rid in rels: d=f"the {REL_NAME.get(rid,rid)} of {d}"
    return d

def cue_score(question,rels):
    q=question.casefold(); hits=0; strong=0
    for rid in rels:
        terms=REL_TERMS.get(rid,[REL_NAME.get(rid,rid)])
        if any(t in q for t in terms):
            hits+=1
            # Distinguishing cues for predicates that used to be conflated.
            if rid in {'P1412','P407','P37','P364','P35','P6','P1308'}: strong+=1
    return hits/max(1,len(rels)), hits, strong

def evaluate_row(row,encoder):
    source=(row.get('metadata') or {}).get('source','')
    facts,latest,adj,entities,unmatched,parsed=compile_graph(row['context'],spouse_inverse=True)
    questions=list(row.get('questions') or []); answers=list(row.get('answers') or [])
    q_emb=encoder.encode(questions,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    variants={
        'semantic_only':[],
        'semantic_plus_schema':[],
        'schema_strong':[],
    }
    subs={k:[] for k in variants}; oracle=[]; anchors=[]; candidate_counts=[]; details=[]
    for qi,(q,gold) in enumerate(zip(questions,answers)):
        anchor=find_anchor(q,adj); anchors.append(int(anchor is not None))
        if anchor is None:
            oracle.append(0); candidate_counts.append(0)
            for k in variants: variants[k].append(0); subs[k].append(0)
            continue
        paths=enumerate_paths(anchor,adj,4); candidate_counts.append(len(paths))
        if not paths:
            oracle.append(0)
            for k in variants: variants[k].append(0); subs[k].append(0)
            continue
        oracle.append(int(any(exact(endpoint,gold) for _rels,endpoint,_nodes in paths)))
        desc=[describe(anchor,rels) for rels,_endpoint,_nodes in paths]
        d_emb=encoder.encode(desc,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        sem=d_emb@q_emb[qi]
        scores={k:[] for k in variants}
        for j,(rels,_endpoint,_nodes) in enumerate(paths):
            coverage,hits,strong=cue_score(q,rels)
            length_pen=0.003*max(0,len(rels)-1)
            scores['semantic_only'].append(float(sem[j])-length_pen)
            scores['semantic_plus_schema'].append(float(sem[j])+0.10*coverage+0.015*hits+0.025*strong-length_pen)
            scores['schema_strong'].append(float(sem[j])+0.15*coverage+0.025*hits+0.06*strong-length_pen)
        chosen={}
        for k in variants:
            j=int(np.argmax(scores[k])); rels,pred,nodes=paths[j]
            variants[k].append(exact(pred,gold)); subs[k].append(substring(pred,gold))
            chosen[k]={'pred':pred,'rels':list(rels),'score':scores[k][j],'semantic':float(sem[j])}
        if len(details)<25:
            details.append({'q':q,'gold':answer_list(gold),'anchor':anchor,'oracle_path_exists':oracle[-1],
                            'semantic_plus_schema':chosen['semantic_plus_schema'],'schema_strong':chosen['schema_strong']})
    return {
        'source':source,'context_chars':len(row['context']),'history_facts':len(facts),
        'parsed_facts':parsed,'parser_coverage':parsed/max(1,len(facts)),'unmatched':len(unmatched),
        'compiled_current_edges':len(latest),'history_to_current_edge_ratio':len(facts)/max(1,len(latest)),
        'questions':len(questions),'anchor_coverage':float(np.mean(anchors)),'path_exists_oracle':float(np.mean(oracle)),
        'exact_match':{k:float(np.mean(v)) for k,v in variants.items()},
        'substring_exact_match':{k:float(np.mean(subs[k])) for k in subs},
        'mean_candidates':float(np.mean(candidate_counts)),'examples':details,
    }

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench',split='Conflict_Resolution',revision='main')
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    rows=[evaluate_row(row,encoder) for row in ds]
    result={
        'stage':'MQuAKE-relation-aware compiled versioned graph',
        'history_read_at_query_time_chars':0,
        'state_rule':'max serial per exact MQuAKE predicate; preserve P1412/P407/P37/P364 and P35/P6/P1308 identities; spouse symmetry fallback',
        'planner':'longest exact current-graph entity anchor; enumerate 1-4 hops; answer-blind MiniLM + schema cue ablations',
        'rows':rows,
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('RELATION_AWARE_GRAPH_RESULT='+json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
