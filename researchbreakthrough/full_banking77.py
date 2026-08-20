import json, re
from collections import Counter
from pathlib import Path
from datasets import load_dataset

OUT = Path('researchbreakthrough/full_banking77_result.json')

# Relation ontology is the published MQuAKE relation inventory.  These patterns
# map the benchmark's declarative fact surface forms into directed triples.
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
    ('religion', re.compile(r'^(.+?) is affiliated with the religion of (.+?)\.$')),
    ('occupation', re.compile(r'^(.+?) works in the field of (.+?)\.$')),
    ('country of origin', re.compile(r'^(.+?) was created in the country of (.+?)\.$')),
    ('founded in', re.compile(r'^(.+?) was founded in the city of (.+?)\.$')),
    ('founded in', re.compile(r'^(.+?) was founded in the country of (.+?)\.$')),
    ('coached by', re.compile(r'^(.+?) is coached by (.+?)\.$')),
    ('child', re.compile(r'^(.+?) is (?:a|the) child of (.+?)\.$')),

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
    ('head of state', re.compile(r'^The name of the current head of state of (.+?) is (.+?)\.$')),
    ('head of government', re.compile(r'^The name of the current head of (?:the )?(.+?) government is (.+?)\.$')),
    ('coached by', re.compile(r'^The coach of (.+?) is (.+?)\.$')),
]

def parse_fact(text):
    for rel, pat in PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).strip(), rel, m.group(2).strip()
    return None

def fact_lines(context):
    out=[]
    for line in context.splitlines():
        m=re.match(r'^(\d+)\.\s+(.*\S)\s*$', line)
        if m: out.append((int(m.group(1)), m.group(2)))
    return out

def main():
    ds=load_dataset('ai-hyz/MemoryAgentBench', split='Conflict_Resolution', revision='main')
    rows=[]; global_unmatched=Counter(); global_rel=Counter()
    for row in ds:
        source=(row.get('metadata') or {}).get('source','')
        facts=fact_lines(row['context']); rels=Counter(); unmatched=[]
        for serial,text in facts:
            z=parse_fact(text)
            if z is None:
                unmatched.append({'serial':serial,'text':text})
                global_unmatched[text]+=1
            else:
                rels[z[1]]+=1; global_rel[z[1]]+=1
        rows.append({
            'source':source,'context_chars':len(row['context']),'facts':len(facts),
            'matched':len(facts)-len(unmatched),'coverage':(len(facts)-len(unmatched))/max(1,len(facts)),
            'relation_counts':dict(rels),'unmatched_count':len(unmatched),'unmatched_examples':unmatched[:80],
        })
    result={
        'stage':'FactConsolidation ontology-parser coverage before QA',
        'rows':rows,'global_relation_counts':dict(global_rel),
        'global_unmatched_unique':len(global_unmatched),
        'global_unmatched_examples':[{'text':t,'count':n} for t,n in global_unmatched.most_common(200)],
    }
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('CONFLICT_COVERAGE='+json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
