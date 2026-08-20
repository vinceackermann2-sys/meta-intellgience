import json, urllib.request, re
from collections import Counter
from pathlib import Path
URL='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/qa_dataset.jsonl'
def stream():
    with urllib.request.urlopen(URL,timeout=120) as r:
        for raw in r:
            if raw.strip():yield json.loads(raw.decode('utf-8'))

def default_phrase(desc):
    d=str(desc or '')
    pats=[r'(?i)\bdefaults?\s+(?:to\s+)?([^.;,]{1,60})',r'(?i)\bdefault\s+value\s*(?:is|=|:)?\s*([^.;,]{1,60})',r'(?i)\bby\s+default\s*[,=:]?\s*([^.;]{1,60})']
    for p in pats:
        m=re.search(p,d)
        if m:return m.group(1).strip()
    return None
qas=list(stream());types=Counter();stats=Counter();examples=[];temporal=Counter()
for q in qas:
    schema=q.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {})
    for name,d0 in props.items():
        d=d0 or {};types[str(d.get('type','unknown'))]+=1;stats['slots']+=1
        if 'default' in d:stats['json_default']+=1
        if isinstance(d.get('enum'),list) and d.get('enum'):stats['enum']+=1
        ph=default_phrase(d.get('description',''))
        if ph:
            stats['description_default_phrase']+=1
            if len(examples)<30:examples.append({'slot':name,'type':d.get('type'),'phrase':ph,'description':str(d.get('description',''))[:180]})
    qt=str(q.get('query','')).casefold()
    for w in ['today','tomorrow','yesterday','current','latest','upcoming','next week','this week','this month','last week','recent']:
        if w in qt:temporal[w]+=1
out={'stage':'Mem2Act schema-executable slot audit','tasks':len(qas),'schema_slots':stats['slots'],'json_default_slots':stats['json_default'],'enum_slots':stats['enum'],'description_default_phrase_slots':stats['description_default_phrase'],'slot_types':dict(types),'relative_temporal_query_counts':dict(temporal),'default_phrase_examples':examples,'guardrail':'No gold arguments, grounding_info, evolution_chain, or memory content is read; this audits target schemas and current queries only.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_SCHEMA_AUDIT='+json.dumps(out,ensure_ascii=False))
