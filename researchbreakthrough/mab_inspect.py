import json, urllib.request, re
from collections import Counter
from pathlib import Path
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'

def rows():
    with urllib.request.urlopen(BASE+'qa_dataset.jsonl',timeout=120) as r:
        for i,raw in enumerate(r):
            if i>=100: break
            if raw.strip(): yield json.loads(raw.decode('utf-8'))

def norm(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None: return 'null'
    if isinstance(v,(dict,list)): return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()

def schema_candidates(d):
    vals=[]
    if 'default' in d: vals.append(d['default'])
    if isinstance(d.get('enum'),list): vals.extend(d['enum'])
    desc=str(d.get('description',''))
    pats=[r"(?i)default\s+value\s+(?:is|=|:)\s*['\"]?([^'\".,;\)]+)",r"(?i)defaults?\s+to\s*['\"]?([^'\".,;\)]+)",r"(?i)by\s+default[^\w]+(?:the\s+)?(?:value\s+)?(?:is\s+)?['\"]?([^'\".,;\)]+)",r"(?i)([A-Za-z0-9._/-]+)\s*\((?:the\s+)?default\)"]
    for p in pats:
        for m in re.finditer(p,desc): vals.append(m.group(1).strip())
    return vals

def classify(v,src,desc,query):
    nv=norm(v); blob=' '.join([src,desc,query]).casefold()
    if nv and nv in norm(desc): return 'description_contains_gold'
    if nv and nv in norm(query): return 'query_contains_gold'
    if nv and nv in norm(src): return 'annotation_source_contains_gold'
    if isinstance(v,bool): return 'implicit_boolean'
    if isinstance(v,(int,float)): return 'implicit_numeric'
    if any(x in blob for x in ['first page','latest','most recent','default','if not','unspecified','fallback']): return 'implicit_policy_default'
    return 'opaque_or_world_knowledge_default'

cases=[]; counts=Counter(); total=0; covered=0
for q in rows():
    gold=((q.get('tool_call') or {}).get('arguments') or {})
    gi=((q.get('tool_call') or {}).get('grounding_info') or {})
    props=(((q.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
    for k,v in gold.items():
        g=gi.get(k) or {}
        if str(g.get('type','')).casefold()!='default': continue
        total+=1; d=props.get(k) or {}; cands=schema_candidates(d); hit=any(norm(x)==norm(v) for x in cands)
        if hit: covered+=1; continue
        src=str(g.get('source_text','')); desc=str(d.get('description','')); query=str(q.get('query',''))
        cat=classify(v,src,desc,query); counts[cat]+=1
        cases.append({'qa_id':q.get('qa_id'),'slot':k,'gold':v,'category':cat,'annotation_source':src[:260],'schema_description':desc[:260],'query':query[:220]})
result={'stage':'Mem2Act unresolved-default dev audit','default_slots':total,'schema_candidate_covered':covered,'uncovered':len(cases),'uncovered_categories':dict(counts),'cases':cases,'guardrail':'QA001-100 development labels only. QA101-400 gold remains unopened. This diagnoses why benchmark-labeled defaults are not executable from released schema metadata.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_DEFAULT_GAP='+json.dumps(result,ensure_ascii=False))
