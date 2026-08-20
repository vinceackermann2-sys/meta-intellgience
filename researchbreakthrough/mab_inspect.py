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

def policy_candidates(name,d,required):
    p=str(name); pl=p.lower().replace('_',''); typ=str(d.get('type','')).lower(); desc=str(d.get('description','')).lower(); optional=p not in required
    vals=[]
    if optional and typ in ('array','list'): vals.append([])
    if optional and typ in ('string','str'): vals.append('')
    if typ in ('bool','boolean'): vals.append(False)
    if 'offset' in pl: vals.append(0)
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl): vals.append(1)
    if pl=='index' and ('latest' in desc or 'most recent' in desc or 'starting from 0' in desc or 'start from 0' in desc): vals.append(0)
    if typ in ('int','integer','float','number') and ('starting from 0' in desc or 'start from 0' in desc) and ('latest' in desc or 'most recent' in desc): vals.append(0)
    return vals

total=0;base_hit=0;policy_added=0;still=[];by_type=Counter()
for q in rows():
    gold=((q.get('tool_call') or {}).get('arguments') or {}); gi=((q.get('tool_call') or {}).get('grounding_info') or {})
    params=((q.get('target_tool_schema') or {}).get('parameters') or {}); props=params.get('properties') or {}; required=set(params.get('required') or [])
    for k,v in gold.items():
        g=gi.get(k) or {}
        if str(g.get('type','')).casefold()!='default': continue
        total+=1; d=props.get(k) or {}; b=schema_candidates(d); ph=policy_candidates(k,d,required)
        if any(norm(x)==norm(v) for x in b): base_hit+=1
        elif any(norm(x)==norm(v) for x in ph): policy_added+=1
        else:
            by_type[str(d.get('type','unknown'))]+=1
            still.append({'qa_id':q.get('qa_id'),'slot':k,'gold':v,'type':d.get('type'),'required':k in required,'description':str(d.get('description',''))[:220],'annotation_source':str(g.get('source_text',''))[:220]})
result={'stage':'Mem2Act schema-policy default oracle on dev only','default_slots':total,'schema_only_covered':base_hit,'general_policy_additional':policy_added,'schema_plus_policy_covered':base_hit+policy_added,'schema_only_rate':base_hit/max(1,total),'schema_plus_policy_rate':(base_hit+policy_added)/max(1,total),'remaining_uncovered':len(still),'remaining_by_type':dict(by_type),'remaining_cases':still,'policy':'optional array -> []; optional string -> empty; boolean -> false; offset -> 0; page/pageIndex -> 1; explicit latest/start-at-zero index -> 0','guardrail':'QA001-100 development labels only. QA101-400 gold remains unopened. General policy contains no task-specific answer constants.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_POLICY_ORACLE='+json.dumps(result,ensure_ascii=False))
