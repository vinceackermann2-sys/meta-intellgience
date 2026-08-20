import json, urllib.request, re
from collections import Counter
from pathlib import Path
URL='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/qa_dataset.jsonl'
def rows():
    with urllib.request.urlopen(URL,timeout=120) as r:
        for i,raw in enumerate(r):
            if i>=100:break
            if raw.strip():yield json.loads(raw.decode('utf-8'))
def cat(slot,gold,src,desc,query):
    t=' '.join(map(str,[slot,gold,src,desc,query])).casefold()
    if any(x in t for x in ['today','tomorrow','yesterday','current season','current year','next week','upcoming','date','year','season']):return 'temporal'
    if isinstance(gold,bool) or str(gold).casefold() in ('true','false'):return 'boolean'
    if re.fullmatch(r'[A-Z][A-Z0-9_-]{1,12}',str(gold)):return 'acronym_or_code'
    if re.search(r'https?://|@|[A-Fa-f0-9]{16,}|[_-].*[_-]',str(gold)):return 'identifier_or_structured'
    if isinstance(gold,(int,float)):return 'numeric_transform'
    if any(x in t for x in ['city','country','location','state','province','region','language','currency','timezone']):return 'entity_or_geo'
    return 'semantic_inference'
out=[];C=Counter()
for q in rows():
    gold=((q.get('tool_call') or {}).get('arguments') or {});gi=((q.get('tool_call') or {}).get('grounding_info') or {});props=(((q.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
    for k,v in gold.items():
        g=gi.get(k) or {}
        if str(g.get('type','')).casefold()!='inferred':continue
        src=str(g.get('source_text',''));desc=str((props.get(k) or {}).get('description',''));query=str(q.get('query',''));c=cat(k,v,src,desc,query);C[c]+=1
        out.append({'qa_id':q.get('qa_id'),'slot':k,'category':c,'gold_type':type(v).__name__,'gold':v,'source_text':src[:260],'schema_description':desc[:220],'query':query[:220]})
result={'stage':'Mem2Act inferred-transformation dev audit','n':len(out),'category_counts':dict(C),'cases':out,'guardrail':'QA001-100 development labels only. QA101-400 gold remains unopened. Categories are diagnostic heuristics, not benchmark-time rules.'}
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_INFERRED_DEV='+json.dumps(result,ensure_ascii=False))
