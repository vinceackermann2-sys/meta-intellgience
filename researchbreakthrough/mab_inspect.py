import json, urllib.request, urllib.parse
from pathlib import Path

def get(url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'research-inspector/1.0'})
        with urllib.request.urlopen(req,timeout=90) as r:
            return {'status':r.status,'url':r.geturl(),'data':json.loads(r.read().decode('utf-8'))}
    except Exception as e:
        return {'error':repr(e),'url':url}
base='ai-hyz/MemoryAgentBench'; q=urllib.parse.quote(base,safe='')
out={}
out['ttl_rows']=get('https://datasets-server.huggingface.co/first-rows?dataset='+q+'&config=default&split=Test_Time_Learning')
# Extract compact structure from returned rows; preserve source/metadata and short prefixes only.
data=out.get('ttl_rows',{}).get('data',{})
compact=[]
for rowwrap in data.get('rows',[]) if isinstance(data,dict) else []:
    row=rowwrap.get('row',{})
    meta=row.get('metadata') or {}
    compact.append({
      'row_idx':rowwrap.get('row_idx'),
      'source':meta.get('source'),
      'context_chars':len(row.get('context') or ''),
      'context_prefix':(row.get('context') or '')[:1200],
      'questions_count':len(row.get('questions') or []),
      'questions':(row.get('questions') or [])[:5],
      'answers':(row.get('answers') or [])[:5],
      'question_types':(meta.get('question_types') or [])[:10],
      'keypoints':(meta.get('keypoints') or [])[:10],
      'demo_prefix':(meta.get('demo') or '')[:1200],
    })
out['compact']=compact
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_ROWS='+json.dumps(compact,ensure_ascii=False))
