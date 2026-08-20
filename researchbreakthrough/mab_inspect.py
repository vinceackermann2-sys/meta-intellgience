import json, urllib.request, urllib.parse
from pathlib import Path

def get(url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'research-inspector/1.0'})
        with urllib.request.urlopen(req,timeout=90) as r:
            return {'status':r.status,'url':r.geturl(),'data':json.loads(r.read().decode('utf-8'))}
    except Exception as e:
        return {'error':repr(e),'url':url}
def obj(v):
    if isinstance(v,dict): return v
    if isinstance(v,str):
        try:return json.loads(v)
        except Exception:return {'_raw':v}
    return {}
base='ai-hyz/MemoryAgentBench'; q=urllib.parse.quote(base,safe='')
out={}
out['ttl_rows']=get('https://datasets-server.huggingface.co/first-rows?dataset='+q+'&config=default&split=Test_Time_Learning')
data=out.get('ttl_rows',{}).get('data',{})
compact=[]
for rowwrap in data.get('rows',[]) if isinstance(data,dict) else []:
    row=rowwrap.get('row',{}); meta=obj(row.get('metadata'))
    questions=row.get('questions') or []; answers=row.get('answers') or []
    if isinstance(questions,str): questions=obj(questions) if questions.startswith('{') else [questions]
    if isinstance(answers,str):
        try: answers=json.loads(answers)
        except Exception: answers=[answers]
    compact.append({
      'row_idx':rowwrap.get('row_idx'),'source':meta.get('source'),
      'context_chars':len(row.get('context') or ''),'context_prefix':(row.get('context') or '')[:1600],
      'questions_count':len(questions),'questions':questions[:5] if isinstance(questions,list) else str(questions)[:1500],
      'answers':answers[:5] if isinstance(answers,list) else str(answers)[:1500],
      'question_types':(meta.get('question_types') or [])[:10] if isinstance(meta.get('question_types'),list) else meta.get('question_types'),
      'keypoints':(meta.get('keypoints') or [])[:10] if isinstance(meta.get('keypoints'),list) else meta.get('keypoints'),
      'demo_prefix':str(meta.get('demo') or '')[:1600],
      'metadata_keys':list(meta.keys())[:40],
    })
out['compact']=compact
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_ROWS='+json.dumps(compact,ensure_ascii=False))
