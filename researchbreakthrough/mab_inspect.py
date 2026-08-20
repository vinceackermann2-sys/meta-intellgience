import json, urllib.request, urllib.parse
from pathlib import Path

def get(url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'research-inspector/1.0'})
        with urllib.request.urlopen(req,timeout=60) as r:
            return {'status':r.status,'url':r.geturl(),'data':json.loads(r.read().decode('utf-8'))}
    except Exception as e:
        return {'error':repr(e),'url':url}
base='ai-hyz/MemoryAgentBench'
q=urllib.parse.quote(base,safe='')
out={}
out['api']=get('https://huggingface.co/api/datasets/'+base)
out['splits']=get('https://datasets-server.huggingface.co/splits?dataset='+q)
config='icl_banking77_5900shot_balance'
for split in ['train','test','validation']:
    out['first_'+split]=get('https://datasets-server.huggingface.co/first-rows?dataset='+q+'&config='+urllib.parse.quote(config)+'&split='+split)
# shrink the huge hub metadata but retain files/config clues
api=out.get('api',{}).get('data')
if isinstance(api,dict):
    keep={k:api.get(k) for k in ['id','sha','lastModified','siblings','cardData','tags'] if k in api}
    if isinstance(keep.get('siblings'),list): keep['siblings']=keep['siblings'][:200]
    out['api']['data']=keep
Path('researchbreakthrough/mab_inspection.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('MAB_INSPECTION='+json.dumps(out,ensure_ascii=False)[:30000])
