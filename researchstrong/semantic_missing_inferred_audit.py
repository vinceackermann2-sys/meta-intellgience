from pathlib import Path
import json,tempfile
import strong_banking77 as base
import semantic_property_ingest_oracle as sw
import episode_scoped_router as es

OUT=Path(__file__).with_name('semantic_missing_inferred_audit_result.json')
N=100

def user_turns(session,limit=18):
    rows=[]
    if session is None:return rows
    for ep in es.episodes(session):
        for ti,t in ep['rows']:
            if str(t.get('role',''))=='user':
                c=t.get('content','')
                if isinstance(c,(dict,list)):c=json.dumps(c,ensure_ascii=False)
                rows.append({'turn':ti,'source_id':ep.get('source_id'),'text':str(c)[:500]})
    return rows[-limit:]

def main():
    td=Path(tempfile.gettempdir())/'semantic_missing_audit';td.mkdir(exist_ok=True)
    qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);miss=[];total=0;hit=0
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by);props_world,records=sw.ingest(ses)
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});defs=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        for p,g in gold.items():
            if str((gi.get(p) or {}).get('type','unknown'))!='inferred':continue
            total+=1;vals,prov=sw.query_world(props_world,records,p,defs.get(p) or {});ok=base.norm(g) in vals;hit+=int(ok)
            if not ok:
                miss.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'query':qa.get('query',''),'target_tool':(qa.get('target_tool_schema') or {}).get('name',''),'schema':defs.get(p) or {},'recent_user_turns':user_turns(ses)})
    result={'stage':'development-only missing inferred semantic-property audit','total_inferred':total,'covered':hit,'missing':len(miss),'cases':miss,'guardrail':'QA001-100 development only. No QA101-400 gold read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_MISSING_INFERRED_AUDIT='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
