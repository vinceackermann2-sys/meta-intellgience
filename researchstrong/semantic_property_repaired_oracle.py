from pathlib import Path
from collections import defaultdict,Counter
import json,tempfile
import strong_banking77 as base
import semantic_property_ingest_oracle as sw
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_property_repaired_oracle_result.json')
N=100

def main():
    td=Path(tempfile.gettempdir())/'semantic_property_repaired';td.mkdir(exist_ok=True);qp=td/'q'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);qas=list(base.load_jsonl(qp))[:N]
    repaired,report=repair.build();stats=defaultdict(lambda:Counter(n=0,covered=0,multi=0));examples=[];modes=Counter()
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};ses=rr.get('session');modes[str(rr.get('mode','missing'))]+=1
        props_world,records=sw.ingest(ses);gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});defs=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));vals,prov=sw.query_world(props_world,records,p,defs.get(p) or {});k=base.norm(g);hit=k in vals;c=stats[typ];c['n']+=1;c['covered']+=int(hit);c['multi']+=int(hit and len(prov[k])>1)
            if typ=='inferred' and hit and len(examples)<30:examples.append({'qa_id':qa.get('qa_id'),'repair_mode':rr.get('mode'),'parameter':p,'gold':g,'provenance':prov[k][:8]})
        if qi%10==0:print('REPAIRED_WORLD_BUILD',qi,rr.get('mode'),len(props_world),flush=True)
    packed={k:{'n':v['n'],'coverage':v['covered']/max(1,v['n']),'multi_provenance_rate':v['multi']/max(1,v['n'])} for k,v in stats.items()}
    result={'stage':'SWM-A strict property oracle with answer-blind release repair','split':'QA001-100 development only; QA101-400 gold sealed','repair_counts_first100':dict(modes),'repair_report':report,'results':packed,'inferred_recoveries':examples,'passes_SWM_A':packed.get('explicit',{}).get('coverage',0)>=0.85 and packed.get('inferred',{}).get('coverage',0)>=0.40,'guardrail':'Session repair uses only qa_id/source_conversation_ids and public upstream conversation files. Historical semantic ingest remains query-independent. Gold is scoring only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_PROPERTY_REPAIRED_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
