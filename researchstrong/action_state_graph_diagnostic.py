from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile, re
import numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es

OUT=Path(__file__).with_name('action_state_graph_diagnostic_result.json')
N=100


def ntool(x):
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def leaf(k):
    return str(k).split('.')[-1].casefold()

def score_choice(choice,gold):
    return choice is not None and base.norm(choice)==base.norm(gold)

def latest(events):
    if not events:return None
    return sorted(events,key=lambda e:e['turn'])[-1]['value']

def events_matching(eps,p,target_tool=None,episode_ids=None):
    out=[]
    ids=set(episode_ids or []) if episode_ids is not None else None
    tt=ntool(target_tool) if target_tool is not None else None
    for ep in eps:
        if ids is not None and ep['source_id'] not in ids:continue
        for ev in ep['events']:
            if leaf(ev['key'])!=str(p).casefold():continue
            if tt is not None and ntool(ev['tool'])!=tt:continue
            out.append(ev)
    return out

def has_gold(events,gold):
    return any(base.norm(e['value'])==base.norm(gold) for e in events)

def main():
    td=Path(tempfile.gettempdir())/'mab_action_state_graph';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    methods=['latest_same_slot_anywhere','latest_same_tool_same_slot_anywhere','latest_same_slot_top1_episode','latest_same_tool_same_slot_top1_episode','latest_same_slot_top2_episode','latest_same_tool_same_slot_top2_episode']
    stats={m:defaultdict(lambda:Counter(n=0,correct=0,coverage=0)) for m in methods}
    oracle={k:defaultdict(lambda:Counter(n=0,has=0)) for k in ['same_slot_anywhere','same_tool_same_slot_anywhere','same_slot_top1','same_tool_same_slot_top1','same_slot_top2','same_tool_same_slot_top2']}
    conflict=defaultdict(lambda:Counter(n=0,conflict=0,tool_history=0))
    examples=[];tasks=0;missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        tasks+=1;eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};target_tool=schema.get('name','');props=((schema.get('parameters') or {}).get('properties') or {});query=str(qa.get('query',''))
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            d=props.get(p) or {}
            qtxt=f"Current request: {query}. Target tool: {target_tool}. Target parameter: {p}. Meaning: {d.get('description','')}. Type: {d.get('type','')}"
            picked1=es.retrieve_episodes(enc,eps,qtxt,1);ids1=[e['source_id'] for _,e,_ in picked1]
            picked2=es.retrieve_episodes(enc,eps,qtxt,2);ids2=[e['source_id'] for _,e,_ in picked2]
            a_all=events_matching(eps,p)
            t_all=events_matching(eps,p,target_tool)
            a1=events_matching(eps,p,episode_ids=ids1)
            t1=events_matching(eps,p,target_tool,ids1)
            a2=events_matching(eps,p,episode_ids=ids2)
            t2=events_matching(eps,p,target_tool,ids2)
            sets={'same_slot_anywhere':a_all,'same_tool_same_slot_anywhere':t_all,'same_slot_top1':a1,'same_tool_same_slot_top1':t1,'same_slot_top2':a2,'same_tool_same_slot_top2':t2}
            for name,evs in sets.items():
                oracle[name][typ]['n']+=1;oracle[name][typ]['has']+=int(has_gold(evs,g))
            choices={'latest_same_slot_anywhere':latest(a_all),'latest_same_tool_same_slot_anywhere':latest(t_all),'latest_same_slot_top1_episode':latest(a1),'latest_same_tool_same_slot_top1_episode':latest(t1),'latest_same_slot_top2_episode':latest(a2),'latest_same_tool_same_slot_top2_episode':latest(t2)}
            for name,ch in choices.items():
                st=stats[name][typ];st['n']+=1;st['coverage']+=int(ch is not None);st['correct']+=int(score_choice(ch,g))
            vals={base.norm(e['value']) for e in a_all};conflict[typ]['n']+=1;conflict[typ]['conflict']+=int(len(vals)>1);conflict[typ]['tool_history']+=int(bool(t_all))
            if len(examples)<12 and typ=='explicit' and (latest(a_all) is not None) and not score_choice(latest(a_all),g):
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'target_tool':target_tool,'gold':g,'latest_same_slot':latest(a_all),'latest_same_tool_same_slot':latest(t_all),'top1_episode_ids':ids1,'top2_episode_ids':ids2,'same_slot_values':[{'turn':e['turn'],'tool':e['tool'],'value':e['value']} for e in a_all[-8:]]})
    def pack(d):
        return {m:{typ:{'n':c['n'],'coverage':c.get('coverage',0)/max(1,c['n']),'accuracy_all':c.get('correct',0)/max(1,c['n']),'accuracy_when_covered':c.get('correct',0)/max(1,c.get('coverage',0))} for typ,c in bytyp.items()} for m,bytyp in d.items()}
    op={m:{typ:{'n':c['n'],'has_gold':c['has']/max(1,c['n'])} for typ,c in bytyp.items()} for m,bytyp in oracle.items()}
    cf={typ:{'n':c['n'],'multi_value_same_slot_fraction':c['conflict']/max(1,c['n']),'exact_target_tool_history_fraction':c['tool_history']/max(1,c['n'])} for typ,c in conflict.items()}
    result={'stage':'Mem2Act action-state graph structural diagnostic','tasks':tasks,'missing_zero_based':missing,'methods':pack(stats),'oracle_coverage':op,'conflict_diagnostic':cf,'examples':examples,'guardrail':'QA001-100 development labels are used only to score structural rules/oracle coverage. Episode selection uses released source_id boundaries and current query/schema only. QA source_conversation_ids and QA101-400 gold remain unused.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_ACTION_STATE_GRAPH='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
