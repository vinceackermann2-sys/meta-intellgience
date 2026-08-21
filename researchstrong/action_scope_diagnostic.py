from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile
import strong_banking77 as base
import episode_scoped_router as es

OUT=Path(__file__).with_name('action_scope_diagnostic_result.json')
N=100

def has_value(ep,gold):
    ng=base.norm(gold)
    return any(base.norm(ev['value'])==ng for ev in ep['events']) or (bool(ng) and ng in base.norm(ep['text']))

def main():
    td=Path(tempfile.gettempdir())/'mab_action_scope';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);stats=defaultdict(Counter);resolved=0
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:continue
        resolved+=1;eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};target=str(schema.get('name',''));gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,gv in gold.items():
            gt=str((gi.get(p) or {}).get('type','unknown'));s=stats[gt];s['n']+=1
            tool_eps=[];slot_eps=[];both=[]
            for ep in eps:
                tools={str(ev.get('tool','')) for ev in ep['events']};slots={str(ev.get('key','')).split('.')[-1].casefold() for ev in ep['events']}
                tm=target in tools if target else False;sm=p.casefold() in slots
                if tm:tool_eps.append(ep)
                if sm:slot_eps.append(ep)
                if tm and sm:both.append(ep)
            s['has_target_tool_episode']+=int(bool(tool_eps));s['gold_in_target_tool_episode']+=int(any(has_value(e,gv) for e in tool_eps));s['has_same_slot_episode']+=int(bool(slot_eps));s['gold_in_same_slot_episode']+=int(any(has_value(e,gv) for e in slot_eps));s['has_tool_and_slot_episode']+=int(bool(both));s['gold_in_tool_and_slot_episode']+=int(any(has_value(e,gv) for e in both))
            # Deterministic structural selector: exact tool+slot, else slot, else tool.
            ranked=both+[e for e in slot_eps if e not in both]+[e for e in tool_eps if e not in both and e not in slot_eps]
            s['structural_selector_gold']+=int(bool(ranked) and has_value(ranked[0],gv))
    def pack(c):
        n=max(1,c['n']);return {'n':c['n'],**{k:c[k]/n for k in ['has_target_tool_episode','gold_in_target_tool_episode','has_same_slot_episode','gold_in_same_slot_episode','has_tool_and_slot_episode','gold_in_tool_and_slot_episode','structural_selector_gold']}}
    allc=Counter()
    for c in stats.values():allc.update(c)
    result={'stage':'Mem2Act structural action-scope diagnostic','resolved_tasks':resolved,'by_grounding':{k:pack(v) for k,v in sorted(stats.items())},'overall':pack(allc),'selector':'prefer first episode containing exact target-tool + target-slot, then target-slot, then target-tool; score whether selected episode contains gold value','guardrail':'QA001-100 development labels only for scoring. Structural routing reads target tool/schema and historical tool-call metadata; never QA source_conversation_ids. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_ACTION_SCOPE='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
