from pathlib import Path
from collections import defaultdict
import json, urllib.request, tempfile

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/'
OUT=Path('researchstrong/mem2act_repaired_sessions_report.json')


def stream(url):
    with urllib.request.urlopen(url,timeout=240) as r:
        for raw in r:
            if raw.strip(): yield json.loads(raw.decode('utf-8'))


def qa_projection(q):
    # Deliberately retain only fields legal for context resolution.
    return {'qa_id':q.get('qa_id'),'source_conversation_ids':[str(x) for x in (q.get('source_conversation_ids') or [])]}


def released_index():
    sessions=[];by=defaultdict(list)
    for s in stream(BASE+'Mem2ActBench/toolmem_conversation.jsonl'):
        i=len(sessions);sessions.append(s)
        for x in s.get('original_conversation_ids') or []:by[str(x)].append(i)
    return sessions,by


def resolve_released(ids,sessions,by):
    if not ids:return None
    cand=None
    for x in ids:
        z=set(by.get(x,[]));cand=z if cand is None else cand&z
    if cand:return sessions[min(cand)]
    union=set()
    for x in ids:union.update(by.get(x,[]))
    return sessions[min(union)] if union else None


def upstream_rows():
    wanted={}
    # These are public provenance sources used by the released construction pipeline.
    for src in ['toolace_formatted_conversations.jsonl','bfcl_formatted_conversations.jsonl','oasst1_formatted_conversations.jsonl']:
        for row in stream(BASE+src):
            rid=str(row.get('id',''))
            if rid:wanted[rid]=(src,row)
    return wanted


def normalize_upstream(source_id,src,row):
    turns=row.get('conversation_history') or row.get('turns') or []
    return {
      'session_id':f'repaired::{source_id}',
      'original_conversation_ids':[source_id],
      'turns':turns,
      'turn_count':len(turns),
      'has_tool_calls':any(bool(t.get('tool_calls')) for t in turns if isinstance(t,dict)),
      'repair_source':src,
      'repair_mode':'exact_upstream_conversation'
    }


def build():
    # IMPORTANT: load raw QA rows but immediately project away labels. No tool_call,
    # grounding_info, evolution_chain, answer, or gold value is retained or inspected.
    qproj=[]
    for q in stream(BASE+'Mem2ActBench/qa_dataset.jsonl'):
        qproj.append(qa_projection(q))
    sessions,by=released_index();up=upstream_rows()
    repaired={};stats=defaultdict(int);missing_upstream=[]
    for q in qproj:
        qid=q['qa_id'];ids=q['source_conversation_ids']
        s=resolve_released(ids,sessions,by)
        if s is not None:
            repaired[qid]={'mode':'released','session':s};stats['released']+=1;continue
        if not ids:
            empty={'session_id':f'empty::{qid}','original_conversation_ids':[],'turns':[],'turn_count':0,'has_tool_calls':False,'repair_mode':'empty_no_source_id'}
            repaired[qid]={'mode':'empty_no_source_id','session':empty};stats['empty_no_source_id']+=1;continue
        rows=[]
        for sid in ids:
            if sid not in up:missing_upstream.append({'qa_id':qid,'source_id':sid});continue
            src,row=up[sid];rows.append(normalize_upstream(sid,src,row))
        if rows:
            # Preserve source-id order from the QA metadata; concatenate each original
            # conversation's turn order without inventing cross-source timestamps.
            turns=[];sources=[]
            for r in rows:
                turns.extend(r['turns']);sources.extend(r['original_conversation_ids'])
            merged={'session_id':f'repaired::{qid}','original_conversation_ids':sources,'turns':turns,'turn_count':len(turns),'has_tool_calls':any(bool(t.get('tool_calls')) for t in turns if isinstance(t,dict)),'repair_source':[r['repair_source'] for r in rows],'repair_mode':'upstream_fallback'}
            repaired[qid]={'mode':'upstream_fallback','session':merged};stats['upstream_fallback']+=1
        else:
            repaired[qid]={'mode':'unresolved','session':None};stats['unresolved']+=1
    report={'stage':'Mem2Act answer-blind repaired session loader','qa_count':len(qproj),'released_session_count':len(sessions),'counts':dict(stats),'missing_upstream':missing_upstream,'all_qas_accounted_for':len(repaired)==len(qproj),'all_contexts_resolved_or_explicit_empty':all(v['session'] is not None for v in repaired.values()),'guardrail':'Context repair uses only qa_id/source_conversation_ids and public conversation files. It never inspects tool_call arguments, grounding_info, evolution_chain, gold answers, or labels.'}
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False));print('MEM2ACT_REPAIRED_SESSIONS='+json.dumps(report,ensure_ascii=False),flush=True)
    return repaired,report

if __name__=='__main__':build()
