from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base

OUT=Path(__file__).with_name('episode_scope_oracle_result.json')
N=100


def flatten_values(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(dict,list)):out.extend(flatten_values(v))
            else:out.append(v)
    elif isinstance(x,list):
        for v in x:
            if isinstance(v,(dict,list)):out.extend(flatten_values(v))
            else:out.append(v)
    return out


def turn_text(t):
    c=t.get('content','');c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
    tc=t.get('tool_calls') or []
    return f"{t.get('role','')}: {c}"+(f" TOOL_CALLS={json.dumps(tc,ensure_ascii=False)}" if tc else '')


def episodes(session):
    by=defaultdict(list);order=[]
    for i,t in enumerate(session.get('turns') or []):
        sid=str(t.get('source_id') or f'unknown_turn_{i}')
        if sid not in by:order.append(sid)
        by[sid].append((i,t))
    out=[]
    for sid in order:
        turns=by[sid];texts=[turn_text(t) for _,t in turns];vals=[]
        for _,t in turns:
            for tc in t.get('tool_calls') or []:vals.extend(flatten_values(base.parse_args(tc)))
        out.append({'source_id':sid,'text':'\n'.join(texts),'values':vals,'turns':[i for i,_ in turns]})
    return out


def contains(ep,gold):
    ng=base.norm(gold)
    if not ng:return False
    if ng in base.norm(ep['text']):return True
    return any(base.norm(v)==ng for v in ep['values'])


def main():
    td=Path(tempfile.gettempdir())/'mab_episode_oracle';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    stats=defaultdict(lambda:Counter(n=0,any_session=0,top1=0,top2=0,top3=0));resolved=0;missing=[];samples=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        resolved+=1;eps=episodes(ses)
        etxt=[e['text'][:6000] for e in eps]
        EE=enc.encode(etxt,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if etxt else np.zeros((0,384),np.float32)
        schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,gv in gold.items():
            d=props.get(p) or {};gt=str((ground.get(p) or {}).get('type','unknown'));s=stats[gt];s['n']+=1
            hits=[contains(e,gv) for e in eps];s['any_session']+=int(any(hits))
            if len(eps):
                qtxt=f"Current request: {qa.get('query','')} Target tool: {schema.get('name','')} Target parameter: {p}. Meaning: {d.get('description','')}"
                qv=enc.encode([qtxt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0];order=np.argsort(-(EE@qv))
                for k in (1,2,3):s[f'top{k}']+=int(any(hits[int(j)] for j in order[:min(k,len(order))]))
                if len(samples)<8 and any(hits):
                    gold_rank=min([r for r,j in enumerate(order) if hits[int(j)]],default=999)+1
                    samples.append({'qa_id':qa.get('qa_id'),'slot':p,'grounding':gt,'gold_episode_rank':gold_rank,'episode_count':len(eps),'top_source_ids':[eps[int(j)]['source_id'] for j in order[:3]]})
    def pack(c):
        n=max(1,c['n']);return {'n':c['n'],'any_session':c['any_session']/n,'top1':c['top1']/n,'top2':c['top2']/n,'top3':c['top3']/n}
    allc=Counter()
    for c in stats.values():allc.update(c)
    result={'stage':'Mem2Act source-ID episode scope oracle','resolved_tasks':resolved,'missing_zero_based':missing,'by_grounding':{k:pack(v) for k,v in sorted(stats.items())},'overall':pack(allc),'samples':samples,'method':'Group released turns by their raw source_id, retrieve episodes with all-MiniLM-L6-v2 using current request + target parameter semantics, then measure whether the exact gold value occurs in top-k episode memory. Gold is scoring only.','guardrail':'QA001-100 development labels only. source_id is part of released memory turns and is not compared against QA source_conversation_ids. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_EPISODE_SCOPE_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
