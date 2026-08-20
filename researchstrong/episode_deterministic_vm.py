from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es

OUT=Path(__file__).with_name('episode_deterministic_vm_result.json')
N=100
TOPK_VALUES=(1,2,3)


def norm(v):return base.norm(v)

def same_slot_values(ep,p):
    out=[]
    for ev in ep['events']:
        if str(ev['key']).split('.')[-1].casefold()==str(p).casefold():out.append((ev['turn'],ev['value']))
    return sorted(out,key=lambda x:x[0])

def schema_default(p,d,required,query):
    if 'default' in d:return True,d['default'],'json_default'
    ds=base.desc_defaults(d.get('description',''))
    if ds:return True,ds[0],'description_default'
    ps=es.policy_values(p,d,required)
    if ps:
        # conservative bool: don't set false if query explicitly asks the boolean concept affirmatively
        if str(d.get('type','')).lower() in ('bool','boolean'):
            desc_words={w for w in str(d.get('description','')).lower().split() if len(w)>4}
            if any(w in str(query).lower() for w in desc_words):return False,None,None
        return True,ps[0][0],ps[0][1]
    return False,None,None

def run_k(qas,sessions,by,enc,k):
    C=P=G=tasks=exact=0;ground=defaultdict(lambda:Counter(n=0,correct=0));src=Counter();samples=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:continue
        tasks+=1;eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};params=schema.get('parameters') or {};props=params.get('properties') or {};required=set(params.get('required') or []);query=str(qa.get('query',''));pred={};debug={}
        for p,d0 in props.items():
            d=d0 or {};qtxt=f"Current request: {query}. Target tool: {schema.get('name','')}. Target parameter: {p}. Meaning: {d.get('description','')}. Type: {d.get('type','')}"
            picked=es.retrieve_episodes(enc,eps,qtxt,k)
            choice=None
            for rank,ep,sim in picked:
                vals=same_slot_values(ep,p)
                if vals:
                    choice=(vals[-1][1],f'episode_same_slot_rank{rank+1}',ep['source_id']);break
            if choice is not None:
                pred[p]=base.coerce(choice[0],d);src[choice[1]]+=1;debug[p]={'source':choice[1],'episode':choice[2],'value':choice[0]};continue
            got,v,why=schema_default(p,d,required,query)
            if got:
                pred[p]=base.coerce(v,d);src[why]+=1;debug[p]={'source':why,'value':v}
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});correct=0
        for p,v in gold.items():
            gt=str((gi.get(p) or {}).get('type','unknown'));ground[gt]['n']+=1
            if p in pred and norm(pred[p])==norm(v):correct+=1;ground[gt]['correct']+=1
        C+=correct;P+=len(pred);G+=len(gold);exact+=int(correct==len(gold) and len(pred)==len(gold))
        if len(samples)<4:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':debug})
    pr=C/max(1,P);rc=C/max(1,G);f=2*pr*rc/max(1e-12,pr+rc)
    return {'top_k':k,'tasks':tasks,'correct':C,'predicted':P,'gold':G,'precision':pr,'recall':rc,'f1':f,'exact':exact/max(1,tasks),'by_grounding':{gt:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for gt,v in sorted(ground.items())},'prediction_sources':dict(src),'samples':samples}

def main():
    td=Path(tempfile.gettempdir())/'mab_episode_det';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    results=[run_k(qas,sessions,by,enc,k) for k in TOPK_VALUES]
    out={'stage':'Mem2Act deterministic episode-scoped VM ablation','architecture':'retrieve source_id episode(s) by current request+slot semantics; execute latest same-name historical value inside highest-ranked episode; otherwise deterministic schema/policy default; no trained router, no generative LLM','results':results,'guardrail':'QA001-100 development labels only for scoring. Episode retrieval uses raw memory source_id boundaries and never QA source_conversation_ids. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_EPISODE_DETERMINISTIC='+json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
