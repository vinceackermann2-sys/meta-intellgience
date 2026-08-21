from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es

OUT=Path(__file__).with_name('address_first_diagnostic_result.json')
N=100
TOP_EPISODES=2


def safe(x):
    return json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x)

def flatten(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            kk=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten(v,kk))
            else:out.append((kk,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            kk=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)):out.extend(flatten(v,kk))
            else:out.append((kk,v))
    return out

def mask_value(text,v):
    s=str(text);q=str(v)
    if not q:return s
    return re.sub(re.escape(q),'<VALUE>',s,flags=re.I)

def local(text,v,width=220):
    s=str(text);q=str(v);i=s.casefold().find(q.casefold()) if q else -1
    if i<0:return s[:2*width]
    return s[max(0,i-width):min(len(s),i+len(q)+width)]

def occurrences(ep,rank,esim):
    out=[]
    for ti,t in ep['rows']:
        role=str(t.get('role',''));content=t.get('content','');txt=safe(content)
        if isinstance(content,(dict,list)):
            for key,val in flatten(content):
                addr=f'role {role}; structured output field path {key}; record context {mask_value(txt,val)[:900]}'
                out.append({'value':val,'kind':'structured_content','address':addr,'key':key,'tool':'','turn':ti,'rank':rank,'episode_sim':esim})
        elif isinstance(content,str):
            st=content.strip()
            if st[:1] in '[{':
                try:
                    obj=json.loads(st)
                    for key,val in flatten(obj):
                        addr=f'role {role}; JSON output field path {key}; record context {mask_value(local(st,val),val)}'
                        out.append({'value':val,'kind':'json_content','address':addr,'key':key,'tool':'','turn':ti,'rank':rank,'episode_sim':esim})
                except Exception:pass
            for val in base.spans(txt):
                addr=f'role {role}; unstructured memory span; local context {mask_value(local(txt,val),val)}'
                out.append({'value':val,'kind':'text_span','address':addr,'key':'','tool':'','turn':ti,'rank':rank,'episode_sim':esim})
        for tc in t.get('tool_calls') or []:
            tool=base.tool_name(tc);args=base.parse_args(tc)
            for key,val in flatten(args):
                # Mask only the dereferenced field value; retain sibling fields as relational record context.
                ctx=mask_value(safe(args),val)
                addr=f'role {role}; tool {tool}; argument field path {key}; sibling record context {ctx[:900]}'
                out.append({'value':val,'kind':'tool_argument','address':addr,'key':key,'tool':tool,'turn':ti,'rank':rank,'episode_sim':esim})
    return out

def rank(enc,target,cands):
    if not cands:return []
    texts=[target]+[c['address'][:1200] for c in cands]
    E=enc.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    sims=E[1:]@E[0]
    order=np.argsort(-sims)
    return [(int(i),float(sims[int(i)])) for i in order]

def main():
    td=Path(tempfile.gettempdir())/'mab_address_first';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    metrics={mode:defaultdict(lambda:Counter(n=0,coverage=0,top1=0,top3=0,top5=0)) for mode in ['structured','all']};source_gold=defaultdict(Counter);examples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};tool=schema.get('name','');props=((schema.get('parameters') or {}).get('properties') or {});query=str(qa.get('query',''));gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};target=f'Current request: {query}. Target tool: {tool}. Target parameter: {p}. Meaning: {d.get("description","")}. Type: {d.get("type","")}'
            picked=es.retrieve_episodes(enc,eps,target,TOP_EPISODES);allc=[]
            for rank0,ep,sim in picked:allc.extend(occurrences(ep,rank0,sim))
            for c in allc:
                if base.norm(c['value'])==base.norm(g):source_gold[typ][c['kind']]+=1
            for mode in ['structured','all']:
                cands=[c for c in allc if mode=='all' or c['kind']!='text_span']
                m=metrics[mode][typ];m['n']+=1
                positives=[i for i,c in enumerate(cands) if base.norm(c['value'])==base.norm(g)];m['coverage']+=int(bool(positives))
                ranked=rank(enc,target,cands)
                ids=[i for i,_ in ranked]
                m['top1']+=int(bool(ids) and ids[0] in positives);m['top3']+=int(any(i in positives for i in ids[:3]));m['top5']+=int(any(i in positives for i in ids[:5]))
                if typ=='explicit' and len(examples)<10 and positives and not (ids and ids[0] in positives):
                    top=[]
                    for i,s in ranked[:4]:top.append({'score':s,'kind':cands[i]['kind'],'key':cands[i]['key'],'tool':cands[i]['tool'],'value':cands[i]['value'],'address':cands[i]['address'][:250]})
                    examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'mode':mode,'top':top,'positive_addresses':[cands[i]['address'][:250] for i in positives[:3]]})
    packed={}
    for mode,bytyp in metrics.items():
        packed[mode]={}
        for typ,c in bytyp.items():
            n=max(1,c['n']);packed[mode][typ]={'n':c['n'],'gold_address_coverage':c['coverage']/n,'top1_accuracy_all':c['top1']/n,'top3_recall_all':c['top3']/n,'top5_recall_all':c['top5']/n,'top1_given_covered':c['top1']/max(1,c['coverage'])}
    result={'stage':'Mem2Act masked address-first diagnostic','architecture':'retrieve top-2 source_id episodes -> enumerate addressable fields/spans -> MASK candidate value from its own address description -> rank target slot against address semantics -> dereference exact value only after address selection','results':packed,'gold_occurrence_source_counts':{k:dict(v) for k,v in source_gold.items()},'missing_zero_based':missing,'examples':examples,'guardrail':'QA001-100 gold is scoring only. Candidate value is masked from its own address text during ranking. QA source_conversation_ids are not used for episode selection. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_ADDRESS_FIRST='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
