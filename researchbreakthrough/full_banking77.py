import json, os, re, urllib.request, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT=Path('researchbreakthrough/full_banking77_result.json')
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
QA_URL=BASE+'qa_dataset.jsonl'
CONV_URL=BASE+'toolmem_conversation.jsonl'
MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_EVAL=int(os.environ.get('MEM2ACT_N','100'))
TOPK=6

def fetch(url,path):
    if not path.exists(): urllib.request.urlretrieve(url,path)

def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def flatten_turn(turn):
    role=str(turn.get('role',''))
    content=turn.get('content','')
    if isinstance(content,(dict,list)): content=json.dumps(content,ensure_ascii=False)
    tc=turn.get('tool_calls')
    extra=''
    if tc: extra=' TOOL_CALLS '+json.dumps(tc,ensure_ascii=False)
    return f"{role}: {content}{extra}".strip()

def norm_scalar(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None: return 'null'
    if isinstance(v,(int,float)): return str(v)
    return re.sub(r'\s+',' ',str(v).strip()).casefold()

def parse_json_obj(text):
    text=text.strip()
    text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I); text=re.sub(r'\s*```$','',text)
    for m in re.finditer(r'\{',text):
        s=m.start(); depth=0
        for i in range(s,len(text)):
            if text[i]=='{': depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:
                    try:
                        z=json.loads(text[s:i+1])
                        if isinstance(z,dict): return z
                    except Exception: break
    return {}

def arg_metrics(pred,gold):
    keys=set(gold)
    correct=sum(1 for k in keys if k in pred and norm_scalar(pred[k])==norm_scalar(gold[k]))
    precision=correct/max(1,len(pred)); recall=correct/max(1,len(gold)); f1=2*precision*recall/max(1e-12,precision+recall)
    return correct,len(gold),precision,recall,f1,int(correct==len(gold) and len(pred)==len(gold))

def build_session_map(conv_path):
    sessions=[]; by_source=defaultdict(list)
    for row in load_jsonl(conv_path):
        i=len(sessions); sessions.append(row)
        for sid in row.get('original_conversation_ids') or []: by_source[str(sid)].append(i)
    return sessions,by_source

def find_session(qa,sessions,by_source):
    ids=[str(x) for x in qa.get('source_conversation_ids') or []]
    cand=None
    for sid in ids:
        z=set(by_source.get(sid,[])); cand=z if cand is None else cand&z
    if cand: return sessions[min(cand)]
    union=set()
    for sid in ids: union.update(by_source.get(sid,[]))
    return sessions[min(union)] if union else None

def evidence_for(qa,session,encoder):
    turns=[flatten_turn(t) for t in (session.get('turns') or [])]
    if not turns: return []
    schema=qa.get('target_tool_schema') or {}; props=((schema.get('parameters') or {}).get('properties') or {})
    base=str(qa.get('query',''))+' tool '+str(schema.get('name',''))+' '+str(schema.get('description',''))
    queries=[base]+[base+f' parameter {p}: '+str((d or {}).get('description','')) for p,d in props.items()]
    E=encoder.encode(turns,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    Q=encoder.encode(queries,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
    score=(Q@E.T).max(0)+np.linspace(0,0.03,len(turns),dtype=np.float32)
    idx=np.argsort(-score)[:min(TOPK,len(turns))]
    return [turns[i] for i in sorted(idx)]

def generate_args(qa,evidence,tok,lm):
    schema=qa.get('target_tool_schema') or {}; params=schema.get('parameters') or {}
    prompt=(
      'You are filling arguments for one tool call from persistent user memory. '
      'Use the memory evidence, current request, and schema. Resolve newer preferences over older conflicting ones. '
      'Infer values only when justified by the request/schema; use obvious schema defaults when required. '
      'Return ONLY one JSON object containing argument names and values.\n\n'
      'CURRENT REQUEST:\n'+str(qa.get('query',''))+'\n\nTOOL:\n'+json.dumps({'name':schema.get('name'),'description':schema.get('description'),'parameters':params},ensure_ascii=False)+'\n\n'
      'MEMORY EVIDENCE:\n'+'\n'.join(evidence)
    )
    msgs=[{'role':'system','content':'Output strict JSON only.'},{'role':'user','content':prompt}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(text,return_tensors='pt',truncation=True,max_length=4096)
    with torch.inference_mode(): out=lm.generate(**inp,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
    gen=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
    return parse_json_obj(gen),gen

def main():
    td=Path(tempfile.gettempdir())/'mem2act_gate'; td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl'; cp=td/'conv.jsonl'; fetch(QA_URL,qp); fetch(CONV_URL,cp)
    qas=list(load_jsonl(qp))[:N_EVAL]; sessions,by_source=build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    tok=AutoTokenizer.from_pretrained(MODEL)
    lm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,device_map=None).eval()
    sum_correct=sum_gold=0; f1s=[]; exacts=[]; missing=0; by_level=defaultdict(lambda:[0,0.0,0]); samples=[]
    for i,qa in enumerate(qas):
        s=find_session(qa,sessions,by_source)
        if s is None: missing+=1; continue
        ev=evidence_for(qa,s,enc); pred,raw=generate_args(qa,ev,tok,lm)
        gold=((qa.get('tool_call') or {}).get('arguments') or {})
        c,n,p,r,f1,ex=arg_metrics(pred,gold); sum_correct+=c; sum_gold+=n; f1s.append(f1); exacts.append(ex)
        lvl=str((qa.get('complexity_metadata') or {}).get('level','?')); by_level[lvl][0]+=1; by_level[lvl][1]+=f1; by_level[lvl][2]+=ex
        if i<3: samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'raw':raw[:300],'evidence_n':len(ev)})
        if (i+1)%10==0: print(f'MEM2ACT_PROGRESS {i+1}/{len(qas)} meanF1={np.mean(f1s):.4f}',flush=True)
    result={'tasks':len(f1s),'micro_parameter_accuracy':sum_correct/max(1,sum_gold),'macro_parameter_f1':float(np.mean(f1s)) if f1s else 0.0,'exact_argument_set':float(np.mean(exacts)) if exacts else 0.0,'missing_session':missing,'by_level':{k:{'n':v[0],'macro_f1':v[1]/max(1,v[0]),'exact':v[2]/max(1,v[0])} for k,v in sorted(by_level.items())},'samples':samples}
    out={'benchmark':'Mem2ActBench released data','subset':'first 100 tasks in release order','N':N_EVAL,'model':MODEL,'memory':'parameter-conditioned semantic compiler: top-6 session turns + weak recency prior','result':result,'guardrail':'tool_call and grounding_info are labels only. Retrieval and generation use session history, current query, and target tool schema. No benchmark answers are used to tune parameters.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)); print('MEM2ACT_GATE='+json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
