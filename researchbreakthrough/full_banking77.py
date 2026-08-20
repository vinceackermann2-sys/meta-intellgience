import json, os, re, urllib.request, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT=Path('researchbreakthrough/full_banking77_result.json')
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
MODEL='Qwen/Qwen2.5-0.5B-Instruct'; N_EVAL=int(os.environ.get('MEM2ACT_N','100'))

def fetch(url,path):
    if not path.exists(): urllib.request.urlretrieve(url,path)
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def norm_scalar(v):
    if isinstance(v,bool):return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(int,float)):return str(v)
    return re.sub(r'\s+',' ',str(v).strip()).casefold()
def parse_json_obj(text):
    text=re.sub(r'^```(?:json)?\s*','',text.strip(),flags=re.I);text=re.sub(r'\s*```$','',text)
    for m in re.finditer(r'\{',text):
        s=m.start();depth=0
        for i in range(s,len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:
                    try:
                        z=json.loads(text[s:i+1]);return z if isinstance(z,dict) else {}
                    except Exception:break
    return {}
def parse_args(tc):
    if not isinstance(tc,dict):return {}
    f=tc.get('function') or {};a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:
            z=json.loads(a);return z if isinstance(z,dict) else {}
        except Exception:return {}
    return {}
def tool_name(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {};return str((f or {}).get('name','')) if isinstance(f,dict) else ''
def flat_turn(t):
    c=t.get('content','');c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
    return f"{t.get('role','')}: {c}".strip()
def arg_metrics(pred,gold):
    correct=sum(1 for k,v in gold.items() if k in pred and norm_scalar(pred[k])==norm_scalar(v));p=correct/max(1,len(pred));r=correct/max(1,len(gold));f=2*p*r/max(1e-12,p+r)
    return correct,len(gold),p,r,f,int(correct==len(gold) and len(pred)==len(gold))

def build_session_map(conv_path):
    sessions=[];by_source=defaultdict(list)
    for row in load_jsonl(conv_path):
        i=len(sessions);sessions.append(row)
        for sid in row.get('original_conversation_ids') or []:by_source[str(sid)].append(i)
    return sessions,by_source
def find_session(qa,sessions,by_source):
    ids=[str(x) for x in qa.get('source_conversation_ids') or []];cand=None
    for sid in ids:
        z=set(by_source.get(sid,[]));cand=z if cand is None else cand&z
    if cand:return sessions[min(cand)]
    u=set()
    for sid in ids:u.update(by_source.get(sid,[]))
    return sessions[min(u)] if u else None

def compile_memory(qa,session,enc):
    turns=session.get('turns') or [];texts=[flat_turn(t) for t in turns]
    schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});target_tool=str(schema.get('name',''))
    # Structured action events are exact, reversible records. Keep provenance/turn position; do not collapse by recency.
    events=[]
    for i,t in enumerate(turns):
        for tc in (t.get('tool_calls') or []):
            a=parse_args(tc)
            if a:events.append({'turn':i,'tool':tool_name(tc),'args':a})
    # Embed raw episodic turns once and all typed event summaries once.
    tev=[f"tool {e['tool']} arguments "+' '.join(f'{k}={v}' for k,v in e['args'].items()) for e in events]
    alltxt=texts+tev
    E=enc.encode(alltxt,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if alltxt else np.zeros((0,384),np.float32)
    turnE=E[:len(texts)];eventE=E[len(texts):]
    query=str(qa.get('query',''))
    compiled={'target_tool':target_tool,'slots':{},'global_evidence':[]}
    # Two globally relevant text memories preserve free-form preferences not tied to one slot.
    if len(turnE):
        qv=enc.encode([query+' '+str(schema.get('description',''))],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
        ids=np.argsort(-(turnE@qv))[:min(2,len(turnE))];compiled['global_evidence']=[{'turn':int(i),'text':texts[int(i)]} for i in sorted(ids)]
    for p,d in props.items():
        desc=str((d or {}).get('description',''));typ=str((d or {}).get('type',''))
        pq=f"current request: {query}. target tool: {target_tool}. required parameter {p} ({typ}): {desc}"
        qv=enc.encode([pq],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
        same=[]
        for e in events:
            if p in e['args']:same.append({'turn':e['turn'],'tool':e['tool'],'value':e['args'][p]})
        # Do not trust latest same-slot blindly; expose up to 4 candidates with provenance.
        same=same[-4:]
        event_candidates=[]
        if len(eventE):
            ids=np.argsort(-(eventE@qv))[:min(3,len(events))]
            event_candidates=[events[int(i)] for i in ids]
        text_candidates=[]
        if len(turnE):
            ids=np.argsort(-(turnE@qv))[:min(2,len(turns))]
            text_candidates=[{'turn':int(i),'text':texts[int(i)]} for i in ids]
        compiled['slots'][p]={'type':typ,'description':desc,'same_name_history':same,'semantic_tool_events':event_candidates,'semantic_text':text_candidates}
    return compiled

def generate_args(qa,compiled,tok,lm):
    schema=qa.get('target_tool_schema') or {};required=((schema.get('parameters') or {}).get('required') or [])
    prompt=(
      'You are an execution-state compiler. Fill the target tool arguments from CURRENT REQUEST, TOOL SCHEMA, and COMPILED MEMORY.\n'
      'Memory has typed prior tool events and free-text episodic evidence. Same-named historical slots are candidates, NOT automatically current: use semantic scope and the request. '
      'Prefer explicit current/request evidence over older events. Use schema/default reasoning where memory does not specify a value. Infer transformations only when justified. '
      'Respect parameter types. Return ONLY the JSON argument object; do not include the tool name or explanation.\n\n'
      'CURRENT REQUEST:\n'+str(qa.get('query',''))+'\n\nTOOL SCHEMA:\n'+json.dumps(schema,ensure_ascii=False)+'\n\nREQUIRED SLOTS:\n'+json.dumps(required,ensure_ascii=False)+'\n\nCOMPILED MEMORY:\n'+json.dumps(compiled,ensure_ascii=False)
    )
    msgs=[{'role':'system','content':'Output one strict JSON object only.'},{'role':'user','content':prompt}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);inp=tok(text,return_tensors='pt',truncation=True,max_length=6144)
    with torch.inference_mode():out=lm.generate(**inp,max_new_tokens=180,do_sample=False,pad_token_id=tok.eos_token_id)
    raw=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True);return parse_json_obj(raw),raw

def main():
    td=Path(tempfile.gettempdir())/'mem2act_compiler';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch(BASE+'qa_dataset.jsonl',qp);fetch(BASE+'toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N_EVAL];sessions,by_source=build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');tok=AutoTokenizer.from_pretrained(MODEL);lm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,device_map=None).eval()
    sum_correct=sum_gold=0;f1s=[];exacts=[];missing=0;by_level=defaultdict(lambda:[0,0.0,0]);state_sizes=[];samples=[]
    for i,qa in enumerate(qas):
        s=find_session(qa,sessions,by_source)
        if s is None:missing+=1;continue
        comp=compile_memory(qa,s,enc);state_sizes.append(len(json.dumps(comp,ensure_ascii=False)));pred,raw=generate_args(qa,comp,tok,lm);gold=((qa.get('tool_call') or {}).get('arguments') or {})
        c,n,p,r,f1,ex=arg_metrics(pred,gold);sum_correct+=c;sum_gold+=n;f1s.append(f1);exacts.append(ex);lvl=str((qa.get('complexity_metadata') or {}).get('level','?'));by_level[lvl][0]+=1;by_level[lvl][1]+=f1;by_level[lvl][2]+=ex
        if i<3:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'compiled_chars':state_sizes[-1],'raw':raw[:250]})
        if (i+1)%10==0:print(f'COMPILER_PROGRESS {i+1}/{len(qas)} meanF1={np.mean(f1s):.4f}',flush=True)
    result={'tasks':len(f1s),'micro_parameter_accuracy':sum_correct/max(1,sum_gold),'macro_parameter_f1':float(np.mean(f1s)) if f1s else 0.0,'exact_argument_set':float(np.mean(exacts)) if exacts else 0.0,'missing_session':missing,'mean_compiled_memory_chars':float(np.mean(state_sizes)) if state_sizes else 0.0,'by_level':{k:{'n':v[0],'macro_f1':v[1]/max(1,v[0]),'exact':v[2]/max(1,v[0])} for k,v in sorted(by_level.items())},'samples':samples}
    out={'benchmark':'Mem2ActBench released data','subset':'first 100 release-order tasks','model':MODEL,'architecture':'typed action-event slot ledger + parameter-conditioned episodic evidence + schema state; target schema given to isolate argument grounding','result':result,'guardrail':'tool_call and grounding_info are scoring labels only. source_conversation_ids locate the containing session but are never used to filter/select turns within that session. No gold arguments tune memory selection.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_TYPED_COMPILER='+json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
