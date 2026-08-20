import json, os, re, urllib.request, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT=Path('researchstrong/strong_banking77_result.json')
BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
MODEL='Qwen/Qwen2.5-0.5B-Instruct'; N_EVAL=int(os.environ.get('MEM2ACT_N','100'))

def fetch(url,path):
    if not path.exists(): urllib.request.urlretrieve(url,path)
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def norm(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(dict,list)):return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()
def parse_json_obj(text):
    text=re.sub(r'^```(?:json)?\s*','',text.strip(),flags=re.I); text=re.sub(r'\s*```$','',text)
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
    f=tc.get('function') or {}; a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:
            z=json.loads(a);return z if isinstance(z,dict) else {}
        except Exception:return {}
    return {}
def tool_name(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {};return str((f or {}).get('name','')) if isinstance(f,dict) else ''
def flat_turn(t):
    c=t.get('content',''); c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
    return f"{t.get('role','')}: {c}".strip()
def flatten(d,prefix=''):
    out=[]
    if isinstance(d,dict):
        for k,v in d.items():
            kk=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten(v,kk))
            else:out.append((kk,v))
    elif isinstance(d,list):
        for i,v in enumerate(d):out.extend(flatten(v,f'{prefix}[{i}]'))
    return out
def arg_metrics(pred,gold):
    correct=sum(1 for k,v in gold.items() if k in pred and norm(pred[k])==norm(v));p=correct/max(1,len(pred));r=correct/max(1,len(gold));f=2*p*r/max(1e-12,p+r)
    return correct,len(gold),f,int(correct==len(gold) and len(pred)==len(gold))

def build_session_map(cp):
    sessions=[];by_source=defaultdict(list)
    for row in load_jsonl(cp):
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

def spans(text):
    vals=[]
    pats=[
      r'https?://[^\s\]\[\)\(<>"\']+', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
      r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?\b', r'\b(?:19|20)\d{2}\b',
      r'\b[A-Z][A-Z0-9._/-]{1,12}\b', r'\b\d+(?:\.\d+)?\b', r'"([^"\n]{1,80})"', r"'([^'\n]{1,80})'"
    ]
    for pat in pats:
        for m in re.finditer(pat,text):
            v=m.group(1) if m.lastindex else m.group(0); vals.append(v.strip())
    # short title-cased entities/phrases; candidates only, never auto-selected.
    for m in re.finditer(r'\b(?:[A-Z][\w.-]+)(?:\s+(?:[A-Z][\w.-]+)){0,3}\b',text):vals.append(m.group(0).strip())
    seen=set();out=[]
    for v in vals:
        z=norm(v)
        if z and z not in seen and len(str(v))<=100:seen.add(z);out.append(v)
    return out

def add_candidate(lst,seen,value,source,evidence=''):
    if isinstance(value,(dict,list)) and not value:return
    z=norm(value)
    if not z or z in seen:return
    seen.add(z);lst.append({'id':None,'value':value,'source':source,'evidence':str(evidence)[:180]})

def compile_candidates(qa,session,enc):
    turns=session.get('turns') or [];texts=[flat_turn(t) for t in turns]
    schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});target=str(schema.get('name',''));query=str(qa.get('query',''))
    events=[]
    for i,t in enumerate(turns):
        for tc in t.get('tool_calls') or []:
            a=parse_args(tc)
            if a:events.append({'turn':i,'tool':tool_name(tc),'args':a})
    tev=[f"tool {e['tool']} arguments "+' '.join(f'{k}={v}' for k,v in flatten(e['args'])) for e in events]
    alltxt=texts+tev;E=enc.encode(alltxt,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if alltxt else np.zeros((0,384),np.float32)
    turnE=E[:len(texts)];eventE=E[len(texts):]
    compiled={'target_tool':target,'slots':{}}
    for p,d0 in props.items():
        d=d0 or {};desc=str(d.get('description',''));typ=str(d.get('type','')); qtxt=f'{query} target tool {target} parameter {p} {typ} {desc}'
        qv=enc.encode([qtxt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
        cand=[];seen=set()
        if 'default' in d:add_candidate(cand,seen,d['default'],'schema_default',desc)
        enum=d.get('enum') or []
        if isinstance(enum,list):
            for v in enum:add_candidate(cand,seen,v,'schema_enum',desc)
        # Exact same-slot values are high-value candidates, but never blindly newest-wins.
        for e in events:
            for k,v in flatten(e['args']):
                if k.split('.')[-1].casefold()==p.casefold():add_candidate(cand,seen,v,'prior_same_slot',f"turn {e['turn']} tool {e['tool']}")
        # Semantically relevant tool events expose all exact scalar values as pointers.
        if len(eventE):
            for ei in np.argsort(-(eventE@qv))[:min(4,len(events))]:
                e=events[int(ei)]
                for k,v in flatten(e['args']):add_candidate(cand,seen,v,'semantic_tool_value',f"{k}; turn {e['turn']} tool {e['tool']}")
        # Relevant prose is evidence; extract copy-safe spans so IDs/acronyms/dates need not be regenerated.
        evidence=[]
        if len(turnE):
            for ti in np.argsort(-(turnE@qv))[:min(3,len(turns))]:
                tx=texts[int(ti)];evidence.append({'turn':int(ti),'text':tx[:500]})
                for v in spans(tx):add_candidate(cand,seen,v,'episodic_span',f'turn {int(ti)}')
        # Current request spans may themselves be exact executable values.
        for v in spans(query):add_candidate(cand,seen,v,'current_request_span','current request')
        # Cap without dropping explicit schema/same-slot candidates: first candidates are prioritized by construction.
        cand=cand[:24]
        for j,c in enumerate(cand):c['id']=f'{p}#{j}'
        compiled['slots'][p]={'type':typ,'description':desc,'candidates':cand,'evidence':evidence}
    return compiled

def recursively_find(obj,key):
    if not isinstance(obj,dict):return None
    if key in obj:return obj[key]
    for v in obj.values():
        if isinstance(v,dict):
            z=recursively_find(v,key)
            if z is not None:return z
    return None

def coerce(v,d):
    typ=str((d or {}).get('type','')).lower();enum=(d or {}).get('enum') or []
    if enum:
        for e in enum:
            if norm(v)==norm(e):return e
    try:
        if typ in ('int','integer'):
            if isinstance(v,str):v=re.sub(r'[^0-9+\-.]','',v)
            return int(float(v))
        if typ in ('float','number'):
            if isinstance(v,str):v=re.sub(r'[^0-9+\-.]','',v)
            return float(v)
        if typ in ('bool','boolean'):
            if isinstance(v,bool):return v
            z=norm(v)
            if z in ('true','yes','1'):return True
            if z in ('false','no','0'):return False
        if typ in ('string','str') and not isinstance(v,(dict,list)):return str(v)
    except Exception:pass
    return v

def execute_choices(choice,compiled,schema):
    props=((schema.get('parameters') or {}).get('properties') or {});out={}
    for p,d in props.items():
        x=recursively_find(choice,p)
        candidates={c['id']:c['value'] for c in (compiled['slots'].get(p,{}).get('candidates') or [])}
        val=None;got=False
        if isinstance(x,dict):
            cid=x.get('candidate') or x.get('id') or x.get('pointer')
            if cid in candidates:val=candidates[cid];got=True
            elif 'derive' in x:val=x['derive'];got=True
            elif 'value' in x:val=x['value'];got=True
        elif isinstance(x,str) and x in candidates:val=candidates[x];got=True
        elif x is not None:val=x;got=True
        if not got and 'default' in (d or {}):val=d['default'];got=True
        if got:out[p]=coerce(val,d)
    return out

def generate(qa,compiled,tok,lm):
    schema=qa.get('target_tool_schema') or {};slots={}
    for p,s in compiled['slots'].items():
        slots[p]={'type':s['type'],'description':s['description'],'candidates':s['candidates'],'evidence':s['evidence']}
    prompt=(
      'Resolve every target tool parameter. You are NOT writing the final tool call. For each schema key, choose an exact candidate pointer when a candidate is correct; otherwise derive the value from the evidence/schema. '
      'Candidate IDs copy values losslessly. Do not choose by recency alone. Prefer current semantic scope. Defaults belong to schema; explicit facts belong to memory; inferred values may require transformation. '
      'Return ONLY JSON with exactly the schema parameter names. Each value must be either {"candidate":"slot#N"} or {"derive":<value>}. Never rename parameters and never add wrappers.\n\n'
      'CURRENT REQUEST:\n'+str(qa.get('query',''))+'\n\nTARGET TOOL SCHEMA:\n'+json.dumps(schema,ensure_ascii=False)+'\n\nPOINTER MEMORY:\n'+json.dumps(slots,ensure_ascii=False)
    )
    msgs=[{'role':'system','content':'Output strict JSON only.'},{'role':'user','content':prompt}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);inp=tok(text,return_tensors='pt',truncation=True,max_length=7168)
    with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=220,do_sample=False,pad_token_id=tok.eos_token_id)
    raw=tok.decode(gen[0,inp['input_ids'].shape[1]:],skip_special_tokens=True);choice=parse_json_obj(raw);return execute_choices(choice,compiled,schema),choice,raw

def main():
    td=Path(tempfile.gettempdir())/'mem2act_pointer';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch(BASE+'qa_dataset.jsonl',qp);fetch(BASE+'toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N_EVAL];sessions,by_source=build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');tok=AutoTokenizer.from_pretrained(MODEL);lm=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map=None).eval()
    sc=sg=0;f1s=[];exacts=[];missing=0;by_level=defaultdict(lambda:[0,0.,0]);sizes=[];samples=[]
    for i,qa in enumerate(qas):
        s=find_session(qa,sessions,by_source)
        if s is None:missing+=1;continue
        comp=compile_candidates(qa,s,enc);sizes.append(len(json.dumps(comp,ensure_ascii=False)));pred,choice,raw=generate(qa,comp,tok,lm);gold=((qa.get('tool_call') or {}).get('arguments') or {})
        c,n,f,ex=arg_metrics(pred,gold);sc+=c;sg+=n;f1s.append(f);exacts.append(ex);lvl=str((qa.get('complexity_metadata') or {}).get('level','?'));by_level[lvl][0]+=1;by_level[lvl][1]+=f;by_level[lvl][2]+=ex
        if len(samples)<4:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'choice':choice,'gold':gold,'compiled_chars':sizes[-1],'raw':raw[:350]})
        if (i+1)%10==0:print(f'POINTER_PROGRESS {i+1}/{len(qas)} meanF1={np.mean(f1s):.4f}',flush=True)
    result={'tasks':len(f1s),'micro_parameter_accuracy':sc/max(1,sg),'macro_parameter_f1':float(np.mean(f1s)) if f1s else 0.,'exact_argument_set':float(np.mean(exacts)) if exacts else 0.,'missing_session':missing,'mean_compiled_memory_chars':float(np.mean(sizes)) if sizes else 0.,'by_level':{k:{'n':v[0],'macro_f1':v[1]/max(1,v[0]),'exact':v[2]/max(1,v[0])} for k,v in sorted(by_level.items())},'samples':samples}
    out={'benchmark':'Mem2ActBench released data','subset':'first 100 release-order tasks; six known release source mappings absent','model':MODEL,'architecture':'pointer-grounded typed slot memory: exact action-event/schema/request/episodic candidates + derive escape hatch + deterministic schema projection/type coercion','result':result,'comparison_reference_same94':{'flat_semantic_f1':0.009658899020601148,'typed_free_generation_f1':0.2868659848009035},'guardrail':'Gold tool arguments and grounding_info are scoring labels only; they do not construct candidates, prompts, pointers, or postprocessing.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_POINTER_COMPILER='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
