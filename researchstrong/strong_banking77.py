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
            if line.strip():yield json.loads(line)
def norm(v):
    if isinstance(v,bool):return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(dict,list)):return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()
def parse_args(tc):
    if not isinstance(tc,dict):return {}
    f=tc.get('function') or {};a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:z=json.loads(a);return z if isinstance(z,dict) else {}
        except:return {}
    return {}
def tool_name(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {};return str((f or {}).get('name','')) if isinstance(f,dict) else ''
def flat_turn(t):
    c=t.get('content','');c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
    tc=t.get('tool_calls') or []
    return (f"{t.get('role','')}: {c}"+(f" TOOL_CALLS={json.dumps(tc,ensure_ascii=False)}" if tc else '')).strip()
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
def parse_json_obj(text):
    text=re.sub(r'^```(?:json)?\s*','',text.strip(),flags=re.I);text=re.sub(r'\s*```$','',text)
    for m in re.finditer(r'\{',text):
        s=m.start();depth=0;ins=False;esc=False
        for i in range(s,len(text)):
            ch=text[i]
            if ins:
                if esc:esc=False
                elif ch=='\\':esc=True
                elif ch=='"':ins=False
                continue
            if ch=='"':ins=True
            elif ch=='{':depth+=1
            elif ch=='}':
                depth-=1
                if depth==0:
                    try:z=json.loads(text[s:i+1]);return z if isinstance(z,dict) else {}
                    except:break
    return {}
def arg_metrics(pred,gold):
    correct=sum(1 for k,v in gold.items() if k in pred and norm(pred[k])==norm(v));P=len(pred);G=len(gold)
    p=correct/max(1,P);r=correct/max(1,G);f=2*p*r/max(1e-12,p+r)
    return correct,P,G,f,int(correct==G and P==G)
def build_session_map(cp):
    sessions=[];by=defaultdict(list)
    for s in load_jsonl(cp):
        i=len(sessions);sessions.append(s)
        for x in s.get('original_conversation_ids') or []:by[str(x)].append(i)
    return sessions,by
def find_session(qa,sessions,by):
    ids=[str(x) for x in qa.get('source_conversation_ids') or []];cand=None
    for x in ids:
        z=set(by.get(x,[]));cand=z if cand is None else cand&z
    if cand:return sessions[min(cand)]
    u=set()
    for x in ids:u.update(by.get(x,[]))
    return sessions[min(u)] if u else None

def literal(v):
    if not isinstance(v,str):return v
    s=v.strip().strip('"\'')
    if s.lower() in ('true','false'):return s.lower()=='true'
    if s.lower() in ('null','none'):return None
    if re.fullmatch(r'-?\d+',s):
        try:return int(s)
        except:pass
    if re.fullmatch(r'-?\d+(?:\.\d+)',s):
        try:return float(s)
        except:pass
    return s
def desc_defaults(desc):
    d=str(desc or '');vals=[]
    pats=[
      r"(?i)default\s+value\s+(?:is|=|:)\s*['\"]?([^'\".,;\)]+)",
      r"(?i)defaults?\s+to\s*['\"]?([^'\".,;\)]+)",
      r"(?i)by\s+default[^\w]+(?:the\s+)?(?:value\s+)?(?:is\s+)?['\"]?([^'\".,;\)]+)",
      r"(?i)([A-Za-z0-9._/-]+)\s*\((?:the\s+)?default\)",
      r"(?i)([A-Za-z0-9._/-]+)\s*=\s*[^|;]{0,50}\(default\)"
    ]
    for p in pats:
        for m in re.finditer(p,d):
            z=m.group(1).strip()
            if z and len(z)<80:vals.append(literal(z))
    return vals
def spans(text):
    vals=[];text=str(text)
    pats=[r'https?://[^\s\]\[\)\(<>"\']+',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?\b',r'\b(?:19|20)\d{2}\b',r'\b[A-Z][A-Z0-9._/-]{1,20}\b',r'\b\d+(?:\.\d+)?(?:h|hr|hrs|d|days|w|weeks|min|minutes)?\b',r'"([^"\n]{1,120})"',r"'([^'\n]{1,120})'"]
    for p in pats:
        for m in re.finditer(p,text):vals.append((m.group(1) if m.lastindex else m.group(0)).strip())
    for m in re.finditer(r'\b(?:[A-Z][\w.-]+)(?:\s+(?:[A-Z][\w.-]+)){0,4}\b',text):vals.append(m.group(0).strip())
    seen=set();out=[]
    for v in vals:
        z=norm(v)
        if z and z not in seen and len(str(v))<=140:seen.add(z);out.append(v)
    return out
def add(c,seen,v,source,evidence='',priority=5):
    if v is None and source!='schema_default':return
    if isinstance(v,(dict,list)) and not v:return
    z=norm(v)
    if not z or z in seen:return
    seen.add(z);c.append({'value':v,'source':source,'evidence':str(evidence)[:220],'priority':priority})
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
            if z in ('true','yes','1','on'):return True
            if z in ('false','no','0','off'):return False
        if typ in ('string','str') and not isinstance(v,(dict,list)):return str(v)
    except:pass
    return v

def compile_task(qa,session,enc):
    turns=session.get('turns') or [];texts=[flat_turn(t) for t in turns]
    events=[]
    for i,t in enumerate(turns):
        for tc in t.get('tool_calls') or []:
            a=parse_args(tc)
            if a:events.append({'turn':i,'tool':tool_name(tc),'args':a})
    event_text=[f"tool {e['tool']} arguments "+' '.join(f'{k}={v}' for k,v in flatten(e['args'])) for e in events]
    alltxt=texts+event_text
    E=enc.encode(alltxt,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if alltxt else np.zeros((0,384),np.float32)
    turnE=E[:len(texts)];eventE=E[len(texts):]
    schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});query=str(qa.get('query',''));target=str(schema.get('name',''))
    slots=[]
    for p,d0 in props.items():
        d=d0 or {};desc=str(d.get('description',''));typ=str(d.get('type',''));qtxt=f'user request {query} target tool {target} parameter {p} type {typ} meaning {desc}'
        qv=enc.encode([qtxt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
        cand=[];seen=set()
        if 'default' in d:add(cand,seen,d['default'],'schema_default',desc,0)
        for v in desc_defaults(desc):add(cand,seen,v,'description_default',desc,1)
        if isinstance(d.get('enum'),list):
            for v in d['enum']:add(cand,seen,v,'schema_enum',desc,2)
        # Same-name historical slots are exact but not assumed current.
        for e in events:
            for k,v in flatten(e['args']):
                if k.split('.')[-1].casefold()==p.casefold():add(cand,seen,v,'prior_same_slot',f"turn {e['turn']} tool {e['tool']}",2)
        # Semantic historical action values.
        if len(eventE):
            for ei in np.argsort(-(eventE@qv))[:min(5,len(events))]:
                e=events[int(ei)]
                for k,v in flatten(e['args']):add(cand,seen,v,'semantic_tool_value',f"{k}; turn {e['turn']} tool {e['tool']}",4)
        evidence=[]
        if len(turnE):
            for ti in np.argsort(-(turnE@qv))[:min(4,len(turns))]:
                tx=texts[int(ti)];evidence.append({'turn':int(ti),'text':tx[:650]})
                for v in spans(tx):add(cand,seen,v,'episodic_span',f'turn {int(ti)}',5)
        for v in spans(query):add(cand,seen,v,'current_request_span','current request',3)
        cand=sorted(cand,key=lambda x:x['priority'])[:28]
        for j,c0 in enumerate(cand):c0['id']=f'C{j}'
        slots.append({'qa':qa,'parameter':p,'def':d,'target':target,'query':query,'candidates':cand,'evidence':evidence})
    return slots

def slot_prompt(s,tok):
    p=s['parameter'];d=s['def'];compact=[{'id':c['id'],'value':c['value'],'source':c['source'],'evidence':c['evidence']} for c in s['candidates']]
    instruction=(
      'Resolve ONE tool parameter from long-term memory. Prefer exact candidate IDs whenever a candidate is semantically correct; do not regenerate exact IDs/codes/URLs/dates that already exist. '
      'Do not use recency alone: select the value in the same semantic task/topic. If the schema itself states a default and memory does not override it, use that candidate. '
      'If the parameter expects a code/symbol/pair/abbreviation/enum, canonicalize the remembered entity to the representation required by the schema. '
      'If the exact final value is not a candidate, derive it from evidence/schema. Return ONLY {"candidate":"C#"} or {"derive":<value>} or {"omit":true}.')
    body=f"CURRENT REQUEST:\n{s['query']}\n\nTARGET TOOL: {s['target']}\nPARAMETER: {p}\nSCHEMA: {json.dumps(d,ensure_ascii=False)}\n\nCANDIDATES:\n{json.dumps(compact,ensure_ascii=False)}\n\nRELEVANT MEMORY EVIDENCE:\n{json.dumps(s['evidence'],ensure_ascii=False)}"
    msgs=[{'role':'system','content':instruction},{'role':'user','content':body}]
    return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
def resolve_choice(obj,s):
    if not isinstance(obj,dict):return False,None
    if obj.get('omit') is True:return False,None
    cid=obj.get('candidate') or obj.get('id') or obj.get('pointer')
    by={c['id']:c['value'] for c in s['candidates']}
    if cid in by:return True,coerce(by[cid],s['def'])
    if 'derive' in obj:return True,coerce(obj['derive'],s['def'])
    if 'value' in obj:return True,coerce(obj['value'],s['def'])
    return False,None

def main():
    td=Path(tempfile.gettempdir())/'mem2act_slotvm';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch(BASE+'qa_dataset.jsonl',qp);fetch(BASE+'toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N_EVAL];sessions,by=build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');tok=AutoTokenizer.from_pretrained(MODEL);tok.pad_token=tok.eos_token;tok.padding_side='left';lm=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map=None).eval()
    task_slots=defaultdict(list);flat=[];missing=0
    for qi,qa in enumerate(qas):
        s=find_session(qa,sessions,by)
        if s is None:missing+=1;continue
        ss=compile_task(qa,s,enc)
        for x in ss:x['qi']=qi;flat.append(x);task_slots[qi].append(x)
    # Batched per-slot inference keeps the operation atomic without multiplying wall-clock excessively.
    choices={};B=4
    for start in range(0,len(flat),B):
        batch=flat[start:start+B];texts=[slot_prompt(s,tok) for s in batch];inp=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=4096)
        with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
        L=inp['input_ids'].shape[1]
        for j,s in enumerate(batch):
            raw=tok.decode(gen[j,L:],skip_special_tokens=True);obj=parse_json_obj(raw);choices[(s['qi'],s['parameter'])]=(obj,raw)
        if start%40==0:print(f'SLOTVM_PROGRESS {min(start+B,len(flat))}/{len(flat)}',flush=True)
    C=P=G=0;mac=[];exact=[];bylvl=defaultdict(lambda:[0,0.,0]);samples=[];compiled_chars=[]
    for qi,qa in enumerate(qas):
        if qi not in task_slots:continue
        pred={};debug={}
        for s in task_slots[qi]:
            obj,raw=choices.get((qi,s['parameter']),({},''));got,val=resolve_choice(obj,s)
            if not got and 'default' in s['def']:got=True;val=coerce(s['def']['default'],s['def'])
            if got:pred[s['parameter']]=val
            debug[s['parameter']]={'choice':obj,'raw':raw[:160]}
            compiled_chars.append(len(json.dumps({'c':s['candidates'],'e':s['evidence']},ensure_ascii=False)))
        gold=((qa.get('tool_call') or {}).get('arguments') or {});c,p,g,f,ex=arg_metrics(pred,gold);C+=c;P+=p;G+=g;mac.append(f);exact.append(ex);lvl=str((qa.get('complexity_metadata') or {}).get('level','?'));bylvl[lvl][0]+=1;bylvl[lvl][1]+=f;bylvl[lvl][2]+=ex
        if len(samples)<4:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':debug})
    prec=C/max(1,P);rec=C/max(1,G);micro=2*prec*rec/max(1e-12,prec+rec)
    result={'tasks':len(mac),'correct_params':C,'predicted_params':P,'gold_params':G,'global_parameter_precision':prec,'global_parameter_recall':rec,'global_parameter_f1':micro,'macro_task_f1':float(np.mean(mac)) if mac else 0.,'exact_argument_set':float(np.mean(exact)) if exact else 0.,'missing_session':missing,'mean_compiled_slot_chars':float(np.mean(compiled_chars)) if compiled_chars else 0.,'by_level':{k:{'n':v[0],'macro_f1':v[1]/max(1,v[0]),'exact':v[2]/max(1,v[0])} for k,v in sorted(bylvl.items())},'samples':samples}
    out={'benchmark':'Mem2ActBench development surface QA001-100','model':MODEL,'architecture':'per-slot pointer VM: operation-specific candidate lattice + schema default parsing + atomic 0.5B resolver + deterministic projection/coercion','result':result,'same94_reference':{'flat_macro_f1':0.009658899020601148,'typed_free_generation_macro_f1':0.2868659848009035},'protocol_guardrail':'Development result only. Gold arguments score predictions but never construct candidate memory or prompts. QA101-400 held-out labels remain unopened.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('MEM2ACT_SLOT_VM='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
