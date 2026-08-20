from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import strong_banking77 as base

OUT=Path('researchstrong/selective_slotvm_result.json')
MODEL=base.MODEL
N_EVAL=100

def required_set(s):
    schema=(s.get('qa') or {}).get('target_tool_schema') or {}
    return set(((schema.get('parameters') or {}).get('required') or []))

def add_policy_candidates(s):
    p=str(s['parameter']); d=s['def'] or {}; typ=str(d.get('type','')).lower(); desc=str(d.get('description','')).lower(); req=required_set(s); optional=p not in req; pl=p.lower().replace('_','')
    vals=[]
    if optional and typ in ('array','list'): vals.append(([], 'policy_optional_empty_array'))
    if optional and typ in ('string','str'): vals.append(('', 'policy_optional_empty_string'))
    if typ in ('bool','boolean'): vals.append((False, 'policy_boolean_false'))
    if 'offset' in pl: vals.append((0, 'policy_offset_zero'))
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl): vals.append((1, 'policy_first_page_one'))
    if pl=='index' and ('latest' in desc or 'most recent' in desc or 'starting from 0' in desc or 'start from 0' in desc): vals.append((0, 'policy_latest_index_zero'))
    cand=list(s.get('candidates') or []); seen={base.norm(c.get('value')) for c in cand}
    for v,src in vals:
        if base.norm(v) not in seen:
            cand.append({'value':v,'source':src,'evidence':'general schema/type executor policy','priority':3})
            seen.add(base.norm(v))
    cand=sorted(cand,key=lambda x:x.get('priority',5))[:32]
    for j,c in enumerate(cand): c['id']=f'C{j}'
    s['candidates']=cand
    return s

def lexical_relevance(s):
    stop={'the','a','an','of','for','to','and','or','in','on','is','are','with','from','which','that','this','you','your','get','retrieve','return','parameter','value'}
    def toks(x): return {w for w in re.findall(r'[a-z0-9]+',str(x).lower()) if len(w)>2 and w not in stop}
    q=toks(s.get('query','')); key=toks(s.get('parameter','')+' '+str((s.get('def') or {}).get('description','')))
    if not q or not key:return 0.0
    return len(q&key)/max(1,min(len(q),len(key)))

def candidate_by_source(s,*sources):
    for c in s.get('candidates') or []:
        if c.get('source') in sources:return c
    return None

def deterministic_choice(s):
    """Return (use,value,reason). Gate only high-confidence executable cases.
    Ambiguous semantic state is deliberately sent to Qwen instead of guessed.
    """
    req=required_set(s); p=s['parameter']; q=str(s.get('query','')).lower(); rel=lexical_relevance(s)
    has_same=any(c.get('source')=='prior_same_slot' for c in s.get('candidates') or [])
    # Explicit machine/schema defaults: execute only for optional slots with no same-slot memory
    # and little evidence that the current request is actively specifying this field.
    c=candidate_by_source(s,'schema_default','description_default')
    if c is not None and p not in req and not has_same and rel < 0.18:
        return True,base.coerce(c['value'],s['def']),'schema_default_low_relevance'
    # Structural empty optional collections/strings are safe only when no same-slot history and low relevance.
    c=candidate_by_source(s,'policy_optional_empty_array','policy_optional_empty_string')
    if c is not None and not has_same and rel < 0.12:
        return True,base.coerce(c['value'],s['def']),'optional_empty_low_relevance'
    # Pagination policies use explicit linguistic guards.
    c=candidate_by_source(s,'policy_offset_zero')
    if c is not None and not has_same and not any(w in q for w in ['next','previous','prev','second page','third page','offset']):
        return True,base.coerce(c['value'],s['def']),'offset_zero_no_pagination_override'
    c=candidate_by_source(s,'policy_first_page_one')
    if c is not None and not has_same and ('first' in q or not any(w in q for w in ['next','previous','prev','second','third','page 2','page 3'])):
        return True,base.coerce(c['value'],s['def']),'first_page_policy'
    c=candidate_by_source(s,'policy_latest_index_zero')
    if c is not None and not has_same and any(w in q for w in ['latest','most recent','current','newest']):
        return True,base.coerce(c['value'],s['def']),'latest_index_zero'
    # Boolean false is intentionally NOT auto-executed: implied true/false semantics are common.
    return False,None,None

def main():
    td=Path(tempfile.gettempdir())/'mem2act_selective';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N_EVAL];sessions,by=base.build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    tok=AutoTokenizer.from_pretrained(MODEL);tok.pad_token=tok.eos_token;tok.padding_side='left'
    task_slots=defaultdict(list);residual=[];det={};missing=0;gate_reasons=Counter()
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing+=1;continue
        ss=base.compile_task(qa,ses,enc)
        for s in ss:
            s['qi']=qi;add_policy_candidates(s);task_slots[qi].append(s)
            use,v,reason=deterministic_choice(s)
            if use:
                det[(qi,s['parameter'])]=v;gate_reasons[reason]+=1
            else: residual.append(s)
    lm=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map=None).eval()
    choices={};prompt_tokens=0;generated_tokens=0;B=8
    for start in range(0,len(residual),B):
        batch=residual[start:start+B];texts=[base.slot_prompt(s,tok) for s in batch]
        inp=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=4096)
        prompt_tokens += int(inp['attention_mask'].sum().item())
        with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
        L=inp['input_ids'].shape[1]
        for j,s in enumerate(batch):
            tail=gen[j,L:];generated_tokens += int((tail!=tok.pad_token_id).sum().item())
            raw=tok.decode(tail,skip_special_tokens=True);obj=base.parse_json_obj(raw);choices[(s['qi'],s['parameter'])]=(obj,raw)
        if start%64==0:print(f'SELECTIVE_PROGRESS {min(start+B,len(residual))}/{len(residual)}',flush=True)
    C=P=G=0;mac=[];exact=[];byground=defaultdict(lambda:Counter(n=0,correct=0));samples=[]
    for qi,qa in enumerate(qas):
        if qi not in task_slots:continue
        pred={};dbg={}
        for s in task_slots[qi]:
            key=(qi,s['parameter'])
            if key in det:
                pred[s['parameter']]=det[key];dbg[s['parameter']]={'mode':'deterministic','value':det[key]}
            else:
                obj,raw=choices.get(key,({},''));got,val=base.resolve_choice(obj,s)
                if got:pred[s['parameter']]=val
                dbg[s['parameter']]={'mode':'qwen','obj':obj,'raw':raw[:160]}
        gold=((qa.get('tool_call') or {}).get('arguments') or {});ground=((qa.get('tool_call') or {}).get('grounding_info') or {})
        c,p,g,f,e=base.arg_metrics(pred,gold);C+=c;P+=p;G+=g;mac.append(f);exact.append(e)
        for k,v in gold.items():
            typ=str((ground.get(k) or {}).get('type','unknown'));byground[typ]['n']+=1
            if k in pred and base.norm(pred[k])==base.norm(v):byground[typ]['correct']+=1
        if len(samples)<6:samples.append({'qa_id':qa.get('qa_id'),'pred':pred,'gold':gold,'debug':dbg})
    prec=C/max(1,P);rec=C/max(1,G);f1=2*prec*rec/max(1e-12,prec+rec)
    total_slots=len(det)+len(residual)
    result={'benchmark':'Mem2ActBench released data','subset':'QA001-100 development; resolvable public-release sessions only','model':MODEL,'architecture':'selective Memory Action IR: deterministic high-confidence execution + Qwen residual slot selector/normalizer','result':{'tasks':len(mac),'missing_session':missing,'correct_params':C,'predicted_params':P,'gold_params':G,'global_precision':prec,'global_recall':rec,'global_parameter_f1':f1,'macro_task_f1':sum(mac)/max(1,len(mac)),'exact_argument_set':sum(exact)/max(1,len(exact)),'total_schema_slots':total_slots,'deterministic_slots':len(det),'qwen_residual_slots':len(residual),'llm_call_fraction':len(residual)/max(1,total_slots),'deterministic_fraction':len(det)/max(1,total_slots),'prompt_tokens':prompt_tokens,'generated_tokens':generated_tokens,'gate_reasons':dict(gate_reasons),'by_grounding':{k:{'n':v['n'],'accuracy':v['correct']/max(1,v['n'])} for k,v in sorted(byground.items())},'samples':samples},'guardrail':'QA001-100 development labels only; QA101-400 gold remains sealed. Deterministic gates use schema/query/candidate provenance only, never grounding_info or gold arguments.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_SELECTIVE_SLOT_VM='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
