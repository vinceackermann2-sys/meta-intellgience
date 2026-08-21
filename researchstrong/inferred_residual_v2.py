from pathlib import Path
from collections import defaultdict
import json,tempfile,numpy as np,torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM
import strong_banking77 as base
import episode_scoped_router as es
import joint_schema_alignment as js
import gatea_source_selector as ss
from gatea_pairwise_cv import CachedEncoder

OUT=Path(__file__).with_name('inferred_residual_v2_result.json')
MODELS=['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct']
N=100;JOINT=(0.75,0.0,2)

def unwrap_target(z,p):
    if isinstance(z,dict):
        # Prefer explicit scalar value; if it is itself a mapping, search the target slot inside it.
        if 'value' in z:
            v=z['value']
            if isinstance(v,dict):
                if p in v:return v[p]
                # case-insensitive target key fallback
                for k,x in v.items():
                    if str(k).casefold()==str(p).casefold():return x
            elif not isinstance(v,(dict,list)):return v
        if p in z:return z[p]
        for k,x in z.items():
            if str(k).casefold()==str(p).casefold():return x
        # Recursively inspect small nested dicts/lists; this is output parsing only, not answer search.
        for x in z.values():
            y=unwrap_target(x,p)
            if y is not None:return y
    elif isinstance(z,list):
        for x in z:
            y=unwrap_target(x,p)
            if y is not None:return y
    return None

def parse_value(text,p):
    s=text.strip()
    candidates=[s]
    a=s.find('{');b=s.rfind('}')
    if a>=0 and b>a:candidates.append(s[a:b+1])
    for q in candidates:
        try:
            z=json.loads(q);y=unwrap_target(z,p)
            if y is not None:return y
            if not isinstance(z,(dict,list)):return z
        except:pass
    # fenced JSON cleanup
    t=s.strip('` \n')
    if t.lower().startswith('json'):t=t[4:].lstrip()
    try:
        z=json.loads(t);y=unwrap_target(z,p)
        if y is not None:return y
    except:pass
    return s.strip('`"\' \n')

def evidence(enc,qa,ses,p,d,jp):
    sch=qa.get('target_tool_schema') or {};q=str(qa.get('query',''));target=f"request {q} target tool {sch.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
    chunks=[]
    if ses is not None:
        eps=es.episodes(ses);picked=es.retrieve_episodes(enc,eps,target,2)
        for rank,ep,sim in picked:chunks.append(f"EPISODE {rank+1} (relevance {sim:.3f})\n{ep['text'][:5000]}")
    j=jp.get(p)
    if j is not None:chunks.append(f"STRUCTURED MEMORY CANDIDATE field={j.get('field')} value={j.get('value')}")
    return '\n\n'.join(chunks)[:10000]

def build_cases():
    td=Path(tempfile.gettempdir())/'mab_inferred_resid_v2';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'));cases=[];total=0;hits=0
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by);props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            if str((gi.get(p) or {}).get('type','unknown'))!='inferred':continue
            total+=1;d=props.get(p) or {};cands=ss.build_slot(enc,qa,ses,p,d,jp);hit=any(c['src']!='omit' and base.norm(c['value'])==base.norm(g) for c in cands);hits+=int(hit)
            if not hit:cases.append({'qi':qi,'qa':qa,'p':p,'d':d,'gold':g,'ev':evidence(enc,qa,ses,p,d,jp)})
    return cases,total,hits

def run_model(model,cases):
    tok=AutoTokenizer.from_pretrained(model);tok.pad_token=tok.eos_token;tok.padding_side='left';lm=AutoModelForCausalLM.from_pretrained(model,dtype=torch.float32,device_map=None).eval();correct=0;samples=[];B=4
    for st in range(0,len(cases),B):
        batch=cases[st:st+B];prompts=[]
        for c in batch:
            qa=c['qa'];sch=qa.get('target_tool_schema') or {};body=(
                'Resolve exactly ONE missing tool argument from the current request and tightly scoped long-term memory. '
                'The deterministic compiler could not produce the exact value, so derive the representation required by the target schema. '
                'Return ONLY JSON with exactly one key named value and a scalar/array payload: {"value": ...}. Do not return tool calls, explanations, or unrelated fields.\n\n'
                f"CURRENT REQUEST: {qa.get('query','')}\nTARGET TOOL: {sch.get('name','')}\nTARGET PARAMETER: {c['p']}\nSCHEMA: {json.dumps(c['d'],ensure_ascii=False)}\n\nSCOPED MEMORY:\n{c['ev']}"
            );prompts.append(tok.apply_chat_template([{'role':'system','content':'Derive one tool argument from scoped memory. Output only {"value": ...} JSON.'},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True))
        inp=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=4096)
        with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)
        L=inp['input_ids'].shape[1]
        for j,c in enumerate(batch):
            raw=tok.decode(gen[j,L:],skip_special_tokens=True);pred=parse_value(raw,c['p']);ok=base.norm(pred)==base.norm(c['gold']);correct+=int(ok);samples.append({'qa_id':c['qa'].get('qa_id'),'parameter':c['p'],'gold':c['gold'],'pred':pred,'correct':ok,'raw':raw[:400]})
        print('RESID_V2_PROGRESS',model,min(st+B,len(cases)),'/',len(cases),flush=True)
    del lm;del tok
    return {'model':model,'residual_correct':correct,'residual_accuracy':correct/max(1,len(cases)),'samples':samples}

def main():
    cases,total,hits=build_cases();print('RESID_V2_CASES',len(cases),'deterministic_hits',hits,'total',total,flush=True);runs=[]
    for m in MODELS:runs.append(run_model(m,cases))
    for r in runs:r['combined_oracle_plus_residual_coverage']=(hits+r['residual_correct'])/max(1,total)
    result={'stage':'Development-only residual semantic-capacity comparison with robust target-slot JSON parsing','total_inferred':total,'deterministic_candidate_hits':hits,'residual_cases':len(cases),'runs':runs,'guardrail':'QA001-100 development labels select/score inferred missing-candidate cases only. No QA101-400 labels are read. Output parser extracts only the requested target parameter from model-generated JSON; it never searches benchmark gold.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_INFERRED_RESIDUAL_V2='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
