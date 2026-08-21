from pathlib import Path
from collections import Counter
import json,tempfile,numpy as np,torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM
import strong_banking77 as base
import episode_scoped_router as es
import joint_schema_alignment as js
import gatea_source_selector as ss
from gatea_pairwise_cv import CachedEncoder

OUT=Path(__file__).with_name('inferred_residual_diagnostic_result.json')
MODEL='Qwen/Qwen2.5-0.5B-Instruct';N=100;JOINT=(0.75,0.0,2)

def parse_value(text):
    s=text.strip()
    # Prefer a JSON object with a value key, otherwise JSON scalar/string.
    try:
        z=json.loads(s)
        if isinstance(z,dict) and 'value' in z:return z['value']
        if not isinstance(z,(dict,list)):return z
    except:pass
    a=s.find('{');b=s.rfind('}')
    if a>=0 and b>a:
        try:
            z=json.loads(s[a:b+1]);return z.get('value') if isinstance(z,dict) else None
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

def main():
    td=Path(tempfile.gettempdir())/'mab_inferred_resid';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'))
    cases=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by);props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            if str((gi.get(p) or {}).get('type','unknown'))!='inferred':continue
            d=props.get(p) or {};cands=ss.build_slot(enc,qa,ses,p,d,jp);hit=any(c['src']!='omit' and base.norm(c['value'])==base.norm(g) for c in cands)
            if not hit:cases.append({'qi':qi,'qa':qa,'p':p,'d':d,'gold':g,'ev':evidence(enc,qa,ses,p,d,jp)})
    print('INFERRED_RESIDUAL_CASES',len(cases),flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL);tok.pad_token=tok.eos_token;tok.padding_side='left';lm=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map=None).eval()
    correct=0;samples=[];B=4
    for st in range(0,len(cases),B):
        batch=cases[st:st+B];prompts=[]
        for c in batch:
            qa=c['qa'];sch=qa.get('target_tool_schema') or {};body=(
                'You resolve ONE missing tool argument from a user request and tightly scoped long-term memory. '
                'The exact answer is NOT available as a precompiled pointer candidate, so derive the representation required by the target schema. '
                'Do not invent unrelated facts. Return JSON only: {"value": ...}.\n\n'
                f"CURRENT REQUEST: {qa.get('query','')}\nTARGET TOOL: {sch.get('name','')}\nTARGET PARAMETER: {c['p']}\nSCHEMA: {json.dumps(c['d'],ensure_ascii=False)}\n\nSCOPED MEMORY:\n{c['ev']}"
            );prompts.append(tok.apply_chat_template([{'role':'system','content':'Ground tool arguments from scoped memory. Output only JSON.'},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True))
        inp=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=4096)
        with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
        L=inp['input_ids'].shape[1]
        for j,c in enumerate(batch):
            raw=tok.decode(gen[j,L:],skip_special_tokens=True);pred=parse_value(raw);ok=base.norm(pred)==base.norm(c['gold']);correct+=int(ok);samples.append({'qa_id':c['qa'].get('qa_id'),'parameter':c['p'],'gold':c['gold'],'pred':pred,'correct':ok,'raw':raw[:300]})
        print('INFERRED_RESIDUAL_PROGRESS',min(st+B,len(cases)),'/',len(cases),flush=True)
    # Existing deterministic oracle covers the complement of cases by construction.
    total_inferred=sum(1 for qa in qas for p in ((qa.get('tool_call') or {}).get('arguments') or {}) if str((((qa.get('tool_call') or {}).get('grounding_info') or {}).get(p) or {}).get('type','unknown'))=='inferred')
    deterministic_hits=total_inferred-len(cases);combined=(deterministic_hits+correct)/max(1,total_inferred)
    result={'stage':'Development-only inferred residual reasoning diagnostic','model':MODEL,'total_inferred':total_inferred,'deterministic_candidate_hits':deterministic_hits,'residual_cases':len(cases),'residual_correct':correct,'residual_accuracy':correct/max(1,len(cases)),'combined_oracle_plus_residual_coverage':combined,'samples':samples,'guardrail':'QA001-100 development labels select/score inferred missing-candidate cases only. No QA101-400 labels are read. This is a capability diagnostic, not a deployable routing policy.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_INFERRED_RESIDUAL='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
