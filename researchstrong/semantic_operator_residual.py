from pathlib import Path
from collections import defaultdict
import json,re,tempfile,numpy as np,torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
import gatea_source_selector as ss
import joint_schema_alignment as js
from gatea_pairwise_cv import CachedEncoder
from semantic_property_ingest_oracle import generic_entities

OUT=Path(__file__).with_name('semantic_operator_residual_result.json')
MODEL='Qwen/Qwen2.5-1.5B-Instruct';N=100;JOINT=(0.75,0.0,2)

def unwrap(z,p):
    if isinstance(z,dict):
        if 'value' in z and not isinstance(z['value'],dict):return z['value']
        if p in z:return z[p]
        for k,v in z.items():
            if str(k).casefold()==str(p).casefold():return v
        for v in z.values():
            x=unwrap(v,p)
            if x is not None:return x
    if isinstance(z,list):
        for v in z:
            x=unwrap(v,p)
            if x is not None:return x
    return None

def parse(text,p):
    s=text.strip();a=s.find('{');b=s.rfind('}');tries=[s]+([s[a:b+1]] if a>=0 and b>a else [])
    for t in tries:
        try:
            z=json.loads(t);x=unwrap(z,p)
            if x is not None:return x
            if not isinstance(z,(dict,list)):return z
        except:pass
    return s.strip('`"\' \n')

def leaf(k):return re.sub(r'\[\d+\]$','',str(k).split('.')[-1])
def user_before(ep,turn):
    x=''
    for ti,t in ep['rows']:
        if ti>turn:break
        if str(t.get('role',''))=='user':x=str(t.get('content',''))
    return x[:500]

def semantic_packet(enc,qa,ses,p,d):
    sch=qa.get('target_tool_schema') or {};query=str(qa.get('query',''));target=f"current request {query}; tool {sch.get('name','')}; parameter {p}; meaning {d.get('description','')}; type {d.get('type','')}"
    rows=[]
    if ses is not None:
        for er,ep,esim in es.retrieve_episodes(enc,es.episodes(ses),target,3):
            for c in af.occurrences(ep,er,esim):
                if c['kind']=='text_span':continue
                role=f"historical intent {user_before(ep,int(c.get('turn',-1)))}; tool {c.get('tool','')}; field {leaf(c.get('key',''))}; path {c.get('key','')}; kind {c.get('kind','')}"
                rows.append({'value':c['value'],'role':role,'src':'structured'})
            for ti,t in ep['rows']:
                if isinstance(t.get('content',''),str):
                    for v,k in generic_entities(t['content']):
                        rows.append({'value':v,'role':f"historical intent {user_before(ep,ti)}; natural entity kind {k}",'src':'text'})
    for v,k in generic_entities(query):rows.append({'value':v,'role':f'current request entity kind {k}','src':'current'})
    # Value-blind role ranking; actual values are only shown after candidate roles are ranked.
    if not rows:return []
    E=enc.encode([target]+[r['role'] for r in rows],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);sc=E[1:]@E[0];order=np.argsort(-sc)
    out=[];seen=set()
    for i in order:
        r=rows[int(i)];k=base.norm(r['value'])
        if not k or k in seen:continue
        seen.add(k);out.append({'value':r['value'],'role':r['role'],'score':float(sc[int(i)]),'src':r['src']})
        if len(out)>=14:break
    return out

def build_cases(enc):
    td=Path(tempfile.gettempdir())/'semantic_operator_resid';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);cases=[];hits=0;total=0
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by);defs=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {};gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            if str((gi.get(p) or {}).get('type','unknown'))!='inferred':continue
            total+=1;d=defs.get(p) or {};basec=ss.build_slot(enc,qa,ses,p,d,jp);hit=any(c['src']!='omit' and base.norm(c['value'])==base.norm(g) for c in basec);hits+=int(hit)
            if not hit:cases.append({'qa':qa,'p':p,'d':d,'gold':g,'packet':semantic_packet(enc,qa,ses,p,d)})
    return cases,total,hits

def main():
    enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'));cases,total,hits=build_cases(enc);print('SEMOP_CASES',len(cases),'base_hits',hits,'/',total,flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL);tok.pad_token=tok.eos_token;tok.padding_side='left';lm=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map=None).eval();correct=0;samples=[];B=4
    for st in range(0,len(cases),B):
        batch=cases[st:st+B];prompts=[]
        for c in batch:
            qa=c['qa'];sch=qa.get('target_tool_schema') or {};cand='\n'.join([f"- candidate {i+1}: {json.dumps(x['value'],ensure_ascii=False)} ; semantic source: {x['role']}" for i,x in enumerate(c['packet'])]) or '- none'
            body=("Resolve one tool argument using a two-stage semantic operator. First identify the entity/concept/state the request refers to from the candidate packet. Then convert or derive ONLY the representation required by the target schema. Candidates are evidence, not necessarily the final surface form. For identifiers/codes, use standard world knowledge when needed (e.g. entity->ticker, country->ISO code, state->postal code). For a search/query parameter, synthesize a concise query from the user's stated goal. For code/SQL parameters, produce the executable representation the schema requests. Return only JSON {\"value\": ...}.\n\n"+f"CURRENT REQUEST: {qa.get('query','')}\nTARGET TOOL: {sch.get('name','')}\nTARGET PARAMETER: {c['p']}\nSCHEMA: {json.dumps(c['d'],ensure_ascii=False)}\nSEMANTIC CANDIDATES:\n{cand}")
            prompts.append(tok.apply_chat_template([{'role':'system','content':'Bind a semantic entity/concept to one exact tool argument. Output only JSON.'},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True))
        inp=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=3072)
        with torch.inference_mode():gen=lm.generate(**inp,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
        L=inp['input_ids'].shape[1]
        for j,c in enumerate(batch):
            raw=tok.decode(gen[j,L:],skip_special_tokens=True);pred=parse(raw,c['p']);ok=base.norm(pred)==base.norm(c['gold']);correct+=int(ok);samples.append({'qa_id':c['qa'].get('qa_id'),'parameter':c['p'],'gold':c['gold'],'pred':pred,'correct':ok,'packet_top':[x['value'] for x in c['packet'][:6]],'raw':raw[:300]})
        print('SEMOP_PROGRESS',min(st+B,len(cases)),'/',len(cases),'correct',correct,flush=True)
    result={'stage':'Development-only semantic operator residual diagnostic','architecture':'value-blind semantic entity/property packet -> 1.5B reasoner applies target-role-specific world-knowledge/normalization/query/program operator','model':MODEL,'total_inferred':total,'baseline_candidate_hits':hits,'residual_cases':len(cases),'residual_correct':correct,'residual_accuracy':correct/max(1,len(cases)),'combined_coverage':(hits+correct)/max(1,total),'samples':samples,'guardrail':'QA001-100 development only. Candidate ranking is value-blind; gold is scoring only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_OPERATOR_RESIDUAL='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
