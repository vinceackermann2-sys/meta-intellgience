from pathlib import Path
from collections import defaultdict
import json,re,tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from llama_cpp import Llama
import proposition_availability_oracle as pa

OUT=Path(__file__).with_name('residual_synthesis_7b_result.json')
N=100
TOP_EPISODES=2
MODEL_REPO='Qwen/Qwen2.5-7B-Instruct-GGUF'


def unwrap(z,p):
    if isinstance(z,dict):
        if p in z:return z[p]
        for k,v in z.items():
            if str(k).casefold()==str(p).casefold():return v
        if 'value' in z and not isinstance(z['value'],dict):return z['value']
        for v in z.values():
            y=unwrap(v,p)
            if y is not None:return y
    elif isinstance(z,list):
        for v in z:
            y=unwrap(v,p)
            if y is not None:return y
    return None

def parse(text,p):
    s=str(text).strip();cands=[s]
    a=s.find('{');b=s.rfind('}')
    if a>=0 and b>a:cands.append(s[a:b+1])
    for t in cands:
        t=t.strip('` \n')
        if t.lower().startswith('json'):t=t[4:].strip()
        try:
            z=json.loads(t);y=unwrap(z,p)
            if y is not None:return y
            if not isinstance(z,(dict,list)):return z
        except Exception:pass
    return s.strip('`"\' \n')

def build_cases(enc):
    td=Path(tempfile.gettempdir())/'resid7b';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';pa.fetch('qa_dataset.jsonl',qp);pa.fetch('toolmem_conversation.jsonl',cp)
    qas=list(pa.load_jsonl(qp))[:N];sessions,by=pa.build_session_map(cp);cases=[];total=0;present=0;missing=[]
    for qi,qa in enumerate(qas):
        ses=pa.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=pa.episodes(ses);scope=[pa.scope_text(e) for e in eps]
        EE=enc.encode(scope,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if scope else np.zeros((0,384),np.float32)
        sc=qa.get('target_tool_schema') or {};props=((sc.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            if str((gi.get(p) or {}).get('type','unknown'))!='inferred':continue
            total+=1;d=props.get(p) or {};qv=enc.encode([pa.target_text(qa,p,d)],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
            order=np.argsort(-(EE@qv))[:min(TOP_EPISODES,len(eps))] if len(EE) else []
            mem='\n\n'.join(f'EPISODE {r+1}:\n{pa.raw_text(eps[int(i)])[:6000]}' for r,i in enumerate(order))
            has=pa.contained(g,mem);present+=int(has)
            if not has:
                cases.append({'qa_id':qa.get('qa_id'),'query':qa.get('query',''),'tool':sc.get('name',''),'p':p,'schema':d,'gold':g,'memory':mem[:11000]})
    return cases,total,present,missing

def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');cases,total,present,missing=build_cases(enc)
    print('RESID7B_CASES',len(cases),'total_inferred',total,'literal_top2',present,flush=True)
    local=snapshot_download(repo_id=MODEL_REPO,allow_patterns=['*q4_k_m*'])
    parts=sorted(Path(local).glob('*q4_k_m*gguf'));assert parts, 'GGUF not downloaded';first=parts[0]
    llm=Llama(model_path=str(first),n_ctx=8192,n_threads=4,n_threads_batch=4,n_batch=512,verbose=False)
    correct=0;samples=[]
    for ix,c in enumerate(cases):
        prompt=(
            'You are the semantic derivation stage of a long-term-memory action compiler. The exact target value is NOT literally present in the supplied memory. '
            'Infer or construct the exact representation required by the target tool schema from the current request and memory. '
            'Examples of legitimate derivations include entity-to-code normalization, time resolution, search-query formulation, SQL/URL construction, boolean inference, or semantic category mapping. '
            'Return ONLY JSON {"value": ...}. Do not explain.\n\n'
            f"CURRENT REQUEST: {c['query']}\nTARGET TOOL: {c['tool']}\nTARGET PARAMETER: {c['p']}\nPARAMETER SCHEMA: {json.dumps(c['schema'],ensure_ascii=False)}\n\nLONG-TERM MEMORY:\n{c['memory']}"
        )
        resp=llm.create_chat_completion(messages=[{'role':'system','content':'Derive exactly one missing tool argument. Output only JSON with key value.'},{'role':'user','content':prompt}],temperature=0.0,max_tokens=160,top_p=1.0)
        raw=resp['choices'][0]['message']['content'];pred=parse(raw,c['p']);ok=pa.norm(pred)==pa.norm(c['gold']);correct+=int(ok)
        samples.append({'qa_id':c['qa_id'],'parameter':c['p'],'gold':c['gold'],'pred':pred,'correct':ok,'raw':raw[:500]})
        print('RESID7B_PROGRESS',ix+1,'/',len(cases),'correct',correct,flush=True)
    out={'stage':'QA001-100 development-only 7B residual semantic synthesis','model':MODEL_REPO+':Q4_K_M','total_inferred':total,'literal_top2_available':present,'true_residual_cases':len(cases),'residual_correct':correct,'residual_accuracy':correct/max(1,len(cases)),'combined_literal_or_7b_oracle_coverage':(present+correct)/max(1,total),'samples':samples,'missing_zero_based':missing,'guardrail':'The residual set is defined only on QA001-100 development labels. Model prompt never includes gold. QA101-400 gold is not read. This is a capability diagnostic, not a final system or SOTA claim.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('RESIDUAL_SYNTHESIS_7B='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
