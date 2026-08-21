from pathlib import Path
from collections import defaultdict
import json,re,tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import proposition_availability_oracle as pa

OUT=Path(__file__).with_name('extractive_proposition_reader_result.json')
N=100
TOP_EPISODES=2
TOP_CHUNKS=5
MODEL='deepset/roberta-base-squad2'


def chunks(text,max_chars=1400):
    ss=pa.sentences(text);out=[];buf=''
    for s in ss:
        if len(buf)+len(s)+1>max_chars and buf:
            out.append(buf);buf=''
        buf=(buf+' '+s).strip()
    if buf:out.append(buf)
    return out

def question(qa,p,d):
    sc=qa.get('target_tool_schema') or {}
    return (f"What exact value from the user's previous experience should be used for parameter '{p}' "
            f"of tool '{sc.get('name','')}' for the current request: {qa.get('query','')} "
            f"Parameter meaning: {d.get('description','')}. Return the value span from the memory.")
def add(stats,typ,correct,present,answered):
    d=stats[typ];d['n']+=1;d['correct']+=int(correct);d['present']+=int(present);d['answered']+=int(answered);d['correct_when_present']+=int(correct and present)
def main():
    td=Path(tempfile.gettempdir())/'extractive_prop_reader';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';pa.fetch('qa_dataset.jsonl',qp);pa.fetch('toolmem_conversation.jsonl',cp)
    qas=list(pa.load_jsonl(qp))[:N];sessions,by=pa.build_session_map(cp)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    reader=pipeline('question-answering',model=MODEL,tokenizer=MODEL,device=-1)
    stats=defaultdict(lambda:{'n':0,'correct':0,'present':0,'answered':0,'correct_when_present':0});samples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=pa.find_session(qa,sessions,by)
        if ses is None: missing.append(qi);continue
        eps=pa.episodes(ses);scope=[pa.scope_text(e) for e in eps]
        EE=enc.encode(scope,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if scope else np.zeros((0,384),np.float32)
        sc=qa.get('target_tool_schema') or {};props=((sc.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};qtxt=pa.target_text(qa,p,d);qv=enc.encode([qtxt],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
            order=np.argsort(-(EE@qv))[:min(TOP_EPISODES,len(eps))] if len(EE) else []
            cs=[]
            for i in order: cs.extend(chunks(pa.raw_text(eps[int(i)])))
            present=any(pa.contained(g,c) for c in cs)
            if not cs:
                add(stats,typ,False,present,False);continue
            CE=enc.encode(cs,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
            corder=np.argsort(-(CE@qv))[:min(TOP_CHUNKS,len(cs))]
            qq=question(qa,p,d);best={'score':-1.0,'answer':'' ,'context':''}
            for ci in corder:
                try:r=reader(question=qq,context=cs[int(ci)],max_answer_len=64,handle_impossible_answer=True)
                except TypeError:r=reader(question=qq,context=cs[int(ci)],max_answer_len=64)
                if float(r.get('score',0.0))>best['score']:best={'score':float(r.get('score',0.0)),'answer':r.get('answer',''),'context':cs[int(ci)]}
            pred=best['answer'];ok=pa.norm(pred)==pa.norm(g);answered=bool(str(pred).strip())
            add(stats,typ,ok,present,answered)
            if (typ=='inferred' or typ=='explicit') and len(samples)<30:
                samples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'gold':g,'present_top2_raw':present,'pred':pred,'score':best['score'],'correct':ok,'context':best['context'][:500]})
        if qi%10==0:print('EXTRACTIVE_PROGRESS',qi,flush=True)
    packed={}
    for typ,d in stats.items():
        n=max(1,d['n']);pr=max(1,d['present']);packed[typ]={'n':d['n'],'accuracy':d['correct']/n,'availability':d['present']/n,'answer_rate':d['answered']/n,'accuracy_given_available':d['correct_when_present']/pr}
    out={'stage':'QA001-100 development-only extractive proposition reader','model':MODEL,'architecture':'value-blind top-2 episode routing -> semantic proposition chunk routing -> pretrained extractive QA span reader -> exact span payload; no generative decoding','results':packed,'samples':samples,'missing_zero_based':missing,'guardrail':'QA001-100 only. Gold is scoring-only. Episode/chunk routing and reader prompts never use gold values. QA101-400 gold is not read.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('EXTRACTIVE_PROPOSITION_READER='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
