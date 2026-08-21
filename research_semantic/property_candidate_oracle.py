from pathlib import Path
from collections import defaultdict
import json,re,tempfile,urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
import pycountry, webcolors

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
OUT=Path(__file__).with_name('property_candidate_oracle_result.json')
N=100

US_STATES={'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO','connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV','new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK','oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX','utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY','district of columbia':'DC'}

def fetch(name,path):
    if not path.exists(): urllib.request.urlretrieve(BASE+name,path)
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def norm(v):
    if isinstance(v,bool): return str(v).lower()
    if v is None:return 'null'
    if isinstance(v,(dict,list)):return json.dumps(v,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(v).strip()).casefold()
def parse_args(tc):
    if not isinstance(tc,dict):return {}
    f=tc.get('function') or {};a=f.get('arguments',{}) if isinstance(f,dict) else {}
    if isinstance(a,dict):return a
    if isinstance(a,str):
        try:z=json.loads(a);return z if isinstance(z,dict) else {}
        except Exception:return {}
    return {}
def tool_name(tc):
    f=(tc or {}).get('function') if isinstance(tc,dict) else {};return str((f or {}).get('name','')) if isinstance(f,dict) else ''
def flatten(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            p=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)):out.extend(flatten(v,p))
            else:out.append((p,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            p=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)):out.extend(flatten(v,p))
            else:out.append((p,v))
    return out
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
def episodes(session):
    groups=defaultdict(list);order=[]
    for ti,t in enumerate(session.get('turns') or []):
        sid=str(t.get('source_id') or f'unknown_{ti}')
        if sid not in groups:order.append(sid)
        groups[sid].append((ti,t))
    return [{'source_id':sid,'rows':groups[sid]} for sid in order]
def scope_text(ep):
    parts=[]
    for _,t in ep['rows']:
        if str(t.get('role','')).lower()=='user':
            c=t.get('content','');c=json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c)
            parts.append('user intent '+c[:1000])
        for tc in t.get('tool_calls') or []:
            a=parse_args(tc);parts.append('tool '+tool_name(tc)+' fields '+', '.join(k for k,_ in flatten(a)[:30]))
    return '\n'.join(parts)[:5000]
def values(ep):
    out=[]
    for ti,t in ep['rows']:
        c=t.get('content','')
        if isinstance(c,(dict,list)):
            for k,v in flatten(c):out.append((v,k,'content'))
        elif isinstance(c,str) and c.strip()[:1] in '[{':
            try:
                z=json.loads(c)
                for k,v in flatten(z):out.append((v,k,'json'))
            except Exception:pass
        for tc in t.get('tool_calls') or []:
            for k,v in flatten(parse_args(tc)):out.append((v,k,'tool:'+tool_name(tc)))
    return out

def variants(v):
    out=[v]
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        if float(v).is_integer():out.extend([int(v),float(v),str(int(v)),f'{float(v):.1f}'])
        else:out.extend([str(v)])
    if isinstance(v,str):
        s=v.strip();low=s.casefold()
        # country name/code canonicalization
        try:
            c=pycountry.countries.lookup(s)
            out.extend([c.name,c.alpha_2,c.alpha_3,getattr(c,'official_name',c.name)])
        except Exception:pass
        try:
            l=pycountry.languages.lookup(s)
            out.extend([getattr(l,'name',''),getattr(l,'alpha_2',''),getattr(l,'alpha_3','')])
        except Exception:pass
        if low in US_STATES:out.append(US_STATES[low])
        rev={v.casefold():k.title() for k,v in US_STATES.items()}
        if low in rev:out.append(rev[low])
        try:
            if s.startswith('#'):out.append(webcolors.hex_to_name(s))
            else:out.append(webcolors.name_to_hex(s))
        except Exception:pass
    return [x for x in out if x not in ('',None)]
def target_text(qa,p,d):
    sc=qa.get('target_tool_schema') or {}
    return f"current request {qa.get('query','')} target tool {sc.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
def coverage_add(store,typ,mode,hit):
    d=store[mode][typ];d['n']+=1;d['hit']+=int(hit)
def main():
    td=Path(tempfile.gettempdir())/'semantic_property_oracle';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch('qa_dataset.jsonl',qp);fetch('toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N];sessions,by=build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    stats=defaultdict(lambda:defaultdict(lambda:{'n':0,'hit':0}));missing=[];examples=[]
    for qi,qa in enumerate(qas):
        ses=find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=episodes(ses);texts=[scope_text(e) for e in eps]
        EE=enc.encode(texts,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if texts else np.zeros((0,384),np.float32)
        schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};qv=enc.encode([target_text(qa,p,d)],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
            order=np.argsort(-(EE@qv)) if len(EE) else np.array([],dtype=int)
            scopes={'top2':order[:2],'top5':order[:5],'all':order}
            for name,ids in scopes.items():
                raw=[]
                for i in ids:raw.extend(values(eps[int(i)]))
                rawhit=any(norm(v)==norm(g) for v,_,_ in raw)
                txhit=any(norm(x)==norm(g) for v,_,_ in raw for x in variants(v))
                coverage_add(stats,typ,name+'_raw',rawhit);coverage_add(stats,typ,name+'_transformed',txhit)
                if typ=='inferred' and name=='all' and txhit and not rawhit and len(examples)<12:
                    matches=[]
                    for v,k,src in raw:
                        for x in variants(v):
                            if norm(x)==norm(g):matches.append({'source_value':v,'derived':x,'field':k,'source':src})
                    examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'query':qa.get('query',''),'matches':matches[:5]})
    packed={}
    for mode,bytyp in stats.items():
        packed[mode]={typ:{'n':d['n'],'coverage':d['hit']/max(1,d['n'])} for typ,d in bytyp.items()}
    result={'stage':'QA001-100 development-only semantic property candidate ceiling','architecture':'value-blind episode routing -> all structured fields -> scope curve top2/top5/all -> general deterministic canonical transforms','results':packed,'new_inferred_transform_examples':examples,'missing_zero_based':missing,'guardrail':'Gold is scoring-only. Candidate generation/routing never uses gold. QA101-400 gold is not read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_PROPERTY_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
