from pathlib import Path
from collections import defaultdict
import json,re,tempfile,urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
import pycountry, webcolors

BASE='https://raw.githubusercontent.com/Cantaloupe-M/Mem2ActBench/main/Mem2ActBench/'
OUT=Path(__file__).with_name('proposition_availability_oracle_result.json')
N=100
TOPS=(2,5)

def fetch(name,path):
    if not path.exists(): urllib.request.urlretrieve(BASE+name,path)
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def norm(x):
    if isinstance(x,bool):return str(x).lower()
    if x is None:return 'null'
    if isinstance(x,(dict,list)):return json.dumps(x,sort_keys=True,ensure_ascii=False).casefold()
    return re.sub(r'\s+',' ',str(x).strip()).casefold()
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
    g=defaultdict(list);order=[]
    for ti,t in enumerate(session.get('turns') or []):
        sid=str(t.get('source_id') or f'unknown_{ti}')
        if sid not in g:order.append(sid)
        g[sid].append((ti,t))
    return [{'sid':sid,'rows':g[sid]} for sid in order]
def stringify(x):
    if isinstance(x,(dict,list)):return json.dumps(x,ensure_ascii=False)
    return str(x or '')
def scope_text(ep):
    # value-blind-ish routing: prioritize user language and tool/schema names, but raw assistant prose is excluded
    parts=[]
    for _,t in ep['rows']:
        if str(t.get('role','')).lower()=='user':parts.append('user intent '+stringify(t.get('content',''))[:1200])
        for tc in t.get('tool_calls') or []:
            a=parse_args(tc);parts.append('historical tool '+tool_name(tc)+' fields '+', '.join(map(str,a.keys())))
    return '\n'.join(parts)[:5000]
def raw_text(ep):
    parts=[]
    for _,t in ep['rows']:
        role=str(t.get('role',''));parts.append(role+': '+stringify(t.get('content','')))
        for tc in t.get('tool_calls') or []:parts.append('tool_call '+tool_name(tc)+': '+stringify(parse_args(tc)))
    return '\n'.join(parts)
def sentences(text):
    # Preserve code/URLs while producing manageable proposition-sized chunks.
    lines=[]
    for line in str(text).splitlines():
        line=line.strip()
        if not line:continue
        chunks=re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])',line)
        lines.extend(c.strip() for c in chunks if c.strip())
    return lines

def canon_variants(g):
    vals={norm(g)}
    s=str(g).strip()
    try:
        c=pycountry.countries.lookup(s);vals|={norm(c.name),norm(c.alpha_2),norm(c.alpha_3),norm(getattr(c,'official_name',c.name))}
    except Exception:pass
    try:
        l=pycountry.languages.lookup(s);vals|={norm(getattr(l,'name','')),norm(getattr(l,'alpha_2','')),norm(getattr(l,'alpha_3',''))}
    except Exception:pass
    try:
        if s.startswith('#'):vals.add(norm(webcolors.hex_to_name(s)))
        else:vals.add(norm(webcolors.name_to_hex(s)))
    except Exception:pass
    return {x for x in vals if x}
def contained(g,text):
    gt=norm(g);tt=norm(text)
    return bool(gt) and gt in tt

def target_text(qa,p,d):
    sc=qa.get('target_tool_schema') or {}
    return f"current request {qa.get('query','')} target tool {sc.get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
def hit_stats(stats,mode,typ,hit):
    d=stats[mode][typ];d['n']+=1;d['hit']+=int(hit)
def main():
    td=Path(tempfile.gettempdir())/'prop_availability';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';fetch('qa_dataset.jsonl',qp);fetch('toolmem_conversation.jsonl',cp)
    qas=list(load_jsonl(qp))[:N];sessions,by=build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    stats=defaultdict(lambda:defaultdict(lambda:{'n':0,'hit':0}));examples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=episodes(ses);scopes=[scope_text(e) for e in eps]
        EE=enc.encode(scopes,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32) if scopes else np.zeros((0,384),np.float32)
        schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};qv=enc.encode([target_text(qa,p,d)],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)[0]
            order=np.argsort(-(EE@qv)) if len(EE) else np.array([],dtype=int)
            groups={'top2':order[:2],'top5':order[:5],'all':order}
            for name,ids in groups.items():
                texts=[raw_text(eps[int(i)]) for i in ids];whole='\n'.join(texts)
                exact=contained(g,whole)
                # proposition containment is stronger: gold is contained in one local sentence/chunk.
                prop=any(contained(g,s) for t in texts for s in sentences(t))
                hit_stats(stats,name+'_raw_substring',typ,exact);hit_stats(stats,name+'_proposition',typ,prop)
            if typ=='inferred':
                alltxt='\n'.join(raw_text(e) for e in eps)
                hit=contained(g,alltxt)
                if len(examples)<20:
                    supporting=[]
                    if hit:
                        for e in eps:
                            for s in sentences(raw_text(e)):
                                if contained(g,s):supporting.append(s[:500])
                    examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'present_in_raw_history':hit,'query':qa.get('query',''),'support':supporting[:3]})
    packed={mode:{typ:{'n':d['n'],'coverage':d['hit']/max(1,d['n'])} for typ,d in bytyp.items()} for mode,bytyp in stats.items()}
    out={'stage':'QA001-100 development-only raw proposition availability oracle','question':'Are missing tool arguments explicitly stated somewhere in longitudinal raw experience, or must they be semantically derived?','results':packed,'inferred_examples':examples,'missing_zero_based':missing,'guardrail':'Gold is scoring-only for substring/proposition availability. Episode routing never uses gold. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False));print('PROPOSITION_AVAILABILITY='+json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
