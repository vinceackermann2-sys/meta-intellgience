from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
from datetime import datetime
from dateutil import parser as dtparser
import pycountry, webcolors
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af

OUT=Path(__file__).with_name('normalization_operator_oracle_result.json')
N=100;TOP_EPISODES=2


def add(out,seen,v,op,src):
    try:z=base.norm(v)
    except:return
    if not z or z in seen:return
    seen.add(z);out.append((v,op,src))


def country_variants(s):
    out=[]
    try:
        c=pycountry.countries.lookup(str(s).strip())
        out.extend([c.name,c.alpha_2,c.alpha_3])
        if getattr(c,'official_name',None):out.append(c.official_name)
    except:pass
    return out


def language_variants(s):
    out=[];q=str(s).strip()
    try:
        x=pycountry.languages.lookup(q)
        for a in ['name','alpha_2','alpha_3','bibliographic','terminology']:
            v=getattr(x,a,None)
            if v:out.append(v)
    except:pass
    return out


def color_variants(s):
    out=[];q=str(s).strip()
    try:out.append(webcolors.name_to_hex(q))
    except:pass
    try:out.append(webcolors.hex_to_name(q.lower()))
    except:pass
    return out


def number_variants(v):
    out=[]
    try:
        x=float(str(v).strip())
        if not (abs(x)<1e20):return out
        if abs(x-round(x))<1e-12:out.append(int(round(x)))
        for n in range(0,7):out.append(f'{x:.{n}f}')
    except:pass
    return out


def date_variants(s):
    q=str(s).strip();out=[]
    if not re.search(r'\d',q):return out
    try:
        d=dtparser.parse(q,fuzzy=False)
        for f in ['%Y-%m-%d','%m/%d/%Y','%d/%m/%Y','%m-%d-%Y','%d-%m-%Y','%Y/%m/%d']:
            out.append(d.strftime(f))
    except:pass
    return out


def string_variants(s):
    q=str(s).strip();out=[]
    if not q:return out
    out.extend([q.lower(),q.upper(),q.title()])
    words=re.findall(r'[A-Za-z]+',q)
    if 2<=len(words)<=8:out.append(''.join(w[0] for w in words).upper())
    if ',' in q:out.append(q.split(',')[0].strip())
    m=re.search(r'\(([^()]{1,20})\)',q)
    if m:out.append(m.group(1).strip())
    return out


def variants(v):
    out=[]
    if isinstance(v,bool):out.extend([str(v).lower(),'yes' if v else 'no'])
    if not isinstance(v,(dict,list)):
        q=str(v)
        out.extend((x,'string') for x in string_variants(q))
        out.extend((x,'country') for x in country_variants(q))
        out.extend((x,'language') for x in language_variants(q))
        out.extend((x,'color') for x in color_variants(q))
        out.extend((x,'number') for x in number_variants(q))
        out.extend((x,'date') for x in date_variants(q))
    return out


def main():
    td=Path(tempfile.gettempdir())/'mab_norm_oracle';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    stats=defaultdict(lambda:Counter(n=0,identity=0,transformed=0,any=0));ops=defaultdict(Counter);samples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});query=str(qa.get('query',''))
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));stats[typ]['n']+=1;d=props.get(p) or {};target=f'Current request: {query}. Target tool: {schema.get("name","")}. Target parameter: {p}. Meaning: {d.get("description","")}. Type: {d.get("type","")}'
            picked=es.retrieve_episodes(enc,eps,target,TOP_EPISODES);raw=[]
            for rank,ep,sim in picked:
                for c in af.occurrences(ep,rank,sim):raw.append((c['value'],c['kind']))
            # Current request and schema enums are legitimate operator inputs too.
            raw.extend((v,'query_span') for v in base.spans(query))
            if isinstance(d.get('enum'),list):raw.extend((v,'schema_enum') for v in d['enum'])
            id_hit=any(base.norm(v)==base.norm(g) for v,_ in raw)
            if id_hit:stats[typ]['identity']+=1
            transformed=[];seen=set()
            for v,src in raw:
                add(transformed,seen,v,'identity',src)
                for x,op in variants(v):add(transformed,seen,x,op,src)
            hits=[(v,op,src) for v,op,src in transformed if base.norm(v)==base.norm(g)]
            nonid=[h for h in hits if h[1]!='identity']
            if nonid:
                stats[typ]['transformed']+=1
                for _,op,_ in nonid:ops[typ][op]+=1
            if hits:stats[typ]['any']+=1
            if typ=='inferred' and len(samples)<20:
                samples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'identity_hit':id_hit,'transform_hits':[{'value':v,'op':op,'source':src} for v,op,src in nonid[:8]]})
    packed={}
    for typ,c in stats.items():
        n=max(1,c['n']);packed[typ]={'n':c['n'],'identity_coverage':c['identity']/n,'transform_coverage':c['transformed']/n,'union_coverage':c['any']/n,'transform_ops':dict(ops[typ])}
    result={'stage':'Mem2Act general normalization operator oracle coverage','operators':['case/title/acronym','country ISO/name','language code/name','CSS color name/hex','numeric precision/type','common date formats','schema enum','query spans'],'results':packed,'missing_zero_based':missing,'samples':samples,'guardrail':'QA001-100 gold is used only to score operator coverage. Operators are general transformations, not QA-specific mappings. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_NORMALIZATION_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
