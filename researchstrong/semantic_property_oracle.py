from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile
import pycountry, webcolors
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
import gatea_source_selector as ss
import typed_span_diagnostic as ts

OUT=Path(__file__).with_name('semantic_property_oracle_result.json')
N=100

US_STATES={
'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'}

def norm(x): return base.norm(x)
def flatten(x,prefix=''):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            kk=f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v,(dict,list)): out.extend(flatten(v,kk))
            else: out.append((kk,v))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            kk=f'{prefix}[{i}]'
            if isinstance(v,(dict,list)): out.extend(flatten(v,kk))
            else: out.append((kk,v))
    return out

def record_values(turn):
    recs=[]
    content=turn.get('content','')
    if isinstance(content,(dict,list)):
        vals=flatten(content)
        if vals: recs.append(vals)
    elif isinstance(content,str) and content.strip()[:1] in '[{':
        try:
            vals=flatten(json.loads(content))
            if vals: recs.append(vals)
        except Exception: pass
    for tc in turn.get('tool_calls') or []:
        vals=flatten(base.parse_args(tc))
        if vals: recs.append(vals)
    return recs

def generic_variants(v,p,d):
    vals=[v]
    # Existing schema-conditioned transforms.
    for op in ss.allowed_ops(v,p,d):
        z=ss.transform(v,op,d)
        if z is not None: vals.append(z)
    s=str(v).strip(); meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    # Generic US administrative ontology.
    if any(x in meta for x in ['state','province','region','location','area']):
        for name,code in US_STATES.items():
            if s.casefold()==name.casefold(): vals.append(code)
            if s.upper()==code: vals.append(name)
    # General country/language aliases even when target description is weak.
    for lookup,attrs in [(pycountry.countries,('alpha_2','alpha_3','name')),(pycountry.languages,('alpha_2','alpha_3','name'))]:
        try:
            obj=lookup.lookup(s)
            for a in attrs:
                z=getattr(obj,a,None)
                if z: vals.append(z)
        except Exception: pass
    try:
        if re.fullmatch(r'#[0-9A-Fa-f]{6}',s): vals.append(webcolors.hex_to_name(s.lower()))
        else: vals.append(webcolors.name_to_hex(s))
    except Exception: pass
    # Numeric surface normalization.
    try:
        f=float(s.replace(',',''))
        if f.is_integer(): vals += [int(f),str(int(f)),f'{f:.1f}']
    except Exception: pass
    out=[];seen=set()
    for z in vals:
        k=norm(z)
        if k not in seen:seen.add(k);out.append(z)
    return out

def compile_world(session,qa,p,d):
    # Ingest-time world state: all historical structured fields + typed entities, preserving provenance.
    props=[]; pair_records=[]
    if session is not None:
        for ep in es.episodes(session):
            for ti,t in ep['rows']:
                for rec in record_values(t):
                    pair_records.append(rec)
                    for key,v in rec: props.append(('structured',key,v,ti))
                content=t.get('content','')
                if isinstance(content,str):
                    for sp in ts.typed_spans(content,p,d): props.append(('text:'+sp['kind'],'text_entity',sp['value'],ti))
    # Current request is legitimate action input, not memory leakage.
    for sp in ts.typed_spans(str(qa.get('query','')),p,d): props.append(('current:'+sp['kind'],'current_request',sp['value'],-1))
    cands=[]
    for src,key,v,ti in props:
        for z in generic_variants(v,p,d): cands.append((z,src,key,ti,'unary'))
    # Generic pair/composition operator for API symbols such as EURUSD/XBTUSD when codes co-occur in a record.
    meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    if any(x in meta for x in ['pair','symbol','ticker','code']):
        for rec in pair_records:
            atoms=[]
            for _,v in rec:
                s=str(v).strip().upper()
                if re.fullmatch(r'[A-Z]{2,5}',s): atoms.append(s)
            atoms=list(dict.fromkeys(atoms))[:20]
            for a in atoms:
                for b in atoms:
                    if a!=b: cands.append((a+b,'composition','record_pair',-1,'concat'))
    # Deduplicate by normalized value only; world-model property provenance remains countable separately.
    vals={}; prov=defaultdict(list)
    for z,src,key,ti,op in cands:
        k=norm(z)
        if not k: continue
        vals.setdefault(k,z); prov[k].append((src,key,ti,op))
    return vals,prov

def main():
    td=Path(tempfile.gettempdir())/'semantic_property_oracle';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp)
    stats=defaultdict(lambda:Counter(n=0,covered=0,multi=0)); examples=[]; missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None: missing.append(qi)
        gold=((qa.get('tool_call') or {}).get('arguments') or {}); gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};vals,prov=compile_world(ses,qa,p,d);k=norm(g);hit=k in vals
            m=stats[typ];m['n']+=1;m['covered']+=int(hit);m['multi']+=int(hit and len(prov[k])>1)
            if typ=='inferred' and hit and len(examples)<20:
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'provenance':prov[k][:8]})
        if qi%10==0:print('SEM_PROPERTY_BUILD',qi,flush=True)
    result={'stage':'Semantic World Model v0 property-store coverage oracle','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'compile entire historical session once into structured field properties + typed natural-language entities + generic ontology/normalization transforms + record-local code composition; query checks target role against this compiled property universe later','results':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n']),'multi_provenance_rate':v['multi']/max(1,v['n'])} for k,v in stats.items()},'inferred_recoveries':examples,'missing_zero_based':missing,'guardrail':'Gold values are scoring-only. World-state compilation uses only conversation history/current request, generic ontologies/transforms, and target schema. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_PROPERTY_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
