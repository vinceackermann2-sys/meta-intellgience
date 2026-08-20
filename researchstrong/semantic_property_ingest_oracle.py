from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile
import pycountry, webcolors
import strong_banking77 as base
import episode_scoped_router as es
import gatea_source_selector as ss

OUT=Path(__file__).with_name('semantic_property_ingest_oracle_result.json')
N=100
STOP={'I','We','You','He','She','They','It','The','A','An','Can','Could','Would','Will','Do','Does','Did','How','What','Which','Where','When','Why','Please','Thanks','Thank','Show','Find','Get','Give','Tell','Also','Now','Current','Latest'}
US_STATES={'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'}

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

def generic_entities(text):
    text=str(text); out=[]; seen=set()
    def add(v,kind):
        v=str(v).strip().strip('"\'')
        if not v or len(v)>180:return
        k=norm(v)
        if k and k not in seen:seen.add(k);out.append((v,kind))
    pats=[(r'https?://[^\s\]\[\)\(<>"\']+','url'),(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','email'),(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','ipv4'),(r'\b0x[a-fA-F0-9]{8,}\b','hex_id'),(r'\b[A-Za-z]+_[A-Za-z0-9_-]+\b','named_id'),(r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b','date'),(r'(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])','number'),(r'\b[A-Z][A-Z0-9._/-]{1,15}\b','code')]
    for pat,kind in pats:
        for m in re.finditer(pat,text):add(m.group(0),kind)
    for m in re.finditer(r'"([^"\n]{1,100})"',text):add(m.group(1),'quoted')
    for m in re.finditer(r"'([^'\n]{1,100})'",text):add(m.group(1),'quoted')
    # Proper-noun phrases and conservative single proper nouns. Ingest rule is target-independent.
    for m in re.finditer(r'\b[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){1,4}\b',text):add(m.group(0),'proper_phrase')
    for m in re.finditer(r'\b[A-Z][A-Za-z0-9.-]{2,}\b',text):
        if m.group(0) not in STOP:add(m.group(0),'proper_single')
    return out

def record_values(turn):
    recs=[]; content=turn.get('content','')
    if isinstance(content,(dict,list)):
        vals=flatten(content)
        if vals:recs.append(vals)
    elif isinstance(content,str) and content.strip()[:1] in '[{':
        try:
            vals=flatten(json.loads(content))
            if vals:recs.append(vals)
        except Exception:pass
    for tc in turn.get('tool_calls') or []:
        vals=flatten(base.parse_args(tc))
        if vals:recs.append(vals)
    return recs

def ingest(session):
    props=[];records=[]
    if session is None:return props,records
    for ep in es.episodes(session):
        for ti,t in ep['rows']:
            for rec in record_values(t):
                records.append(rec)
                for key,v in rec:props.append((v,'structured',key,ti))
            content=t.get('content','')
            if isinstance(content,str):
                for v,kind in generic_entities(content):props.append((v,'text:'+kind,'text_entity',ti))
    return props,records

def variants(v,p,d):
    vals=[v];s=str(v).strip();meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    for op in ss.allowed_ops(v,p,d):
        z=ss.transform(v,op,d)
        if z is not None:vals.append(z)
    if any(x in meta for x in ['state','province','region','location','area']):
        for name,code in US_STATES.items():
            if s.casefold()==name.casefold():vals.append(code)
            if s.upper()==code:vals.append(name)
    for lookup,attrs in [(pycountry.countries,('alpha_2','alpha_3','name')),(pycountry.languages,('alpha_2','alpha_3','name'))]:
        try:
            obj=lookup.lookup(s)
            for a in attrs:
                z=getattr(obj,a,None)
                if z:vals.append(z)
        except Exception:pass
    try:
        if re.fullmatch(r'#[0-9A-Fa-f]{6}',s):vals.append(webcolors.hex_to_name(s.lower()))
        else:vals.append(webcolors.name_to_hex(s))
    except Exception:pass
    try:
        f=float(s.replace(',',''))
        if f.is_integer():vals += [int(f),str(int(f)),f'{f:.1f}']
    except Exception:pass
    out=[];seen=set()
    for z in vals:
        k=norm(z)
        if k and k not in seen:seen.add(k);out.append(z)
    return out

def query_world(props,records,p,d):
    vals={};prov=defaultdict(list)
    for v,src,key,ti in props:
        for z in variants(v,p,d):
            k=norm(z);vals.setdefault(k,z);prov[k].append((src,key,ti,'unary'))
    meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    if any(x in meta for x in ['pair','symbol','ticker','code']):
        for rec in records:
            atoms=[]
            for _,v in rec:
                s=str(v).strip().upper()
                if re.fullmatch(r'[A-Z]{2,5}',s):atoms.append(s)
            atoms=list(dict.fromkeys(atoms))[:20]
            for a in atoms:
                for b in atoms:
                    if a!=b:
                        z=a+b;k=norm(z);vals.setdefault(k,z);prov[k].append(('composition','record_pair',-1,'concat'))
    return vals,prov

def main():
    td=Path(tempfile.gettempdir())/'semantic_property_ingest';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);stats=defaultdict(lambda:Counter(n=0,covered=0,multi=0));examples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi)
        props_world,records=ingest(ses);gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});defs=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));vals,prov=query_world(props_world,records,p,defs.get(p) or {});k=norm(g);hit=k in vals;m=stats[typ];m['n']+=1;m['covered']+=int(hit);m['multi']+=int(hit and len(prov[k])>1)
            if typ=='inferred' and hit and len(examples)<20:examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'provenance':prov[k][:8]})
        if qi%10==0:print('INGEST_WORLD_BUILD',qi,'properties',len(props_world),'records',len(records),flush=True)
    result={'stage':'Semantic World Model v0 strict query-independent ingest property oracle','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'INGEST ONCE per session: all structured fields + target-independent generic entities/codes/IDs/dates/numbers/proper nouns. QUERY: target-schema-conditioned deterministic ontology/format transforms and record-local code composition only.','results':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n']),'multi_provenance_rate':v['multi']/max(1,v['n'])} for k,v in stats.items()},'inferred_recoveries':examples,'missing_zero_based':missing,'guardrail':'Historical extraction is query/target independent. Gold is scoring-only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_PROPERTY_INGEST_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
