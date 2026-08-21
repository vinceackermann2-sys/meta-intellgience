from pathlib import Path
from collections import defaultdict, Counter
import json,re,tempfile
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af

OUT=Path(__file__).with_name('typed_span_diagnostic_result.json')
N=100;TOP_EPISODES=2

CSS_COLORS={'black','silver','gray','white','maroon','red','purple','fuchsia','green','lime','olive','yellow','navy','blue','teal','aqua','orange','aliceblue','gold','pink','brown','cyan','magenta','violet','indigo','beige'}

def local(text,start,end,w=180):
    return text[max(0,start-w):min(len(text),end+w)]

def add(out,seen,val,kind,text,s,e):
    v=str(val).strip().strip('"\'')
    if not v or len(v)>180:return
    z=base.norm(v)
    if z in seen:return
    seen.add(z);out.append({'value':v,'kind':kind,'address':f'unstructured typed {kind}; local context '+af.mask_value(local(text,s,e),v),'key':'','tool':'','turn':-1,'rank':0,'episode_sim':0.0})

def typed_spans(text,p,d):
    text=str(text);meta=(str(p)+' '+str(d.get('description',''))+' '+str(d.get('type',''))).casefold();out=[];seen=set()
    def rx(pat,kind,flags=0,group=0):
        for m in re.finditer(pat,text,flags):
            try:v=m.group(group)
            except:v=m.group(0)
            s,e=(m.span(group) if group else m.span())
            add(out,seen,v,kind,text,s,e)
    if any(x in meta for x in ['url','link','uri','website']):rx(r'https?://[^\s\]\[\)\(<>"\']+','url')
    if 'email' in meta:rx(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','email')
    if any(x in meta for x in ['ip address','ip_address',' ipv4','ipaddr']):rx(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','ipv4')
    if any(x in meta for x in ['date','day','time','timestamp']):
        rx(r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b','date')
        rx(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b','date')
        rx(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b','date')
        rx(r'\b(?:today|tomorrow|yesterday|current|latest|\d+\s*(?:h|hr|hrs|hours|d|days|w|weeks|min|minutes))\b','relative_time',re.I)
    if any(x in meta for x in ['id','identifier','hash','wallet','address','contract']):
        rx(r'\b0x[a-fA-F0-9]{8,}\b','hex_id')
        rx(r'\b[A-Za-z]+_[A-Za-z0-9_-]+\b','named_id')
        rx(r'\b[A-Za-z0-9]{6,}(?:[-_][A-Za-z0-9]{2,})+\b','id')
    if any(x in meta for x in ['code','symbol','ticker','league','currency','pair','language','country']):
        rx(r'\b[A-Z][A-Z0-9._/-]{1,15}\b','code')
        rx(r'\b[a-z]{2,3}\b','short_code')
    if any(x in meta for x in ['user','username','handle','account']):
        rx(r'@[A-Za-z0-9_.-]{2,32}','username')
        rx(r"['\"]([A-Za-z0-9_.-]{2,40})['\"]",'quoted_username',group=1)
    if 'color' in meta:
        rx(r'#[0-9A-Fa-f]{6}\b','hex_color')
        for m in re.finditer(r'\b[A-Za-z]+\b',text):
            if m.group(0).casefold() in CSS_COLORS:add(out,seen,m.group(0),'color_name',text,*m.span())
    typ=str(d.get('type','')).casefold()
    if typ in ('integer','int','number','float') or any(x in meta for x in ['limit','count','page','offset','latitude','longitude','amount','rate','score','results']):
        rx(r'(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])','number')
    # Quoted values are high-precision generic evidence for string-like parameters.
    if typ in ('string','str',''):
        rx(r'"([^"\n]{1,100})"','quoted',group=1);rx(r"'([^'\n]{1,100})'",'quoted',group=1)
    # Entity-like slots get conservative proper-noun phrases, but never lone sentence-initial words.
    if any(x in meta for x in ['name','location','city','country','destination','origin','search','keyword','topic','company','stock','sport']):
        for m in re.finditer(r'\b[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){1,4}\b',text):add(out,seen,m.group(0),'entity_phrase',text,*m.span())
    return out

def main():
    td=Path(tempfile.gettempdir())/'mab_typed_span';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';cp=td/'conv.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    stats=defaultdict(lambda:Counter(n=0,structured=0,typed=0,union=0));kinds=defaultdict(Counter);samples=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi);continue
        eps=es.episodes(ses);schema=qa.get('target_tool_schema') or {};props=((schema.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});query=str(qa.get('query',''))
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));stats[typ]['n']+=1;d=props.get(p) or {};target=f'Current request: {query}. Target tool: {schema.get("name","")}. Target parameter: {p}. Meaning: {d.get("description","")}. Type: {d.get("type","")}'
            picked=es.retrieve_episodes(enc,eps,target,TOP_EPISODES);structured=[];typed=[]
            for rank,ep,sim in picked:
                for c in af.occurrences(ep,rank,sim):
                    if c['kind']!='text_span':structured.append(c)
                for ti,t in ep['rows']:
                    if str(t.get('role','')) not in ('user','assistant','tool'):continue
                    content=t.get('content','');txt=json.dumps(content,ensure_ascii=False) if isinstance(content,(dict,list)) else str(content)
                    for c in typed_spans(txt,p,d):c['turn']=ti;c['rank']=rank;c['episode_sim']=sim;typed.append(c)
            hs=any(base.norm(c['value'])==base.norm(g) for c in structured);ht=any(base.norm(c['value'])==base.norm(g) for c in typed)
            stats[typ]['structured']+=int(hs);stats[typ]['typed']+=int(ht);stats[typ]['union']+=int(hs or ht)
            for c in typed:
                if base.norm(c['value'])==base.norm(g):kinds[typ][c['kind']]+=1
            if typ=='explicit' and len(samples)<15 and ht and not hs:samples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'typed_hits':[{'kind':c['kind'],'value':c['value'],'address':c['address'][:240]} for c in typed if base.norm(c['value'])==base.norm(g)][:4]})
    packed={}
    for typ,c in stats.items():
        n=max(1,c['n']);packed[typ]={'n':c['n'],'structured_coverage':c['structured']/n,'typed_span_coverage':c['typed']/n,'union_coverage':c['union']/n,'positive_typed_kinds':dict(kinds[typ])}
    result={'stage':'Mem2Act schema-typed unstructured span coverage diagnostic','results':packed,'samples':samples,'missing_zero_based':missing,'guardrail':'QA001-100 gold is scoring only. Extraction rules are schema/type driven and do not use gold values. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_TYPED_SPAN='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
