from pathlib import Path
from collections import defaultdict, Counter, deque
import json,re,tempfile,numpy as np
from sentence_transformers import SentenceTransformer
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af
import joint_schema_alignment as js
import gatea_source_selector as ss
from gatea_pairwise_cv import CachedEncoder

OUT=Path(__file__).with_name('lineage_candidate_oracle_result.json')
N=100;K=2;JOINT=(0.75,0.0,2);TOP_SEM=3


def rec_fields(obj):
    try:return af.flatten(obj)
    except:return []

def make_records(ep,er,esim):
    out=[];rid=0
    for ti,t in ep['rows']:
        role=str(t.get('role',''));content=t.get('content','')
        def add(kind,tool,fields):
            nonlocal rid
            fields=[(str(k),v) for k,v in fields if not isinstance(v,(dict,list))]
            if not fields:return
            desc=f'role {role}; {kind}; tool {tool}; field paths: '+', '.join(k for k,_ in fields[:80])
            out.append({'id':f'{er}:{ti}:{rid}','rank':er,'turn':ti,'role':role,'tool':tool,'fields':fields,'desc':desc,'episode_sim':esim});rid+=1
        if isinstance(content,(dict,list)):add('structured content','',rec_fields(content))
        elif isinstance(content,str):
            st=content.strip()
            if st[:1] in '[{':
                try:add('json content','',rec_fields(json.loads(st)))
                except:pass
        for tc in t.get('tool_calls') or []:
            add('tool arguments',base.tool_name(tc),rec_fields(base.parse_args(tc)))
    return out

def scalar_norm(v):
    if isinstance(v,(dict,list)) or v is None:return None
    s=base.norm(v)
    if not s:return None
    # Reject generic connector values that would create meaningless graph hubs.
    if s in {'true','false','none','null','unknown','0','1','yes','no','success','ok'}:return None
    if len(s)<3:return None
    return s

def literal_anchor(query,v):
    s=scalar_norm(v)
    if not s:return False
    q=str(query).casefold()
    raw=str(v).strip().casefold()
    if len(raw)<2:return False
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',raw):return re.search(r'(?<!\w)'+re.escape(raw)+r'(?!\w)',q) is not None
    return raw in q

def lineage_sets(records,query,target,enc):
    if not records:return set(),set(),set(),{}
    E=enc.encode([target]+[r['desc'] for r in records],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32);sims=E[1:]@E[0]
    for r,s in zip(records,sims):r['sem']=float(s)
    literal={i for i,r in enumerate(records) if any(literal_anchor(query,v) for _,v in r['fields'])}
    sem=set(np.argsort(-sims)[:min(TOP_SEM,len(records))].tolist())
    seed=literal|sem
    idx=defaultdict(set)
    for i,r in enumerate(records):
        for _,v in r['fields']:
            z=scalar_norm(v)
            if z:idx[z].add(i)
    adj=defaultdict(set)
    for ids in idx.values():
        if len(ids)<2 or len(ids)>12:continue
        ids=list(ids)
        for i in ids:
            adj[i].update(j for j in ids if j!=i)
    hop1=set(seed)
    for i in list(seed):hop1.update(adj.get(i,set()))
    hop2=set(hop1)
    for i in list(hop1):hop2.update(adj.get(i,set()))
    meta={'literal_seed_count':len(literal),'semantic_seed_count':len(sem),'records':len(records),'edges':sum(len(v) for v in adj.values())//2}
    return seed,hop1,hop2,meta

def generated_values(records,ids,p,d):
    out=[]
    for i in ids:
        r=records[i]
        for key,v in r['fields']:
            for op in ss.allowed_ops(v,p,d):
                z=ss.transform(v,op,d)
                if z is not None:out.append((z,op,r,key))
    return out

def main():
    td=Path(tempfile.gettempdir())/'mab_lineage_oracle';td.mkdir(exist_ok=True);qp=td/'q';cp=td/'c'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:N];sessions,by=base.build_session_map(cp);enc=CachedEncoder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'))
    totals=Counter();bylvl=defaultdict(Counter);examples=[];missing=[];graphstats=Counter()
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:missing.append(qi)
        props=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        pack=js.qa_pack(enc,qa,ses) if ses is not None else None;jp=js.predict(pack,*JOINT) if pack is not None else {}
        for p,g in gold.items():
            lvl=str((gi.get(p) or {}).get('type','unknown'));d=props.get(p) or {};totals[lvl]+=1;bylvl[lvl]['n']+=1
            basec=ss.build_slot(enc,qa,ses,p,d,jp);basehit=any(c['src']!='omit' and base.norm(c['value'])==base.norm(g) for c in basec);bylvl[lvl]['baseline']+=int(basehit)
            if ses is None:continue
            target=f"request {qa.get('query','')} target tool {(qa.get('target_tool_schema') or {}).get('name','')} parameter {p} meaning {d.get('description','')} type {d.get('type','')}"
            rec=[]
            for er,ep,sm in es.retrieve_episodes(enc,es.episodes(ses),target,K):rec.extend(make_records(ep,er,sm))
            seed,h1,h2,meta=lineage_sets(rec,str(qa.get('query','')),target,enc)
            for k,v in meta.items():graphstats[k]+=v
            hits=[]
            for name,ids in [('seed',seed),('hop1',h1),('hop2',h2)]:
                vals=generated_values(rec,ids,p,d);hit=any(base.norm(z)==base.norm(g) for z,_,_,_ in vals);bylvl[lvl][name]+=int(hit);hits.append((name,hit,vals))
            if lvl=='inferred' and not basehit and hits[-1][1] and len(examples)<20:
                match=[]
                for z,op,r,key in hits[-1][2]:
                    if base.norm(z)==base.norm(g):match.append({'value':z,'op':op,'field':key,'tool':r['tool'],'turn':r['turn'],'rank':r['rank'],'desc':r['desc'][:250]})
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'query':qa.get('query',''),'matches':match[:5],'graph':meta})
        if qi%10==0:print('LINEAGE_BUILD',qi,'cache',len(enc.cache),flush=True)
    packed={}
    for lvl,c in bylvl.items():
        n=max(1,c['n']);packed[lvl]={'n':c['n'],'baseline_candidate_coverage':c['baseline']/n,'seed_record_coverage':c['seed']/n,'one_hop_coverage':c['hop1']/n,'two_hop_coverage':c['hop2']/n}
    result={'stage':'Development-only entity/data-lineage candidate coverage oracle','split':'QA001-100 only; QA101-400 labels remain sealed','architecture':'top-2 source_id episodes -> structured records -> literal query entity + masked record-semantic seeds -> exact-value producer/consumer links -> sibling/1-hop/2-hop fields -> deterministic schema transforms','results':packed,'examples_new_inferred_recoveries':examples,'average_graph_stats':{k:v/max(1,N) for k,v in graphstats.items()},'missing_zero_based':missing,'guardrail':'Gold values are used only to score coverage and print matching examples. Episode retrieval, entity anchoring, graph edges, field traversal, and transforms are answer-blind. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_LINEAGE_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
