from pathlib import Path
from collections import defaultdict,Counter
import json,re,tempfile,statistics
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import strong_banking77 as base
import semantic_property_ingest_oracle as sw
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_concept_ingest_oracle_result.json')
N=100
KEEP_CONNECTORS={'of','for','to','in','on','with','and','from','by'}
EXTRA_STOP={'can','could','would','please','thanks','thank','show','find','get','give','tell','also','now','current','latest','earlier','before','same','just','really','like','want','need'}
STOP=set(ENGLISH_STOP_WORDS)|EXTRA_STOP

def concepts(text):
    """Target-independent compact semantic concepts from a historical user utterance.
    Keep content unigrams plus short connector-bearing phrases, but cap output aggressively."""
    text=str(text)
    toks=re.findall(r"[A-Za-z][A-Za-z0-9'_-]*",text)
    low=[t.casefold() for t in toks]
    scored=[]
    # Content unigrams: useful latent concepts such as atmosphere/technology/station.
    for i,(raw,z) in enumerate(zip(toks,low)):
        if len(z)>=4 and z not in STOP:
            scored.append((2.0+min(len(z),12)/20.0,raw,'concept_word'))
    # Short semantic phrases. Stop at hard stopwords, but allow relation connectors.
    for i in range(len(toks)):
        if low[i] in STOP:continue
        parts=[];content=0
        for j in range(i,min(len(toks),i+7)):
            z=low[j]
            if z in STOP and z not in KEEP_CONNECTORS:break
            if z not in STOP:content+=1
            parts.append(toks[j])
            if 2<=content<=4 and len(parts)>=2:
                phrase=' '.join(parts)
                if 5<=len(phrase)<=90:
                    scored.append((3.0+0.3*content-0.04*len(parts),phrase,'concept_phrase'))
    # Deduplicate and cap per turn by informativeness/length. This is intentionally not a bag of every ngram.
    best={}
    for score,v,k in scored:
        n=base.norm(v)
        if not n:continue
        if n not in best or score>best[n][0]:best[n]=(score,v,k)
    rows=sorted(best.values(),key=lambda x:(-x[0],len(x[1]),x[1].casefold()))[:48]
    return [(v,k) for _,v,k in rows]

def ingest_with_concepts(session):
    props,records=sw.ingest(session)
    if session is None:return props,records,0
    added=0
    for ep in __import__('episode_scoped_router').episodes(session):
        for ti,t in ep['rows']:
            if str(t.get('role',''))!='user':continue
            c=t.get('content','')
            if isinstance(c,(dict,list)):c=json.dumps(c,ensure_ascii=False)
            for v,k in concepts(c):props.append((v,k,'semantic_concept',ti));added+=1
    return props,records,added

def main():
    td=Path(tempfile.gettempdir())/'semantic_concept_ingest';td.mkdir(exist_ok=True);qp=td/'q';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N];repaired,report=repair.build();stats=defaultdict(lambda:Counter(n=0,covered=0,multi=0));examples=[];counts=[];added_counts=[]
    for qi,qa in enumerate(qas):
        ses=(repaired.get(qa.get('qa_id')) or {}).get('session');props_world,records,added=ingest_with_concepts(ses);counts.append(len(props_world));added_counts.append(added)
        gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {});defs=(((qa.get('target_tool_schema') or {}).get('parameters') or {}).get('properties') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'));vals,prov=sw.query_world(props_world,records,p,defs.get(p) or {});k=base.norm(g);hit=k in vals;c=stats[typ];c['n']+=1;c['covered']+=int(hit);c['multi']+=int(hit and len(prov[k])>1)
            if typ=='inferred' and hit and len(examples)<30:examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'gold':g,'provenance':prov[k][:8]})
        if qi%10==0:print('CONCEPT_WORLD_BUILD',qi,'properties',len(props_world),'concept_added',added,flush=True)
    packed={k:{'n':v['n'],'coverage':v['covered']/max(1,v['n']),'multi_provenance_rate':v['multi']/max(1,v['n'])} for k,v in stats.items()}
    result={'stage':'SWM-A query-independent semantic concept-property ingest oracle','split':'QA001-100 development only; QA101-400 gold sealed','architecture':'answer-blind repaired sessions -> existing structured/entity world state + at most 48 provenance-preserving semantic concept properties per historical user turn -> existing deterministic schema transforms','results':packed,'memory_size':{'mean_total_properties_per_session':statistics.mean(counts),'median_total_properties_per_session':statistics.median(counts),'mean_concept_properties_added':statistics.mean(added_counts),'median_concept_properties_added':statistics.median(added_counts),'max_total_properties':max(counts)},'inferred_recoveries':examples,'passes_SWM_A':packed.get('explicit',{}).get('coverage',0)>=0.85 and packed.get('inferred',{}).get('coverage',0)>=0.40,'guardrail':'Concept extraction is query/target independent and capped per user turn. Gold is scoring-only. Session repair is answer-blind. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_CONCEPT_INGEST_ORACLE='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
