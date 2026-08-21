from pathlib import Path
from collections import defaultdict, Counter
import json, tempfile, re
import numpy as np
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_anchor_role_binding_cv as ar
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_role_ontology_result.json')
N=100

# Fixed before scoring. Generic tool/action ontology, not benchmark answer labels.
ONTOLOGY={
 'entity_name':'human-readable name/title/label of a person organization object product book media component or entity',
 'entity_identifier':'stable unique identifier id uuid key handle contract collection account project book component or object reference',
 'market_symbol':'stock ticker trading symbol market symbol security code for a company or asset',
 'currency_code':'currency code such as USD EUR SEK',
 'currency_pair':'forex crypto or trading pair combining two asset/currency symbols',
 'country_name':'country nation name',
 'country_code':'ISO country code alpha-2 alpha-3 geographic country identifier',
 'state_region_name':'state province region subregion geographic area name',
 'state_region_code':'state province region postal or administrative area code',
 'city_location':'city place destination location geographic name',
 'latitude':'latitude geographic coordinate north south',
 'longitude':'longitude geographic coordinate east west',
 'url':'URL URI web link endpoint address',
 'email':'email e-mail address',
 'username':'username user handle login screen name',
 'phone':'phone telephone mobile number',
 'date':'calendar date departure return start end day',
 'datetime':'timestamp date and time instant',
 'year_season':'year season campaign current period',
 'duration':'duration interval timeout elapsed time',
 'boolean':'boolean yes no true false flag enabled disabled',
 'count_limit':'count number limit page size quantity max results',
 'page_offset':'page index offset cursor pagination position',
 'numeric_measure':'numeric measurement rate percentage price amount score latitude-like scalar',
 'search_query':'search query q keyword keywords text query phrase terms to search',
 'sql_program':'SQL database query executable query program statement',
 'code_program':'code script program source component code',
 'category':'category type genre class domain topic',
 'person_assignee':'person actor owner assignee contact author artist player employee',
 'organization':'company organization employer institution team exchange platform provider',
 'file_path':'file path directory location filesystem path filename',
 'connection_string':'database connection string DSN JDBC URI',
 'network':'network chain blockchain environment mainnet testnet protocol',
 'color':'color name RGB hex colour',
 'language':'language locale language code',
 'address':'postal street mailing physical address',
 'coordinates':'coordinate pair lat lon position',
 'time_zone':'timezone time zone UTC offset',
 'free_text':'free-form text content description message body prompt note',
}


def leaf(k):return re.sub(r'\[\d+\]$','',str(k).split('.')[-1])

def target_role_text(qa,p,d):
    sch=qa.get('target_tool_schema') or {}
    return f"future tool {sch.get('name','')}; parameter {p}; description {d.get('description','')}; type {d.get('type','')}"

def source_role_text(n,c):
    prop=c['prop'];field=str(prop.get('field',''));intent=wb.mask(n.get('intent',''),prop.get('value'))
    return f"historical tool {n.get('tool','')}; field {leaf(field)}; full path {field}; siblings {n.get('siblings','')}; record kind {n.get('kind','')}; historical intent {intent}; transform {c.get('op','identity')}"

def main():
    td=Path(tempfile.gettempdir())/'semantic_role_ontology';td.mkdir(exist_ok=True);qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp)
    qas=list(base.load_jsonl(qp))[:N];repaired,report=repair.build();enc=wb.CachedEncoder()
    labels=list(ONTOLOGY);ontE=enc.encode([f"semantic role {k}: {ONTOLOGY[k]}" for k in labels])
    stats=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0));examples=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'));direct,graph,scope_labels=ar.graph_scope(nodes,qa.get('query',''))
        strictE=enc.encode([n['strict_desc'] for n in nodes]) if nodes else np.zeros((0,384),np.float32)
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};m=stats[typ];m['n']+=1
            tv=enc.encode(target_role_text(qa,p,d));tont=ontE@tv;target_ix=int(np.argmax(tont));target_label=labels[target_ix]
            sv=enc.encode(h.target_scope(qa,p,d));ns=strictE@sv if len(nodes) else np.asarray([]);fallback={int(i) for i in np.argsort(-ns)[:4]} if len(nodes) else set();scoped=set(graph)|fallback
            rows=[]
            for ni in scoped:
                n=nodes[ni];scope=scope_labels.get(ni,'semantic_fallback');scope_bonus={'direct':0.35,'episode':0.22,'shared':0.27,'semantic_fallback':0.0}.get(scope,0.0)
                for c in h.expanded(n,p,d):
                    sr=source_role_text(n,c);rv=enc.encode(sr);os=ontE@rv;source_ix=int(np.argmax(os));source_label=labels[source_ix]
                    role_sim=float(rv@tv);ontology_sim=float(os[target_ix]);exact=float(source_label==target_label)
                    score=1.15*exact+0.55*ontology_sim+0.35*role_sim+scope_bonus+0.08*float(ns[ni] if len(ns) else 0.0)
                    rows.append({'value':c['value'],'score':score,'field':c['prop'].get('field',''),'op':c.get('op'),'scope':scope,'source_label':source_label,'target_label':target_label})
            if not rows:continue
            pos=[i for i,r in enumerate(rows) if base.norm(r['value'])==base.norm(g)];m['covered']+=int(bool(pos))
            # Aggregate same executor payload by strongest semantic-role evidence.
            groups=defaultdict(list)
            for i,r in enumerate(rows):groups[base.norm(r['value'])].append(i)
            packed=[]
            for k,ids in groups.items():
                if not k:continue
                best=max(ids,key=lambda i:rows[i]['score']);packed.append((rows[best]['score'],best,ids))
            order=sorted(range(len(packed)),key=lambda j:-packed[j][0]);goldg={j for j,(_,_,ids) in enumerate(packed) if any(i in pos for i in ids)}
            m['top1']+=int(bool(order) and order[0] in goldg);m['top3']+=int(any(j in goldg for j in order[:3]));m['top5']+=int(any(j in goldg for j in order[:5]))
            if len(examples)<14 and goldg and (not order or order[0] not in goldg):
                examples.append({'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'target_role':target_label,'top':[{'score':packed[j][0],'field':rows[packed[j][1]]['field'],'source_role':rows[packed[j][1]]['source_label'],'scope':rows[packed[j][1]]['scope'],'op':rows[packed[j][1]]['op']} for j in order[:4]],'gold_roles':[rows[packed[j][1]]['source_label'] for j in goldg if j<len(packed)][:4]})
        if qi%10==0:print('ROLE_ONTOLOGY_BUILD',qi,'nodes',len(nodes),flush=True)
    out={}
    for typ,c in stats.items():
        n=max(1,c['n']);cov=max(1,c['covered']);out[typ]={'n':c['n'],'candidate_coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov}
    result={'stage':'SWM-B fixed canonical semantic-role ontology binder','split':'QA001-100 development diagnostic only; QA101-400 gold remains sealed','ontology_size':len(labels),'architecture':'entity/shared-value focus graph + masked target/source role descriptions -> independent nearest canonical role classification -> exact payload dereference/transform after role matching','results':out,'examples':examples,'repair_report':report,'guardrail':'Ontology and scoring formula are fixed before QA scoring. Candidate payload values are not embedded or used in role classification. Gold values score exact dereference only. No QA101-400 gold is read.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ROLE_ONTOLOGY='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
