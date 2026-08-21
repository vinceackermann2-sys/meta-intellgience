from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import strong_banking77 as base
import semantic_entity_hierarchy_diagnostic as h
import semantic_entity_anchor_oracle as a
import semantic_world_binding_cv as wb
import mem2act_repaired_sessions as repair

OUT=Path(__file__).with_name('semantic_anchor_role_binding_cv_result.json')
N=100; SEED=20260820
CS=(0.01,0.03,0.1,0.3,1.0,3.0)
MARGINS=(-0.5,-0.25,0.0,0.25,0.5)


def toks(x): return set(re.findall(r'[a-z0-9]+',str(x).casefold()))
def jac(x,y):
    A,B=toks(x),toks(y); return len(A&B)/max(1,len(A|B))
def leaf(k): return re.sub(r'\[\d+\]$','',str(k).split('.')[-1])
def canon(x): return re.sub(r'[^a-z0-9]+','',str(x).casefold())

def semantic_type(text):
    s=' '+str(text).casefold()+' '
    rules=[
      ('url',(' url ',' uri ',' link ',' endpoint')),
      ('email',('email','e-mail')),
      ('lat',('latitude',' lat ')),('lon',('longitude',' lon ',' lng ')),
      ('country_code',('country code','country_code','iso country','alpha-2','alpha 2')),
      ('state_code',('state code','province code','region code','postal code')),
      ('market_symbol',('ticker','stock symbol','market symbol',' symbol ')),
      ('currency_pair',('currency pair','forex pair',' pair ')),
      ('date_time',('date','time','year','season','timestamp')),
      ('query_program',('sql','query','search terms','search term','keyword','keywords',' q ')),
      ('identifier',(' id ','_id','identifier','unique identifier','uuid')),
      ('boolean',('boolean',' bool ',' true ',' false ')),
      ('numeric',('number','count','amount','price','rate','limit','offset','page','index','float','integer',' int ')),
    ]
    for name,keys in rules:
        if any(k in s for k in keys): return name
    return 'text'

def compatible(tgt,src):
    if tgt=='text': return 0.5
    if tgt==src:return 1.0
    groups=[{'country_code','state_code','identifier'},{'market_symbol','identifier'},{'currency_pair','market_symbol'},{'numeric','identifier'},{'query_program','text'},{'date_time','text'}]
    return 0.6 if any(tgt in g and src in g for g in groups) else 0.0

def graph_scope(nodes,query):
    qa_alias=a.query_aliases(str(query)); aliases=[a.node_aliases(n) for n in nodes]
    direct={i for i,x in enumerate(aliases) if x & qa_alias}
    ep_groups=defaultdict(list)
    for i,n in enumerate(nodes):ep_groups[n.get('episode')].append(i)
    same=set(direct)
    for i in list(direct):same.update(ep_groups.get(nodes[i].get('episode'),[]))
    inv=defaultdict(list)
    for i,xs in enumerate(aliases):
        for x in xs:inv[x].append(i)
    shared=set(direct)
    for i in list(direct):
        for x in aliases[i]:shared.update(inv.get(x,[]))
    labels={i:'direct' for i in direct}
    for i in same:
        labels.setdefault(i,'episode')
    for i in shared:
        labels.setdefault(i,'shared')
    return direct,same|shared,labels

def build_slots(enc):
    td=Path(tempfile.gettempdir())/'semantic_anchor_role';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';base.fetch(base.BASE+'qa_dataset.jsonl',qp);qas=list(base.load_jsonl(qp))[:N]
    repaired,report=repair.build();slots=[];node_counts=[]
    for qi,qa in enumerate(qas):
        rr=repaired.get(qa.get('qa_id')) or {};nodes=h.build_nodes(rr.get('session'));node_counts.append(len(nodes))
        sch=qa.get('target_tool_schema') or {};defs=((sch.get('parameters') or {}).get('properties') or {});gold=((qa.get('tool_call') or {}).get('arguments') or {});gi=((qa.get('tool_call') or {}).get('grounding_info') or {})
        direct,graph,scope_labels=graph_scope(nodes,qa.get('query',''))
        strictE=enc.encode([n['strict_desc'] for n in nodes]) if nodes else np.zeros((0,384),np.float32)
        max_turn=max([int(n.get('turn',-1)) for n in nodes]+[1])
        for p,g in gold.items():
            typ=str((gi.get(p) or {}).get('type','unknown'))
            if typ not in ('explicit','inferred'):continue
            d=defs.get(p) or {};sv=enc.encode(h.target_scope(qa,p,d));rv=enc.encode(h.target_role(qa,p,d));ns=strictE@sv if len(nodes) else np.array([])
            fallback=set(int(i) for i in np.argsort(-ns)[:5]) if len(nodes) else set()
            scoped=set(graph)|fallback
            rows=[]
            for ni in scoped:
                n=nodes[ni]; node_sim=float(ns[ni]) if len(ns) else 0.0; scope=scope_labels.get(ni,'semantic_fallback')
                for c in h.expanded(n,p,d):
                    role_v=enc.encode(c['role']); role_sim=float(role_v@rv); field=str(c['prop'].get('field','')); src_text=f"{field} {n.get('tool','')} {n.get('siblings','')} {n.get('kind','')}"
                    tgt_text=f"{p} {d.get('description','')} {sch.get('name','')}"
                    tgt_type=semantic_type(tgt_text); src_type=semantic_type(src_text+' '+str(c.get('op','')))
                    feat={
                      'node_sim':node_sim,'role_sim':role_sim,'intent_sim':float(enc.encode(n.get('intent',''))@sv) if n.get('intent') else 0.0,
                      'field_jacc':jac(field,p+' '+str(d.get('description',''))),'tool_jacc':jac(n.get('tool',''),sch.get('name','')),
                      'siblings_jacc':jac(n.get('siblings',''),p+' '+str(d.get('description',''))),'query_intent_jacc':jac(qa.get('query',''),n.get('intent','')),
                      'recency':float(int(n.get('turn',-1))/max(1,max_turn)),'scope_direct':float(scope=='direct'),'scope_episode':float(scope=='episode'),'scope_shared':float(scope=='shared'),'scope_fallback':float(scope=='semantic_fallback'),
                      'node_structured':float(n.get('kind')=='structured_record'),'node_semantic_text':float(n.get('kind')=='semantic_text'),
                      'target_type_'+tgt_type:1.0,'source_type_'+src_type:1.0,'type_compat':compatible(tgt_type,src_type),
                      'op_'+str(c.get('op','')):1.0,'field_leaf_exact':float(canon(leaf(field))==canon(p) and bool(field)),
                    }
                    rows.append({'value':c['value'],'feat':feat,'label':int(base.norm(c['value'])==base.norm(g)),'scope':scope,'field':field,'op':c.get('op'),'tool':n.get('tool','')})
            # aggregate identical executor values while preserving evidence counts/maxima, still never exposing the value itself to features
            groups=defaultdict(list)
            for r in rows:groups[base.norm(r['value'])].append(r)
            agg=[]
            for k,rs in groups.items():
                if not k:continue
                f={};numeric=set()
                for r in rs:numeric.update(r['feat'].keys())
                for key in numeric:
                    vals=[float(r['feat'].get(key,0.0)) for r in rs]
                    f[key+'_max']=max(vals);f[key+'_mean']=sum(vals)/len(vals)
                f['evidence_count']=float(len(rs));f['multi_source']=float(len({(r['scope'],r['field'],r['tool']) for r in rs})>1)
                agg.append({'value':rs[0]['value'],'feat':f,'label':int(any(r['label'] for r in rs)),'evidence':[(r['scope'],r['field'],r['op']) for r in rs[:6]]})
            slots.append({'qi':qi,'qa_id':qa.get('qa_id'),'parameter':p,'grounding':typ,'gold':g,'rows':agg})
        if qi%10==0:print('ANCHOR_ROLE_BUILD',qi,'nodes',len(nodes),'slots',len(slots),flush=True)
    return slots,report,node_counts

def train_pairwise(slots,C):
    feats=[];ys=[]
    for s in slots:
        pos=[r for r in s['rows'] if r['label']];neg=[r for r in s['rows'] if not r['label']]
        if not pos or not neg:continue
        for p in pos[:3]:
            for n in neg[:40]:
                keys=set(p['feat'])|set(n['feat']);d={k:float(p['feat'].get(k,0))-float(n['feat'].get(k,0)) for k in keys};feats+=[d,{k:-v for k,v in d.items()}];ys += [1,0]
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(feats);clf=LogisticRegression(C=C,max_iter=3000,random_state=42).fit(X,ys);return vec,clf,len(ys)

def score_row(r,vec,clf):
    z=clf.decision_function(vec.transform([r['feat']]))[0]
    return float(z)

def evaluate(slots,vec,clf,margin):
    total=Counter(correct=0,pred=0,gold=0);lv=defaultdict(lambda:Counter(n=0,correct=0,pred=0,covered=0));exact=0;tasks=defaultdict(lambda:{'gold':0,'pred':0,'correct':0})
    for s in slots:
        typ=s['grounding'];m=lv[typ];m['n']+=1;m['covered']+=int(any(r['label'] for r in s['rows']));total['gold']+=1;tasks[s['qi']]['gold']+=1
        if not s['rows']:continue
        ranked=sorted(s['rows'],key=lambda r:score_row(r,vec,clf),reverse=True);best=ranked[0];sc=score_row(best,vec,clf)
        if sc<margin:continue
        total['pred']+=1;m['pred']+=1;tasks[s['qi']]['pred']+=1
        ok=int(best['label']);total['correct']+=ok;m['correct']+=ok;tasks[s['qi']]['correct']+=ok
    P=total['correct']/max(1,total['pred']);R=total['correct']/max(1,total['gold']);F=2*P*R/max(1e-12,P+R)
    for q,t in tasks.items():exact+=int(t['gold']==t['pred']==t['correct'])
    return {'f1':F,'precision':P,'recall':R,'exact_argument_set':exact/max(1,len(tasks)),'levels':{k:{'n':v['n'],'coverage':v['covered']/max(1,v['n']),'accuracy':v['correct']/max(1,v['n']),'prediction_rate':v['pred']/max(1,v['n'])} for k,v in lv.items()}}

def main():
    enc=wb.CachedEncoder();slots,report,node_counts=build_slots(enc);qids=sorted({s['qi'] for s in slots});kf=KFold(n_splits=5,shuffle=True,random_state=SEED);grid=[]
    for C in CS:
      folds=[]
      for tri,tei in kf.split(qids):
        tq={qids[i] for i in tri};vq={qids[i] for i in tei};tr=[s for s in slots if s['qi'] in tq];va=[s for s in slots if s['qi'] in vq];vec,clf,nrows=train_pairwise(tr,C)
        best=None
        for margin in MARGINS:
            m=evaluate(va,vec,clf,margin)
            if best is None or m['f1']>best['f1']:best={'margin':margin,'pairwise_rows':nrows,**m}
        folds.append(best)
      grid.append({'C':C,'mean_f1':float(np.mean([x['f1'] for x in folds])),'mean_explicit':float(np.mean([x['levels'].get('explicit',{}).get('accuracy',0) for x in folds])),'mean_inferred':float(np.mean([x['levels'].get('inferred',{}).get('accuracy',0) for x in folds])),'folds':folds})
    best=max(grid,key=lambda x:(x['mean_f1'],x['mean_explicit']+x['mean_inferred']));vec,clf,nrows=train_pairwise(slots,best['C']);margin=float(np.median([x['margin'] for x in best['folds']]));refit=evaluate(slots,vec,clf,margin)
    result={'stage':'SWM-B anchor-scoped typed semantic role binding CV','split':'QA001-100 development-only grouped 5-fold CV; QA101-400 gold remains sealed','architecture':'query-independent world nodes -> exact entity/concept anchor + same-episode/shared-value graph expansion, union value-masked semantic fallback nodes -> type-aware pairwise role binding -> exact dereference','grid':grid,'selected_C':best['C'],'selected_mean_cv_f1':best['mean_f1'],'selected_mean_cv_explicit':best['mean_explicit'],'selected_mean_cv_inferred':best['mean_inferred'],'refit_margin':margin,'all_dev_refit':refit,'mean_nodes':float(np.mean(node_counts)),'repair_report':report,'guardrail':'Entity aliases may use exact identity values for indexing, but payload values are never learned/ranking features. Pairwise labels and scoring use QA001-100 gold only. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('SEMANTIC_ANCHOR_ROLE_BINDING='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
