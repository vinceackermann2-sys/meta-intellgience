from pathlib import Path
from collections import defaultdict, Counter
import json, re, tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import strong_banking77 as base
import episode_scoped_router as es
import address_first_diagnostic as af

OUT = Path(__file__).with_name('learned_address_resolver_result.json')
TRAIN_END = 70
DEV_END = 100
TOP_EPISODES = 2


def toks(s):
    return set(re.findall(r'[a-z0-9]+', str(s).casefold()))


def jacc(a,b):
    A,B=toks(a),toks(b)
    return len(A&B)/max(1,len(A|B))


def canon(s):
    return re.sub(r'[^a-z0-9]+','',str(s).casefold())


def leaf(key):
    x=str(key).split('.')[-1]
    return re.sub(r'\[\d+\]$','',x)


def target_text(qa,p,d):
    schema=qa.get('target_tool_schema') or {}
    return f"Current request: {qa.get('query','')}. Target tool: {schema.get('name','')}. Target parameter: {p}. Meaning: {d.get('description','')}. Type: {d.get('type','')}"


def build_slots(enc):
    td=Path(tempfile.gettempdir())/'mab_learned_address';td.mkdir(exist_ok=True)
    qp=td/'qa.jsonl';cp=td/'conv.jsonl'
    base.fetch(base.BASE+'qa_dataset.jsonl',qp);base.fetch(base.BASE+'toolmem_conversation.jsonl',cp)
    qas=list(base.load_jsonl(qp))[:DEV_END]
    sessions,by=base.build_session_map(cp)
    slots=[];missing=[]
    for qi,qa in enumerate(qas):
        ses=base.find_session(qa,sessions,by)
        if ses is None:
            missing.append(qi);continue
        eps=es.episodes(ses)
        schema=qa.get('target_tool_schema') or {}; target_tool=str(schema.get('name',''))
        props=((schema.get('parameters') or {}).get('properties') or {})
        gold=((qa.get('tool_call') or {}).get('arguments') or {})
        grounding=((qa.get('tool_call') or {}).get('grounding_info') or {})
        for p,g in gold.items():
            d=props.get(p) or {}; target=target_text(qa,p,d)
            picked=es.retrieve_episodes(enc,eps,target,TOP_EPISODES)
            cands=[]
            for erank,ep,esim in picked:
                for c in af.occurrences(ep,erank,esim):
                    if c['kind']=='text_span':
                        continue
                    cands.append(c)
            if not cands:
                slots.append({'qi':qi,'qa':qa,'p':p,'d':d,'gold':g,'grounding':str((grounding.get(p) or {}).get('type','unknown')),'cands':[],'features':[],'labels':[]})
                continue
            texts=[target]+[c['address'][:1400] for c in cands]
            E=enc.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
            sims=E[1:]@E[0]
            max_turn=max([int(c.get('turn',-1)) for c in cands]+[1])
            feats=[];labels=[]
            for i,c in enumerate(cands):
                k=str(c.get('key',''));kl=leaf(k);ct=str(c.get('tool',''));addr=str(c.get('address',''))
                feats.append({
                    'semantic_sim':float(sims[i]),
                    'slot_leaf_exact':float(canon(kl)==canon(p) and bool(kl)),
                    'slot_leaf_jacc':jacc(kl,p),
                    'slot_desc_jacc':jacc(k, str(p)+' '+str(d.get('description',''))),
                    'tool_exact':float(bool(ct) and canon(ct)==canon(target_tool)),
                    'tool_jacc':jacc(ct,target_tool),
                    'query_address_jacc':jacc(str(qa.get('query','')),addr),
                    'desc_address_jacc':jacc(str(d.get('description','')),addr),
                    'episode_rank':float(c.get('rank',-1)),
                    'episode_sim':float(c.get('episode_sim',0.0)),
                    'recency':float(int(c.get('turn',-1))/max(1,max_turn)),
                    'path_depth':float(k.count('.')+k.count('[')),
                    'kind_'+str(c.get('kind','unknown')):1.0,
                    'role_user':float(addr.startswith('role user')),
                    'role_assistant':float(addr.startswith('role assistant')),
                    'role_tool':float(addr.startswith('role tool')),
                    'type_'+str(d.get('type','')).lower():1.0,
                })
                labels.append(int(base.norm(c['value'])==base.norm(g)))
            slots.append({'qi':qi,'qa':qa,'p':p,'d':d,'gold':g,'grounding':str((grounding.get(p) or {}).get('type','unknown')),'cands':cands,'features':feats,'labels':labels})
    return slots,missing


def evaluate(slots,clf,vec,lo,hi):
    by=defaultdict(lambda:Counter(n=0,covered=0,top1=0,top3=0,top5=0))
    samples=[]
    for s in slots:
        if not (lo<=s['qi']<hi):continue
        typ=s['grounding'];m=by[typ];m['n']+=1
        if not s['cands']:continue
        positives=[i for i,y in enumerate(s['labels']) if y]
        m['covered']+=int(bool(positives))
        X=vec.transform(s['features']);prob=clf.predict_proba(X)[:,1]
        order=np.argsort(-prob)
        m['top1']+=int(bool(positives) and int(order[0]) in positives)
        m['top3']+=int(bool(positives) and any(int(i) in positives for i in order[:3]))
        m['top5']+=int(bool(positives) and any(int(i) in positives for i in order[:5]))
        if s['grounding']=='explicit' and len(samples)<10 and positives and int(order[0]) not in positives:
            rows=[]
            for i in order[:4]:
                c=s['cands'][int(i)]
                rows.append({'p':float(prob[int(i)]),'kind':c['kind'],'key':c['key'],'tool':c['tool'],'value':c['value'],'address':c['address'][:240]})
            samples.append({'qa_id':s['qa'].get('qa_id'),'parameter':s['p'],'gold':s['gold'],'top':rows,'positive_addresses':[s['cands'][i]['address'][:240] for i in positives[:3]]})
    packed={}
    for typ,c in by.items():
        n=max(1,c['n']);cov=max(1,c['covered'])
        packed[typ]={'n':c['n'],'coverage':c['covered']/n,'top1_all':c['top1']/n,'top3_all':c['top3']/n,'top5_all':c['top5']/n,'top1_given_covered':c['top1']/cov,'top3_given_covered':c['top3']/cov,'top5_given_covered':c['top5']/cov}
    return packed,samples


def main():
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    slots,missing=build_slots(enc)
    rows=[];ys=[];train_slots=0
    for s in slots:
        if s['qi']<TRAIN_END and any(s['labels']):
            rows.extend(s['features']);ys.extend(s['labels']);train_slots+=1
    vec=DictVectorizer(sparse=True);X=vec.fit_transform(rows)
    clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=1.0,random_state=42).fit(X,ys)
    tr,_=evaluate(slots,clf,vec,0,TRAIN_END);va,samples=evaluate(slots,clf,vec,TRAIN_END,DEV_END)
    names=vec.get_feature_names_out();w=clf.coef_[0]
    coef=[{'feature':str(names[i]),'weight':float(w[i])} for i in np.argsort(-np.abs(w))[:25]]
    result={'stage':'Mem2Act learned value-blind structured address resolver','architecture':'top-2 source_id episodes -> structured memory addresses with candidate value masked -> discriminative address scorer -> dereference only after address selection; no value text feature and no generative LLM','split':'QA001-70 train, QA071-100 validation; QA101-400 gold sealed','training':{'rows':len(ys),'positives':int(sum(ys)),'covered_slots':train_slots},'missing_zero_based':missing,'train':tr,'validation':va,'top_abs_coefficients':coef,'samples':samples,'guardrail':'Candidate value is not used as a model feature. Gold on QA001-70 labels address correctness; QA071-100 only scores validation. QA101-400 gold remains sealed.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('MEM2ACT_LEARNED_ADDRESS='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
