from pathlib import Path
from collections import defaultdict, Counter
import json, re
import numpy as np
import semantic_world_binding_cv as swm

OUT=Path(__file__).with_name('semantic_binding_layer_audit_result.json')
N=100

def op_prior(op,p,d):
    op=str(op); meta=(str(p)+' '+str((d or {}).get('description',''))).lower()
    score=0.0
    if op=='identity': score+=1.0
    if 'code' in meta or 'symbol' in meta or 'ticker' in meta or 'state' in meta or 'country' in meta:
        if any(x in op for x in ['code','alpha_2','alpha_3']): score+=1.4
    if 'color' in meta and 'color' in op: score+=1.4
    if any(x in meta for x in ['integer','number','id']) and 'numeric' in op: score+=0.5
    if op.startswith('schema:'): score+=0.4
    if op.startswith('ontology:'): score+=0.3
    return score

def main():
    enc=swm.CachedEncoder()
    slots,report,world_sizes=swm.build(enc)
    stats=defaultdict(lambda:Counter(n=0,covered=0,semantic1=0,episode_oracle1=0,field_oracle1=0,address_oracle1=0,transform_required=0,wrong_episode=0,right_episode_wrong_field=0,right_field_wrong_transform=0))
    samples=[]
    for s in slots:
        typ=s['typ']; m=stats[typ]; m['n']+=1
        if not s['cands']: continue
        pos=[i for i,y in enumerate(s['labels']) if y]
        if not pos: continue
        m['covered']+=1
        sem=np.asarray([float(f.get('semantic',0.0)) for f in s['feats']],dtype=np.float32)
        order=np.argsort(-sem); top=int(order[0]); top_ok=top in pos; m['semantic1']+=int(top_ok)
        pos_eps={int(s['cands'][i].get('episode',-1)) for i in pos}
        pos_fields={(int(s['cands'][i].get('episode',-1)),str(s['cands'][i].get('field',''))) for i in pos}
        pos_addr={str(s['cands'][i].get('address','')) for i in pos}
        identity_positive=any(str(s['cands'][i].get('op','identity'))=='identity' for i in pos)
        m['transform_required']+=int(not identity_positive)
        # Episode oracle: assume entity/episode scope solved; field+transform still selected semantically.
        ep_idx=[i for i,c in enumerate(s['cands']) if int(c.get('episode',-1)) in pos_eps]
        if ep_idx:
            j=max(ep_idx,key=lambda i:float(sem[i])); m['episode_oracle1']+=int(j in pos)
        # Field oracle: assume correct episode+field role solved; operation/transform remains.
        fi=[i for i,c in enumerate(s['cands']) if (int(c.get('episode',-1)),str(c.get('field',''))) in pos_fields]
        if fi:
            # Break semantic ties with a generic, answer-blind operation prior.
            d={}
            score=[float(sem[i])+0.08*op_prior(c=s['cands'][i].get('op','identity'),p=s['p'],d=d) if False else 0 for i in []]
            def fs(i):
                c=s['cands'][i]
                return float(sem[i])+0.08*op_prior(c.get('op','identity'),s['p'],{})
            j=max(fi,key=fs); m['field_oracle1']+=int(j in pos)
        # Address oracle: exact provenance/address solved; only transform choice remains.
        ai=[i for i,c in enumerate(s['cands']) if str(c.get('address','')) in pos_addr]
        if ai:
            j=max(ai,key=lambda i:float(sem[i])+0.08*op_prior(s['cands'][i].get('op','identity'),s['p'],{})); m['address_oracle1']+=int(j in pos)
        if not top_ok:
            tc=s['cands'][top]
            if int(tc.get('episode',-1)) not in pos_eps:
                m['wrong_episode']+=1; failure='wrong_episode'
            elif (int(tc.get('episode',-1)),str(tc.get('field',''))) not in pos_fields:
                m['right_episode_wrong_field']+=1; failure='right_episode_wrong_field'
            else:
                m['right_field_wrong_transform']+=1; failure='right_field_wrong_transform'
            if len(samples)<24:
                samples.append({'qa_id':s['qa_id'],'parameter':s['p'],'grounding':typ,'failure':failure,'gold':s['g'],'top':{'value':tc.get('value'),'field':tc.get('field'),'episode':tc.get('episode'),'op':tc.get('op'),'semantic':float(sem[top])},'positive':[{'field':s['cands'][i].get('field'),'episode':s['cands'][i].get('episode'),'op':s['cands'][i].get('op')} for i in pos[:6]]})
    packed={}
    for typ,c in stats.items():
        n=max(1,c['n']); cov=max(1,c['covered']); misses=max(1,c['covered']-c['semantic1'])
        packed[typ]={
            'n':c['n'],'candidate_coverage':c['covered']/n,
            'semantic_top1_all':c['semantic1']/n,
            'episode_oracle_top1_all':c['episode_oracle1']/n,
            'field_oracle_top1_all':c['field_oracle1']/n,
            'address_oracle_top1_all':c['address_oracle1']/n,
            'transform_required_rate_among_covered':c['transform_required']/cov,
            'failure_share_among_semantic_misses':{
                'wrong_episode':c['wrong_episode']/misses,
                'right_episode_wrong_field':c['right_episode_wrong_field']/misses,
                'right_field_wrong_transform':c['right_field_wrong_transform']/misses,
            }
        }
    result={'stage':'Semantic world-model binding layer decomposition','split':'QA001-100 development only; QA101-400 gold remains sealed','architecture':'same query-independent semantic world state and value-blind TOPK candidates as semantic_world_binding_cv; gold is used only to identify which layer contains the correct provenance for oracle decomposition','results':packed,'sample_failures':samples,'world_size_mean':float(np.mean(world_sizes)),'repair_report':report,'guardrail':'No hidden labels read. Oracle episode/field/address are diagnostics only and are not candidate/ranking features.'}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print('SEMANTIC_BINDING_LAYER_AUDIT='+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
