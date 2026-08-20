from pathlib import Path
import re
import strong_banking77 as base

base.OUT = Path('researchstrong/slotvm_policy_result.json')
_orig_compile = base.compile_task

def _required(slot):
    schema = (slot.get('qa') or {}).get('target_tool_schema') or {}
    params = schema.get('parameters') or {}
    return set(params.get('required') or [])

def _policy_values(slot):
    p = str(slot.get('parameter',''))
    d = slot.get('def') or {}
    typ = str(d.get('type','')).lower()
    desc = str(d.get('description','')).lower()
    req = _required(slot)
    optional = p not in req
    out=[]
    # General null/default conventions only; no benchmark/task-specific constants.
    if optional and typ in ('array','list'): out.append(([], 'optional_empty_array'))
    if optional and typ in ('string','str'): out.append(('', 'optional_empty_string'))
    if typ in ('bool','boolean'): out.append((False, 'boolean_false_default'))
    pl = p.lower().replace('_','')
    if 'offset' in pl: out.append((0, 'offset_zero'))
    if pl in ('page','pageindex') or ('page' in pl and 'index' in pl): out.append((1, 'first_page_one'))
    if pl == 'index' and ('latest' in desc or 'most recent' in desc or 'starting from 0' in desc or 'start from 0' in desc): out.append((0, 'latest_index_zero'))
    if typ in ('int','integer','float','number') and ('starting from 0' in desc or 'start from 0' in desc) and ('latest' in desc or 'most recent' in desc): out.append((0, 'latest_zero'))
    return out

def compile_task(qa, session, enc):
    slots = _orig_compile(qa, session, enc)
    for s in slots:
        cand=list(s.get('candidates') or [])
        seen={base.norm(c.get('value')) for c in cand}
        # Keep explicit schema defaults/enums first, then general executor policy defaults.
        inserts=[]
        for v,src in _policy_values(s):
            nv=base.norm(v)
            if nv not in seen:
                inserts.append({'value':v,'source':src,'evidence':'general schema/type executor policy','priority':3})
                seen.add(nv)
        cand = cand[:3] + inserts + cand[3:]
        cand = cand[:32]
        for j,c in enumerate(cand): c['id']=f'C{j}'
        s['candidates']=cand
    return slots

base.compile_task = compile_task
base.main()
