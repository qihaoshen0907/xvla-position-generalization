from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
rows=[]
for name in ['mug_5x5_100_base_20261118','mug_rings_100_base_20261118']:
 for r in json.loads((P.parent/name/'episodes/results.json').read_text()):
  d=P.parent/name/'episodes'/r['layout']['id']
  rows.append({'id':r['layout']['id'],'source':str(d),'level':r['layout']['level'],'success':r['released_stable_success'],'layout':r['layout'],'steps':r['steps'],'hashes':{n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['result.json','metrics.json','trace.jsonl']}})
assert len(rows)==400 and len({r['id'] for r in rows})==400
pilot=[]
for l in range(4):
 rr=[r for r in rows if r['level']==l]; ss=[r for r in rr if r['success']];ff=[r for r in rr if not r['success']]
 chosen=ss[:1]+ff[:1]
 while len(chosen)<3:
  remain=[r for r in rr if r not in chosen]
  chosen.append(max(remain,key=lambda r:min(np.linalg.norm(np.array(r['layout']['target_xy_m'])-x['layout']['target_xy_m']) for x in chosen)))
 pilot+=chosen
m={'episodes':rows,'pilot_ids':[r['id'] for r in pilot],'pilot_rule':'For each level first success if present, first failure, then maximize nearest selected XY distance until3. R3 has only failures.','replay_tolerance_m':1e-8,'action_tolerance':1e-6,'sampling_scope':'400 existing episodes, no replacement of failures','query_timing':'pre-action observation at saved policy_step; step0 is before first policy action after warmup','feature_schema_version':1}
(P/'manifest.json').write_text(json.dumps(m,indent=2));print('PILOT',m['pilot_ids'])
