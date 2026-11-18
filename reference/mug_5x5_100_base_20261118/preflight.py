"""Freeze uniform stratified positions before any model outcome is observed."""
from common import *
from common import _create_libero_env
P=OUT
rng=np.random.default_rng(2026111801)
task,states=load_case_task(case,benchmark)
env,_=create_libero_env_with_retry(lambda:_create_libero_env(task,360,195,bddl_root=get_libero_path('bddl_files'),env_type=OffScreenRenderEnv),policy_seed=195,reseed=reseed)
env=wrap_case_env(env,case);env.seed(195);env.reset();obs=env.set_init_state(states[0])
for _ in range(10):obs,_,_,_=env.step(get_libero_dummy_action())
sim=env.sim;initial=sim.get_state().flatten().copy();q0=sim.data.get_joint_qpos('porcelain_mug_1_joint0').copy()
# Use the previously verified post-warmup original mug XY as relocation origin.
origin=np.array([-.0874851104853002,-.15589814830207166])
center=np.array([-.1,-.15]);table_boxes=[]
for i in range(sim.model.ngeom):
 body=sim.model.body_id2name(int(sim.model.geom_bodyid[i])) or ''
 if 'living_room_table' not in body or int(sim.model.geom_type[i])!=6:continue
 ext=np.abs(sim.data.geom_xmat[i].reshape(3,3))@sim.model.geom_size[i]
 if ext[2]<.03 and min(ext[:2])>.3:table_boxes.append((sim.data.geom_xpos[i].copy(),ext))
assert table_boxes, 'table support geometry not found'
base=sim.data.body_xpos[sim.model.body_name2id('robot0_base')].copy()
all_rows=[];stats=[];area_draws=[]
for level,width in enumerate([.05]):
 candidates=[]
 while len(candidates)<3000:
  xy=center+rng.uniform(-width/2,width/2,size=2)
  if level and np.max(np.abs(xy-center))<=(width-.05)/2:continue
  candidates.append(xy)
 accepted=[];reasons=collections.Counter();flags=[]
 for ci,xy in enumerate(candidates):
  reason=None
  # Conservative 4.5 cm footprint margin; broad reach bound only, not a feasibility oracle.
  if not any(np.all(np.abs(xy-pos[:2])+.045<ext[:2]) for pos,ext in table_boxes):reason='outside_supported_table'
  if np.linalg.norm(np.r_[xy,q0[2]+.07]-base)>.8:reason='outside_coarse_reach_bound'
  sim.set_state_from_flattened(initial);q=q0.copy();q[:2]=xy;sim.data.set_joint_qpos('porcelain_mug_1_joint0',q);sim.forward()
  contacts=[]
  for c in sim.data.contact[:sim.data.ncon]:
   gids=[int(c.geom1),int(c.geom2)];names=[sim.model.geom_id2name(g) or '' for g in gids]
   own=[n.startswith('porcelain_mug_1') for n in names]
   if sum(own)!=1:continue
   other=gids[1 if own[0] else 0];body=sim.model.body_id2name(int(sim.model.geom_bodyid[other])) or ''
   if 'living_room_table' not in body:contacts.append(names)
  if contacts:reason='initial_non_support_contact'
  if reason:reasons[reason]+=1
  else:accepted.append((ci,xy))
  flags.append(reason is None)
 assert len(accepted)>=100
 # First 100 valid iid samples, no selection based on policy success.
 for j,(ci,xy) in enumerate(accepted[:100]):
  all_rows.append({'id':f'R{level}_{j:02d}','level':level,'outer_width_m':width,'candidate_index':ci,'target_xy_m':xy.tolist(),'delta_m':np.r_[xy-origin,0.].tolist()})
 area=width**2-(width-.05)**2 if level else width**2
 stats.append({'level':level,'outer_width_m':width,'candidate_count':len(candidates),'valid_count':len(accepted),'rejected':dict(reasons),'nominal_ring_area_m2':area,'estimated_valid_area_m2':area*len(accepted)/len(candidates),'selected':100})
 area_draws.append({'level':level,'positions':np.array(candidates).tolist(),'valid':flags})
 print('PREFLIGHT',stats[-1],flush=True)
sim.set_state_from_flattened(initial);sim.forward();env.close()
manifest={'sampling_seed':2026111801,'policy_seed':195,'center_xy_m':center.tolist(),'origin_xy_m':origin.tolist(),'instruction':'put the white mug on the left plate','query_interval':30,'budget':520,'positions':all_rows,'strata':stats,'filters':'table footprint with 4.5 cm margin; broad 0.8 m reach bound; actual MuJoCo initial contacts excluding table. No policy-outcome filtering. No claim of exhaustive reachability/planning proof.','known_lora_coordinates':'Continuous independent samples; no LoRA model used in this experiment.'}
(P/'positions.json').write_text(json.dumps(manifest,indent=2));(P/'area_samples.json').write_text(json.dumps(area_draws));print('FROZEN',len(all_rows),flush=True)
