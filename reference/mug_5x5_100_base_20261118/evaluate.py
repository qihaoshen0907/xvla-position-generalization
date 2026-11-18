from model import *
from common import _create_libero_env, _predict_action_chunk
import argparse
parser=argparse.ArgumentParser();args=parser.parse_args()
policy,pre,post,ep,eo=setup()
P=OUT/'episodes';P.mkdir(exist_ok=True)
rows=[]
for layout in json.loads((OUT/'positions.json').read_text())['positions']:
 dest=P/layout['id'];dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():rows.append(json.loads((dest/'result.json').read_text()));continue
 task,states=load_case_task(case,benchmark)
 env,desc=create_libero_env_with_retry(lambda:_create_libero_env(task,360,195,bddl_root=get_libero_path('bddl_files'),env_type=OffScreenRenderEnv),policy_seed=195,reseed=reseed)
 env=wrap_case_env(env,case);env.seed(195)
 scenario=copy.deepcopy(case['scenario']);scenario['change']['delta_position_m']=layout['delta_m'];scenario['setup']=[copy.deepcopy(scenario['change'])]
 instruction='put the white mug on the left plate'
 w=CosmosInterventionEnv(env=env,task_description=instruction,scenario=scenario,arm='control',trace_path=dest/'trace.jsonl',original_task_index=235,init_state_index=0)
 w.configure_episode(task_suite_name='libero_10',policy_seed=195,query_interval=30,max_policy_steps=520)
 w.reset();obs=w.set_init_state(states[0]);w.backend.env.parsed_problem['goal_state']=[['on','porcelain_mug_1','plate_1']]
 for _ in range(w.warmup_steps):obs,_,_,_=w.step(get_libero_dummy_action())
 for robot in env.robots:robot.controller.use_delta=False
 objs=w.backend.env.object_states_dict
 initial_mug=objs['porcelain_mug_1'].get_geom_state()['pos'].tolist()
 initial_other_contacts=[n for n in ['white_yellow_mug_1','red_coffee_mug_1','plate_1','plate_2'] if objs[n].check_contact(objs['porcelain_mug_1'])]
 writer=imageio.get_writer(str(dest/'rollout.mp4'),fps=20,codec='libx264',quality=7,macro_block_size=2)
 queue=collections.deque();timeline=[];first_success=None;first_grasp=None;stable=0;lifted_run=0;lifted_success=False;started=time.monotonic()
 try:
  for i in range(520):
   if not queue:
    actions=_predict_action_chunk(raw_obs=obs,env=env,instruction=instruction,policy=policy,preprocessor=pre,postprocessor=post,env_preprocessor=ep,env_postprocessor=eo,query_interval=30,query_seed=195+i,action_key=ACTION)
    w.record_policy_query(actions,instruction=instruction,source='model');queue.extend(actions)
   obs,_,done,_=w.step(np.asarray(queue.popleft()).tolist())
   m=w.backend.env.object_states_dict['porcelain_mug_1'];pl=w.backend.env.object_states_dict['plate_1']
   pos=m.get_geom_state()['pos'];platepos=pl.get_geom_state()['pos'];grasp=bool(w.backend.is_grasping('porcelain_mug_1'))
   if done and first_success is None:first_success=i+1
   if grasp and first_grasp is None:first_grasp=i+1
   body=w.backend.env.obj_body_id['porcelain_mug_1'];speed=float(np.linalg.norm(w.backend.env.sim.data.get_body_xvelp(w.backend.env.sim.model.body_id2name(body))))
   lifted_run=lifted_run+1 if grasp and pos[2]>initial_mug[2]+.04 else 0
   lifted_success=lifted_success or lifted_run>=5
   stable=stable+1 if done and not grasp and speed<.01 else 0
   timeline.append({'step':i+1,'goal':bool(done),'grasped':grasp,'xy_error_m':float(np.linalg.norm(pos[:2]-platepos[:2])),'speed_m_s':speed,'mug_xyz':pos.tolist()})
   writer.append_data(frame(obs,'Base XVLA '+layout['id'],i+1,'XVLA model policy | no expert recovery'))
   if stable>=20:break
  assert not initial_other_contacts, initial_other_contacts
  assert np.linalg.norm(np.array(initial_mug[:2])-layout['target_xy_m'])<1e-5
  result={'layout':layout,'lifted_grasp_success':bool(lifted_success),'elapsed_seconds':time.monotonic()-started,'initial_mug':initial_mug,'initial_other_contacts':initial_other_contacts,'native_success':first_success is not None,'released_stable_success':stable>=20,'steps':i+1,'first_goal_step':first_success,'first_grasp_step':first_grasp,'final':timeline[-1]}
  (dest/'result.json').write_text(json.dumps(result,indent=2));(dest/'metrics.json').write_text(json.dumps(timeline));w.record_outcome(result['native_success']);rows.append(result)
  (P/'results.json').write_text(json.dumps(rows,indent=2));(OUT/'progress.json').write_text(json.dumps({'completed':len(rows),'total':100,'latest':layout['id'],'stable_successes':sum(r['released_stable_success'] for r in rows)}));print('RESULT',layout['id'],result['native_success'],result['released_stable_success'],flush=True)
 finally:writer.close();env.close()
(P/'results.json').write_text(json.dumps(rows,indent=2))
