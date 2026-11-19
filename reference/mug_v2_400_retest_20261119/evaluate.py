from adapter import *
from common import _create_libero_env, _predict_action_chunk
import capture, h5py, hashlib, os, shutil, traceback, platform, importlib.metadata, subprocess
M=json.loads((OUT/'manifest.json').read_text())
def write_json(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
assert hashlib.sha256(Path(M['adapter']).read_bytes()).hexdigest()==M['adapter_sha256']
assert subprocess.check_output(['findmnt','-n','-o','UUID','--target',str(OUT)],text=True).strip()=='58B8D8C5B8D8A334'
policy,pre,post,ep,eo,meta=setup(M['adapter']);policy.eval();policy.requires_grad_(False)
capture.policy=policy;cap=capture.Capture();save=capture.save_array
write_json(OUT/'environment.json',{'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','mujoco','robosuite','transformers','h5py']},'model':meta,'manifest':M,'gpu':torch.cuda.get_device_name()})
P=OUT/'episodes';P.mkdir(exist_ok=True);rows=[];run_started=time.monotonic()
def run(layout,pilot=False):
 dest=P/layout['id'];dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():
  assert (dest/'features.h5').exists() and (dest/'video_validation.json').exists()
  return json.loads((dest/'result.json').read_text())
 assert shutil.disk_usage(OUT).free>30*1024**3
 task,states=load_case_task(case,benchmark)
 env,_=create_libero_env_with_retry(lambda:_create_libero_env(task,360,195,bddl_root=get_libero_path('bddl_files'),env_type=OffScreenRenderEnv),policy_seed=195,reseed=reseed)
 env=wrap_case_env(env,case);env.seed(195)
 scenario=copy.deepcopy(case['scenario']);scenario['change']['delta_position_m']=layout['delta_m'];scenario['setup']=[copy.deepcopy(scenario['change'])]
 instruction='put the white mug on the left plate'
 w=CosmosInterventionEnv(env=env,task_description=instruction,scenario=scenario,arm='control',trace_path=dest/'trace.jsonl',original_task_index=235,init_state_index=0)
 w.configure_episode(task_suite_name='libero_10',policy_seed=195,query_interval=30,max_policy_steps=520)
 writer=None;started=time.monotonic();timeline=[];physical=[];executed=[];qs=[]
 try:
  w.reset();obs=w.set_init_state(states[0]);w.backend.env.parsed_problem['goal_state']=[['on','porcelain_mug_1','plate_1']]
  for _ in range(w.warmup_steps):obs,_,_,_=w.step(get_libero_dummy_action())
  for robot in env.robots:robot.controller.use_delta=False
  objs=w.backend.env.object_states_dict;m=objs['porcelain_mug_1'];initial=m.get_geom_state()['pos'].tolist()
  contacts=[n for n in ['white_yellow_mug_1','red_coffee_mug_1','plate_1','plate_2'] if objs[n].check_contact(m)]
  assert not contacts,contacts
  assert np.linalg.norm(np.array(initial[:2])-layout['target_xy_m'])<1e-5
  writer=imageio.get_writer(str(dest/'rollout.mp4'),fps=20,codec='libx264',quality=7,macro_block_size=2)
  queue=collections.deque();first_goal=None;first_grasp=None;stable=0;lift_run=0;lifted=False
  with h5py.File(dest/'features.h5.partial','w') as f:
   f.attrs.update(episode_id=layout['id'],level=layout['level'],instruction=instruction,schema_version=2,adapter_sha256=M['adapter_sha256'],action_layer_indices=cap.layers)
   save(f,'initial_sim_state',env.sim.get_state().flatten());save(f,'camera_fovy',env.sim.model.cam_fovy)
   f.attrs['camera_names']=json.dumps([env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)])
   for i in range(520):
    if not queue:
     kw=dict(raw_obs=obs,env=env,instruction=instruction,policy=policy,preprocessor=pre,postprocessor=post,env_preprocessor=ep,env_postprocessor=eo,query_interval=30,query_seed=195+i,action_key=ACTION)
     control=None
     if pilot:
      cap.active=False;control=_predict_action_chunk(**kw)
     cap.start();t0=time.monotonic();actions=_predict_action_chunk(**kw);cap.active=False
     assert np.isfinite(actions).all()
     if control is not None:assert np.array_equal(control,actions),'Feature hooks changed actions'
     assert cap.iteration+1==policy.config.num_denoising_steps
     g=f.create_group(f'queries/{i:04d}');g.attrs.update(policy_step=i,query_seed=195+i,query_index=len(qs))
     for k,v in cap.values.items():
      if v.is_floating_point():assert torch.isfinite(v).all(),k
      save(g,k,v)
     for k in ['vision/valid_view_tokens','fusion/vlm_features','fusion/aux_visual_inputs','projected/vlm_proj','projected/aux_visual_proj']+[f'action_layers/block_{x:02d}' for x in cap.layers]:assert k in g,k
     for k,v in obs.items():
      if isinstance(v,(np.ndarray,float,int,bool)):save(g,('raw_images/' if k.endswith('_image') else 'raw_observation/')+k,v)
     for k,v in capture.state(w,obs).items():save(g,'physical/'+k,v)
     save(g,'actions/predicted_control',actions)
     qs.append({'step':i,'seed':195+i,'seconds':time.monotonic()-t0,'hook_control_exact':True if pilot else None})
     w.record_policy_query(actions,instruction=instruction,source='model');queue.extend(actions);cap.values={};f.flush()
    action=np.asarray(queue.popleft());executed.append(action.copy());obs,_,done,_=w.step(action.tolist());st=capture.state(w,obs);physical.append(st)
    if done and first_goal is None:first_goal=i+1
    if st['grasped'] and first_grasp is None:first_grasp=i+1
    lift_run=lift_run+1 if st['grasped'] and st['mug_xyz'][2]>initial[2]+.04 else 0;lifted=lifted or lift_run>=5
    stable=stable+1 if done and not st['grasped'] and st['speed_m_s']<.01 else 0
    timeline.append({'step':i+1,'goal':bool(done),'grasped':st['grasped'],'xy_error_m':st['xy_error_m'],'speed_m_s':st['speed_m_s'],'mug_xyz':st['mug_xyz'].tolist(),'eef_xyz':st['eef_xyz'].tolist(),'stable_count':stable})
    writer.append_data(frame(obs,'LoRA coverage v2 '+layout['id'],i+1,'Fresh closed-loop policy | seed 195'))
    if i%30==0:write_json(OUT/'current_episode.json',{'id':layout['id'],'step':i+1,'queries':len(qs),'seconds':time.monotonic()-started})
    if stable>=20:break
   for k in physical[0]:save(f,'steps/'+k,np.array([s[k] for s in physical]))
   save(f,'steps/action',np.array(executed));f.attrs['released_stable_success']=stable>=20;f.attrs['complete']=True
  writer.close();writer=None
  reader=imageio.get_reader(str(dest/'rollout.mp4'));frames=0
  try:
   for image in reader:frames+=1
  finally:reader.close()
  assert frames==len(timeline),(frames,len(timeline))
  write_json(dest/'video_validation.json',{'decoded_frames':frames,'expected_frames':len(timeline),'passed':True,'fps':20})
  os.replace(dest/'features.h5.partial',dest/'features.h5')
  with h5py.File(dest/'features.h5','r') as f:
   assert len(f['queries'])==len(qs) and len(f['steps/action'])==len(timeline)
  result={'layout':layout,'initial_mug':initial,'initial_other_contacts':contacts,'native_success':first_goal is not None,'released_stable_success':stable>=20,'lifted_grasp_success':bool(lifted),'steps':i+1,'first_goal_step':first_goal,'first_grasp_step':first_grasp,'final':timeline[-1],'elapsed_seconds':time.monotonic()-started,'queries':qs,'pilot':pilot,'adapter_sha256':M['adapter_sha256'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
  w.record_outcome(result['native_success']);write_json(dest/'metrics.json',timeline);write_json(dest/'result.json',result)
  print('RESULT',layout['id'],result['released_stable_success'],result['steps'],round(result['elapsed_seconds'],1),flush=True)
  return result
 finally:
  cap.active=False;cap.values={}
  if writer:writer.close()
  env.close()
try:
 for layout in M['positions']:
  rows.append(run(layout,pilot=layout['id'].endswith('_00')))
  write_json(OUT/'progress.json',{'phase':'running','completed':len(rows),'total':400,'latest':layout['id'],'stable_successes':sum(r['released_stable_success'] for r in rows),'elapsed_seconds':time.monotonic()-run_started})
  write_json(OUT/'results.json',rows)
 summary={'status':'complete','episodes':len(rows),'stable_successes':sum(r['released_stable_success'] for r in rows),'native_successes':sum(r['native_success'] for r in rows),'groups':{str(j):{'n':sum(r['layout']['level']==j for r in rows),'successes':sum(r['released_stable_success'] for r in rows if r['layout']['level']==j)} for j in range(4)},'baseline_successes':sum(r['success'] for r in M['baseline']),'all_videos_decoded':True,'wall_seconds':time.monotonic()-run_started}
 assert len(rows)==400;write_json(OUT/'complete.json',summary);write_json(OUT/'progress.json',summary)
 print('ALL_400_COMPLETE',json.dumps(summary),flush=True)
except BaseException as e:
 write_json(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'completed':len(rows)})
 raise
