from model import *
from common import _create_libero_env, _predict_action_chunk
import h5py,hashlib,shutil,os,platform
M=json.loads((OUT/'manifest.json').read_text())
policy,pre,post,ep,eo=setup()
class Capture:
 def __init__(self):
  self.active=False;self.values={};self.iteration=-1
  self.depth=len(policy.model.transformer.blocks);self.layers=[0,self.depth//2-1,self.depth-1]
  def wrap(obj,name,callback):
   original=getattr(obj,name)
   def f(*a,**kw):
    out=original(*a,**kw)
    if self.active:callback(a,kw,out)
    return out
   setattr(obj,name,f)
  wrap(policy,'_build_model_inputs',lambda a,k,o:self.mapping('inputs',o))
  wrap(policy.model.vlm,'get_image_features',lambda a,k,o:self.put('vision/valid_view_tokens',o.pooler_output))
  wrap(policy.model,'forward_vlm',lambda a,k,o:self.mapping('fusion',o))
  wrap(policy.model,'generate_actions',lambda a,k,o:self.put('actions/model_output_20d',o))
  wrap(policy.model.action_space,'preprocess',lambda a,k,o:self.put('generation/initial_noise_raw',a[1]) if self.iteration==-1 else None)
  def trans_pre(mod,args,kw):
   if not self.active:return
   self.iteration+=1;self.put(f'generation/iteration_{self.iteration:02d}/t',kw['t'])
   if self.iteration==0:self.put('generation/initial_noise_after_action_preprocess',kw['action_with_noise'])
  policy.model.transformer.register_forward_pre_hook(trans_pre,with_kwargs=True)
  def trans_post(mod,args,out):
   if self.active:self.put(f'generation/iteration_{self.iteration:02d}/prediction',out)
  policy.model.transformer.register_forward_hook(trans_post)
  for name in ['vlm_proj','aux_visual_proj']:
   getattr(policy.model.transformer,name).register_forward_hook(self.last_hook('projected/'+name))
  for ix in self.layers:policy.model.transformer.blocks[ix].register_forward_hook(self.last_hook(f'action_layers/block_{ix:02d}',action_only=True))
 def put(self,key,value):
  if isinstance(value,torch.Tensor):self.values[key]=value.detach().cpu().clone()
 def mapping(self,prefix,values):
  for k,v in values.items():self.put(prefix+'/'+k,v)
 def last_hook(self,key,action_only=False):
  def hook(mod,args,out):
   if self.active and self.iteration==policy.config.num_denoising_steps-1:self.put(key,out[:,:policy.config.chunk_size] if action_only else out)
  return hook
 def start(self):self.values={};self.iteration=-1;self.active=True
cap=Capture()
def save_array(g,key,x):
 dtype=None
 if isinstance(x,torch.Tensor):
  dtype=str(x.dtype)
  x=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
 else:x=np.asarray(x)
 if np.issubdtype(x.dtype,np.floating):assert np.isfinite(x).all(),key
 ds=g.create_dataset(key,data=x,compression='gzip',compression_opts=1,shuffle=True) if x.ndim else g.create_dataset(key,data=x)
 if dtype:ds.attrs['torch_dtype']=dtype
 return ds

def state(w,obs):
 e=w.backend.env;sim=e.sim;obj=e.object_states_dict
 mug=obj['porcelain_mug_1'].get_geom_state()['pos'].copy();plate=obj['plate_1'].get_geom_state()['pos'].copy();b=e.obj_body_id['porcelain_mug_1'];speed=float(np.linalg.norm(sim.data.get_body_xvelp(sim.model.body_id2name(b))))
 return {'mug_xyz':mug,'plate_xyz':plate,'eef_xyz':np.array(obs['robot0_eef_pos']),'eef_rotation':e.robots[0].controller.ee_ori_mat.copy(),'mug_quat':sim.data.body_xquat[b].copy(),'grasped':bool(w.backend.is_grasping('porcelain_mug_1')),'goal':bool(obj['plate_1'].check_ontop(obj['porcelain_mug_1'])),'speed_m_s':speed,'xy_error_m':float(np.linalg.norm(mug[:2]-plate[:2])),'sim_state':sim.get_state().flatten().copy(),'qpos':sim.data.qpos.copy(),'qvel':sim.data.qvel.copy(),'camera_xpos':sim.data.cam_xpos.copy(),'camera_xmat':sim.data.cam_xmat.copy()}

def run(row,is_pilot):
 dest=OUT/'episodes'/row['id'];dest.mkdir(parents=True,exist_ok=True)
 if (dest/'validation.json').exists():
  v=json.loads((dest/'validation.json').read_text());assert v['passed'];return v
 source=Path(row['source'])
 for n,h in row['hashes'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h
 r=json.loads((source/'result.json').read_text());tr=json.loads((source/'trace.jsonl').read_text().splitlines()[-1]);metrics=json.loads((source/'metrics.json').read_text());queries={q['policy_step']:q for q in tr['policy_queries']}
 task,states=load_case_task(case,benchmark)
 env,_=create_libero_env_with_retry(lambda:_create_libero_env(task,360,195,bddl_root=get_libero_path('bddl_files'),env_type=OffScreenRenderEnv),policy_seed=195,reseed=reseed);env=wrap_case_env(env,case);env.seed(195)
 scenario=copy.deepcopy(case['scenario']);scenario['change']['delta_position_m']=row['layout']['delta_m'];scenario['setup']=[copy.deepcopy(scenario['change'])]
 instruction='put the white mug on the left plate'
 w=CosmosInterventionEnv(env=env,task_description=instruction,scenario=scenario,arm='control',trace_path=dest/'replay.trace.jsonl',original_task_index=235,init_state_index=0)
 w.configure_episode(task_suite_name='libero_10',policy_seed=195,query_interval=30,max_policy_steps=520)
 maxima={'eef_m':0.,'mug_qpos_m':0.,'mug_body_m':0.,'speed_m_s':0.,'action':0.,'hook_action':0.};qs=[];physical=[];started=time.monotonic()
 try:
  w.reset();obs=w.set_init_state(states[0]);w.backend.env.parsed_problem['goal_state']=[['on','porcelain_mug_1','plate_1']]
  for _ in range(w.warmup_steps):obs,_,_,_=w.step(get_libero_dummy_action())
  for robot in env.robots:robot.controller.use_delta=False
  with h5py.File(dest/'features.h5.partial','w') as f:
   f.attrs['episode_id']=row['id'];f.attrs['source']=row['source'];f.attrs['episode_success']=row['success'];f.attrs['level']=row['level'];f.attrs['instruction']=instruction;f.attrs['schema_version']=1;f.attrs['action_layer_indices']=cap.layers;f.attrs['feature_generation_iteration']=policy.config.num_denoising_steps-1
   f.attrs['camera_names']=json.dumps([env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)])
   save_array(f,'camera_fovy',env.sim.model.cam_fovy);save_array(f,'initial_sim_state',env.sim.get_state().flatten())
   for i,old in enumerate(tr['executed_actions']):
    if i in queries:
     if shutil.disk_usage(OUT).free<3*1024**3:raise RuntimeError('Less than3GiB free: stop before exhausting disk')
     q=queries[i];g=f.create_group(f'queries/{i:04d}');g.attrs['policy_step']=i;g.attrs['query_seed']=195+i;g.attrs['query_index']=q['query_index'];g.attrs['episode_success']=row['success']
     # Uninstrumented control at every pilot query; reset RNG inside helper makes it matched.
     if is_pilot:
      cap.active=False
      control=_predict_action_chunk(raw_obs=obs,env=env,instruction=instruction,policy=policy,preprocessor=pre,postprocessor=post,env_preprocessor=ep,env_postprocessor=eo,query_interval=30,query_seed=195+i,action_key=ACTION)
     cap.start()
     predicted=_predict_action_chunk(raw_obs=obs,env=env,instruction=instruction,policy=policy,preprocessor=pre,postprocessor=post,env_preprocessor=ep,env_postprocessor=eo,query_interval=30,query_seed=195+i,action_key=ACTION)
     cap.active=False
     err=float(np.max(np.abs(predicted-np.asarray(q['actions']))));maxima['action']=max(maxima['action'],err)
     if is_pilot:
      hookerr=float(np.max(np.abs(predicted-control)));maxima['hook_action']=max(maxima['hook_action'],hookerr);assert hookerr==0.,('hook changes output',row['id'],i,hookerr)
     assert err<=M['action_tolerance'],('action mismatch',row['id'],i,err)
     assert cap.iteration+1==policy.config.num_denoising_steps
     for k,v in cap.values.items():
      if v.is_floating_point():assert torch.isfinite(v).all(),k
      save_array(g,k,v)
     required=['vision/valid_view_tokens','fusion/vlm_features','fusion/aux_visual_inputs','projected/vlm_proj','projected/aux_visual_proj']+[f'action_layers/block_{x:02d}' for x in cap.layers]
     for k in required:assert k in g,k
     for k in ['agentview_image','robot0_eye_in_hand_image']:save_array(g,'raw_images/'+k,obs[k])
     for k,v in obs.items():
      if not k.endswith('_image') and isinstance(v,(np.ndarray,float,int,bool)):save_array(g,'raw_observation/'+k,v)
     st=state(w,obs)
     for k,v in st.items():save_array(g,'physical/'+k,v)
     if st['goal'] and not st['grasped']:stage='placed_or_releasing'
     elif st['grasped']:stage='transport' if st['mug_xyz'][2]>r['initial_mug'][2]+.04 else 'grasped_low'
     else:stage='ungrasped_approach_or_recovery'
     g.attrs['stage_heuristic']=stage
     save_array(g,'actions/recomputed_control',predicted);save_array(g,'actions/original_control',q['actions']);save_array(g,'actions/executed_control',[a['action'] for a in tr['executed_actions'][i:i+30]])
     tokens=cap.values['vision/valid_view_tokens'].shape[1];g.attrs['primary_visual_token_count']=tokens;g.attrs['fusion_token_order']='primary visual tokens, then text input_ids';g.attrs['aux_token_order']='remaining configured views flattened; includes masked empty views';g.attrs['action_token_order']='future action offsets 0..29';g.attrs['valid_view_order']='configured image_mask true entries in order; see inputs/image_mask'
     qs.append({'step':i,'query_index':q['query_index'],'seed':195+i,'action_max_abs_error':err,'stage_heuristic':stage});f.flush()
    obs,_,done,_=w.step(old['action']);now=w.executed_actions[-1];st=state(w,obs);om=metrics[i]
    for key,a,b in [('eef_m',now['eef_position_m'],old['eef_position_m']),('mug_qpos_m',now['trigger_entity_position_m'],old['trigger_entity_position_m']),('mug_body_m',st['mug_xyz'],om['mug_xyz']),('speed_m_s',st['speed_m_s'],om['speed_m_s'])]:maxima[key]=max(maxima[key],float(np.max(np.abs(np.array(a)-b))))
    assert max(maxima[k] for k in ['eef_m','mug_qpos_m','mug_body_m','speed_m_s'])<=M['replay_tolerance_m'],('state mismatch',row['id'],i,maxima)
    assert bool(done)==om['goal'] and st['grasped']==om['grasped'],('discrete mismatch',row['id'],i)
    physical.append(st)
   for k in physical[0]:save_array(f,'steps/'+k,np.array([x[k] for x in physical]))
   save_array(f,'steps/action',np.array([a['action'] for a in tr['executed_actions']]))
   assert len(qs)==len(queries);f.attrs['validated']=True
  os.replace(dest/'features.h5.partial',dest/'features.h5')
  v={'id':row['id'],'passed':True,'pilot':is_pilot,'steps':len(physical),'queries':len(qs),'max_abs_errors':maxima,'query_checks':qs,'size_bytes':(dest/'features.h5').stat().st_size,'elapsed_seconds':time.monotonic()-started,'original_pixels_available':False,'note':'Replay state and recomputed control checked; original raw pixels were not saved. Hook control run at all pilot queries.'}
  (dest/'validation.json').write_text(json.dumps(v,indent=2));print('PASS',row['id'],len(qs),maxima,flush=True);return v
 finally:cap.active=False;env.close()

pilot=[next(r for r in M['episodes'] if r['id']==id) for id in M['pilot_ids']]
completed=[]
for r in pilot:
 completed.append(run(r,True));(OUT/'progress.json').write_text(json.dumps({'phase':'pilot','completed':len(completed),'total':400,'latest':r['id']}))
assert len(completed)==12 and all(r['passed'] for r in completed)
size_per_query=sum(r['size_bytes'] for r in completed)/sum(r['queries'] for r in completed)
expected_queries=sum((r['steps']+29)//30 for r in M['episodes'])
projected=size_per_query*expected_queries
free=shutil.disk_usage(OUT).free
assert projected*1.3<free+sum(r['size_bytes'] for r in completed),('insufficient projected disk space',projected,free)
(OUT/'pilot_gate.json').write_text(json.dumps({'passed':True,'pilot_ids':M['pilot_ids'],'queries':sum(r['queries'] for r in completed),'all_hook_outputs_exact':all(r['max_abs_errors']['hook_action']==0 for r in completed),'projected_total_bytes':projected,'expected_queries':expected_queries,'free_bytes_at_gate':free},indent=2))
print('PILOT_GATE_PASSED',projected,expected_queries,flush=True)
for r in M['episodes']:
 if r['id'] in M['pilot_ids']:continue
 completed.append(run(r,False));(OUT/'progress.json').write_text(json.dumps({'phase':'full','completed':len(completed),'total':400,'latest':r['id'],'queries':sum(x['queries'] for x in completed)}))
(OUT/'collection_complete.json').write_text(json.dumps({'episodes':len(completed),'queries':sum(r['queries'] for r in completed),'bytes':sum(r['size_bytes'] for r in completed),'all_passed':all(r['passed'] for r in completed),'max_errors':{k:max(r['max_abs_errors'][k] for r in completed) for k in completed[0]['max_abs_errors']}},indent=2))
print('COLLECTION_COMPLETE',flush=True)
