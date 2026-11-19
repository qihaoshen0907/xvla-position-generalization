from common import *
import h5py
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
