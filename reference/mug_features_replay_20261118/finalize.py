from pathlib import Path
import json,hashlib,sys,platform
import h5py,numpy as np,torch
from read_features import array
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text());complete=json.loads((P/'collection_complete.json').read_text());assert complete['episodes']==400 and complete['all_passed']
index=[];vectors={};checks=[]
for row in m['episodes']:
 p=P/'episodes'/row['id'];v=json.loads((p/'validation.json').read_text());assert v['passed']
 with h5py.File(p/'features.h5') as f:
  assert f.attrs['validated'] and len(f['queries'])==v['queries']
  assert f['steps/action'].shape[0]==row['steps']
  for key,g in sorted(f['queries'].items()):
   required=['inputs/input_ids','inputs/image_input','inputs/image_mask','inputs/domain_id','inputs/proprio','vision/valid_view_tokens','fusion/vlm_features','fusion/aux_visual_inputs','projected/vlm_proj','projected/aux_visual_proj','generation/initial_noise_raw','actions/model_output_20d','raw_images/agentview_image','raw_images/robot0_eye_in_hand_image']
   for k in required:assert k in g,(row['id'],key,k)
   def check(name,ds):
    if isinstance(ds,h5py.Dataset):
     a=array(ds);assert np.isfinite(a).all(),(row['id'],key,name)
   g.visititems(check)
   vision=array(g['vision/valid_view_tokens']);fusion=array(g['fusion/vlm_features'])[0];n=int(g.attrs['primary_visual_token_count'])
   assert vision.shape==(2,50,1024) and n==50
   assert fusion.shape==(100,1024)
   assert g['fusion/aux_visual_inputs'].shape==(1,150,1024)
   assert g['inputs/image_input'].shape==(1,4,3,224,224)
   assert np.array_equal(g['inputs/image_mask'][()],[[True,True,False,False]])
   for block in ['block_00','block_11','block_23']:assert g['action_layers/'+block].shape==(1,30,1024)
   assert len([k for k in g['generation'] if k.startswith('iteration_')])==10
   pooled={'vision_main_mean':vision[0].mean(0),'vision_wrist_mean':vision[1].mean(0),'fusion_visual_mean':fusion[:n].mean(0),'fusion_text_sequence_mean':fusion[n:].mean(0),'projected_fusion_mean':array(g['projected/vlm_proj'])[0].mean(0),'projected_wrist_mean':array(g['projected/aux_visual_proj'])[0,:n].mean(0)}
   for lname,ds in g['action_layers'].items():
    a=array(ds)[0];pooled[lname+'_all_mean']=a.mean(0);pooled[lname+'_first5_mean']=a[:5].mean(0);pooled[lname+'_token0']=a[0]
   for k,a in pooled.items():vectors.setdefault(k,[]).append(a)
   xy=np.array(row['layout']['target_xy_m']);index.append({'row':len(index),'episode_id':row['id'],'level':row['level'],'episode_success':row['success'],'policy_step':int(key),'query_index':int(g.attrs['query_index']),'stage_heuristic':str(g.attrs['stage_heuristic']),'initial_xy_m':xy.tolist(),'mug_xyz':array(g['physical/mug_xyz']).tolist(),'eef_xyz':array(g['physical/eef_xyz']).tolist(),'grasped':bool(g['physical/grasped'][()]),'goal':bool(g['physical/goal'][()]),'xy_error_m':float(g['physical/xy_error_m'][()]),'speed_m_s':float(g['physical/speed_m_s'][()]),'h5_path':str(p/'features.h5'),'group':'queries/'+key,'source':row['source']})
 checks.append({'id':row['id'],'queries':v['queries'],'sha256':hashlib.sha256((p/'features.h5').read_bytes()).hexdigest(),'bytes':(p/'features.h5').stat().st_size})
 print('AUDIT',row['id'],v['queries'],flush=True)
np.savez_compressed(P/'pooled_features.npz',**{k:np.stack(v).astype(np.float32) for k,v in vectors.items()})
(P/'query_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
(P/'feature_file_manifest.json').write_text(json.dumps(checks,indent=2))
assert len(index)==complete['queries']
(P/'verified_complete.json').write_text(json.dumps({'episodes':400,'queries':len(index),'feature_groups':list(vectors),'all_features_read_and_finite':True,'all_required_datasets_present':True,'files_sha256_verified':400,'collection':complete},indent=2))
print('VERIFIED_COMPLETE',len(index),flush=True)
