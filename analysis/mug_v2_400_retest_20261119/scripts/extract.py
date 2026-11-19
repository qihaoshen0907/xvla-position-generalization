import os
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,csv,sys,hashlib
import numpy as np,h5py
R=Path(os.environ.get('XVLA_ROOT', './xvla_local_test')); S=R/'mug_v2_400_retest_20261119';P=Path(sys.argv[1]);P.mkdir(parents=True,exist_ok=True)
rows=sorted(json.loads((S/'results.json').read_text()),key=lambda r:r['layout']['id']);assert len(rows)==400
old={r['id']:r for r in csv.DictReader((R/'mug_400_distance_analysis_20261118/official_mug_center/episodes_with_distances.csv').open())}
base_mech={r['id']:r for r in json.loads((R/'mug_mechanism_analysis_20261118/episodes.json').read_text())}
keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean']; feats={k:[] for k in keys};records=[];query=[];trajectories={}
def arr(d):
 a=d[()]
 if d.attrs.get('torch_dtype')=='torch.bfloat16':return (a.astype(np.uint32)<<16).view(np.float32)
 return a
def segments(a):
 d=np.diff(np.r_[False,a,False].astype(int));return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
for ix,r in enumerate(rows):
 id=r['layout']['id'];b=old[id];dest=S/'episodes'/id
 assert max(abs(float(b[k])-r['layout']['target_xy_m'][i]) for i,k in enumerate(['x_m','y_m']))<1e-5
 with h5py.File(dest/'features.h5') as f:
  assert f.attrs['complete'];q=f['queries/0000'];vis=arr(q['vision/valid_view_tokens'])
  vv=[vis[0].mean(0),vis[1].mean(0),arr(q['fusion/vlm_features'])[0,:vis.shape[1]].mean(0)]+[arr(q[f'action_layers/block_{i:02d}'])[0].mean(0) for i in [0,11,23]]
  for k,x in zip(keys,vv):assert np.isfinite(x).all();feats[k].append(x)
  g=np.array(f['steps/grasped'],bool);goal=np.array(f['steps/goal'],bool);xyz=f['steps/mug_xyz'][()];eef=f['steps/eef_xyz'][()];a=f['steps/action'][()];err=f['steps/xy_error_m'][()];init=q['physical/mug_xyz'][()];initial_eef=q['physical/eef_xyz'][()]
  sustained=[(s,e) for s,e in segments(g) if e-s>=5];nearheld=np.any((goal|(err<=.03))&g)
  stage='success' if r['released_stable_success'] else 'no_grasp' if not g.any() else 'brief_grasp' if not sustained else 'near_plate_unfinished' if goal.any() or nearheld else 'lost_grasp_before_near_plate' if not g[-1] else 'held_without_near_plate'
  close=np.flatnonzero((a[:,6]>0)&np.r_[True,a[:-1,6]<=0]);t=int(close[0]) if len(close) else None
  prexyz=np.vstack([init,xyz[:-1]]);preef=np.vstack([initial_eef,eef[:-1]])
  rec={'id':id,'level':r['layout']['level'],'x_m':float(b['x_m']),'y_m':float(b['y_m']),'distance_cm':float(b['distance_cm']),'area_weight_m2':float(b['area_weight_m2']),'baseline_success':int(b['success']),'v2_success':int(r['released_stable_success']),'native_success':int(r['native_success']),'steps':r['steps'],'baseline_steps':base_mech[id]['steps'],'stage':stage,'baseline_stage':base_mech[id]['stage'],'final_error_cm':float(err[-1]*100),'baseline_final_error_cm':base_mech[id]['final_xy_error_cm'],'max_lift_cm':float((xyz[:,2].max()-init[2])*100),'ever_grasped':bool(g.any()),'first_close_step':t+1 if t is not None else None,'first_close_error_cm':float(np.linalg.norm(preef[t,:2]-prexyz[t,:2])*100) if t is not None else None,'baseline_first_close_error_cm':base_mech[id]['first_close_xy_error_cm'],'first_close_eef_xy_cm':(preef[t,:2]*100).tolist() if t is not None else None,'first_endpoint_xy_cm':(q['actions/predicted_control'][-1,:2]*100).tolist(),'initial_eef_xy_cm':(initial_eef[:2]*100).tolist(),'queries':len(f['queries']),'elapsed_seconds':r['elapsed_seconds']}
  records.append(rec);trajectories[id]=np.c_[xyz,eef,err,g,goal]
  for qq in r['queries']: query.append({'id':id,'level':rec['level'],'step':qq['step'],'seconds':qq['seconds'],'pilot':r['pilot']})
 if (ix+1)%25==0:print('EXTRACT',ix+1,flush=True)
(P/'episodes.json').write_text(json.dumps(records,indent=2));(P/'queries.json').write_text(json.dumps(query,indent=2));np.savez_compressed(P/'first_query_features.npz',**{k:np.array(v) for k,v in feats.items()});np.savez_compressed(P/'trajectories.npz',**trajectories)
with (P/'episodes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
source={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [S/'results.json',S/'manifest.json',R/'mug_400_distance_analysis_20261118/official_mug_center/episodes_with_distances.csv']}
(P/'source_hashes.json').write_text(json.dumps(source,indent=2));print('EXTRACTION_COMPLETE',flush=True)
