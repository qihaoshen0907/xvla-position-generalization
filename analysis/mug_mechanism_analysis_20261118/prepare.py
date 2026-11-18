import os
from pathlib import Path
import json,hashlib
import numpy as np,h5py
P=Path(__file__).resolve().parent;S=Path(os.environ.get('XVLA_DATA_DIR', str(P.parents[1]/'data'/'mug_features_replay_20261118')));m=json.loads((S/'manifest.json').read_text())
def segments(mask):
 z=np.r_[False,mask,False].astype(int);return [(int(a),int(b)) for a,b in zip(np.where(np.diff(z)==1)[0],np.where(np.diff(z)==-1)[0])]
rows=[];arrays={};index=[]
for r in m['episodes']:
 with h5py.File(S/'episodes'/r['id']/'features.h5') as f:
  g=np.array(f['steps/grasped'],bool);goal=np.array(f['steps/goal'],bool);xyz=f['steps/mug_xyz'][()];eef=f['steps/eef_xyz'][()];a=f['steps/action'][()];err=f['steps/xy_error_m'][()];speed=f['steps/speed_m_s'][()];init=f['queries/0000/physical/mug_xyz'][()];init_eef=f['queries/0000/physical/eef_xyz'][()];plate=f['queries/0000/physical/plate_xyz'][()]
  sustained=segments(g);stable=[(b,e) for b,e in sustained if e-b>=5];lift=segments(g&(xyz[:,2]>init[2]+.04));lifted=any(e-b>=5 for b,e in lift)
  near=goal|(err<=.03);nearheld=bool(np.any(near&g));evergrasp=bool(g.any());longest=max([e-b for b,e in sustained],default=0)
  # Descriptive mutually exclusive furthest-observed-state partition, not a causal failure label.
  if r['success']:stage='success'
  elif not evergrasp:stage='no_grasp'
  elif not stable:stage='brief_grasp'
  elif np.any(goal) or nearheld:stage='near_plate_unfinished'
  elif not g[-1]:stage='lost_grasp_before_near_plate'
  else:stage='held_without_near_plate'
  close=np.flatnonzero((a[:,6]>0)&np.r_[True,a[:-1,6]<=0]);first=int(close[0]) if len(close) else None
  q0=f['queries/0000/actions/original_control'][()];pred_end=q0[-1,:3]
  prexyz=np.vstack([init,xyz[:-1]]);preef=np.vstack([init_eef,eef[:-1]])
  row={'id':r['id'],'level':r['level'],'success':r['success'],'source':r['source'],'steps':len(g),'initial_xy_cm':((init[:2]-[-.1,-.15])*100).tolist(),'stage':stage,'ever_grasped':evergrasp,'sustained_grasp_5steps':bool(stable),'lifted_grasp_4cm_5steps':lifted,'longest_grasp_steps':longest,'native_goal_ever':bool(goal.any()),'ever_near_plate_while_grasped':nearheld,'first_grasp_step':int(np.flatnonzero(g)[0]+1) if evergrasp else None,'first_close_step':first+1 if first is not None else None,'first_close_eef_xy_cm':(preef[first,:2]*100).tolist() if first is not None else None,'first_close_mug_xy_cm':(prexyz[first,:2]*100).tolist() if first is not None else None,'first_close_xy_error_cm':float(np.linalg.norm(preef[first,:2]-prexyz[first,:2])*100) if first is not None else None,'first_close_z_offset_cm':float((preef[first,2]-prexyz[first,2])*100) if first is not None else None,'first_close_command_xy_cm':(a[first,:2]*100).tolist() if first is not None else None,'first_chunk_endpoint_xy_cm':((pred_end[:2]-[-.1,-.15])*100).tolist(),'first_chunk_endpoint_z_cm':float(pred_end[2]*100),'first_chunk_closes':bool(np.any(q0[:,6]>0)),'final_xy_error_cm':float(err[-1]*100),'grasp_segments_1based_inclusive':[[b+1,e] for b,e in sustained],'close_steps_1based':(close+1).tolist()}
  row['object_lift_4cm_5steps_independent_of_grasp']=any(e-b>=5 for b,e in segments(xyz[:,2]>init[2]+.04));row['max_object_height_increase_cm']=float((xyz[:,2].max()-init[2])*100)
  rows.append(row);arrays[r['id']]=dict(grasp=g,xyz=xyz,eef=eef,a=a,goal=goal,err=err)
(P/'episodes.json').write_text(json.dumps(rows,indent=2))
# Predeclare spatial partitions, independent of outcomes or probe results.
xy=np.array([r['initial_xy_cm'] for r in rows]);tiles=np.floor((xy+10)/2.5).astype(int);groups=tiles[:,0]*8+tiles[:,1]
(P/'spatial_groups.json').write_text(json.dumps({'tile_width_cm':2.5,'test_buffer_cm':.5,'ids':[r['id'] for r in rows],'xy_cm':xy.tolist(),'tiles':tiles.tolist(),'groups':groups.tolist()},indent=2))
summary={l:{k:sum(r['level']==l and r['stage']==k for r in rows) for k in ['success','no_grasp','brief_grasp','lost_grasp_before_near_plate','held_without_near_plate','near_plate_unfinished']} for l in range(4)}
(P/'stage_counts.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
# Sensitivity to sustained grasp threshold, avoiding treating 4cm lift as mandatory success.
sens={}
for threshold in [1,5,10]:
 sens[threshold]={l:sum(r['level']==l and not r['success'] and any(e-b>=threshold for b,e in segments(arrays[r['id']]['grasp'])) for r in rows) for l in range(4)}
(P/'grasp_threshold_sensitivity.json').write_text(json.dumps(sens,indent=2))

(P/'grasp_detection_caveat.json').write_text(json.dumps({l:{'no_grasp_detection':sum(r['level']==l and r['stage']=='no_grasp' for r in rows),'no_grasp_detection_but_object_lift':sum(r['level']==l and r['stage']=='no_grasp' and r['object_lift_4cm_5steps_independent_of_grasp'] for r in rows)} for l in range(4)},indent=2))
