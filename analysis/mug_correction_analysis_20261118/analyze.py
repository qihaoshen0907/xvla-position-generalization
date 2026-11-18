import os
from pathlib import Path
import json,numpy as np,h5py
P=Path(__file__).resolve().parent;S=Path(os.environ.get('XVLA_DATA_DIR', P.parent/'mug_features_replay_20261118'));manifest=json.loads((S/'manifest.json').read_text());index=[json.loads(l) for l in (S/'query_index.jsonl').read_text().splitlines()];lookup={(r['episode_id'],r['policy_step']):i for i,r in enumerate(index)};F=np.load(S/'pooled_features.npz');keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean']
F={k:F[k] for k in keys}
def seg(a):
 d=np.diff(np.r_[False,a,False].astype(int));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def cos(a,b):return float(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
events=[];queries=[];inventory=[]
for row in manifest['episodes']:
 id=row['id']
 with h5py.File(S/'episodes'/id/'features.h5') as f:
  st=f['steps'];g=st['grasped'][()].astype(bool);goal=st['goal'][()].astype(bool);mug=st['mug_xyz'][()];eef=st['eef_xyz'][()];act=st['action'][()];err=st['xy_error_m'][()];N=len(g);qsteps=sorted(int(k) for k in f['queries']);stable=seg(g);stable=[(int(a),int(b)) for a,b in stable if b-a>=5];stable_starts=[a+1 for a,b in stable]
  close=int(np.flatnonzero((act[:,6]>0)&np.r_[True,act[:-1,6]<=0])[0]+1)
  losses=[]
  for a,b in stable:
   if b+3>N or np.any(g[b:b+3]):continue
   step=b+1;near=bool(goal[b] or err[b]<=.03);openrecent=bool(np.any(act[max(0,b-4):b+1,6]<0))
   losses.append({'step':step,'near_plate':near,'open_command_recent':openrecent,'xy_plate_cm':float(err[b]*100)})
  far=[v for v in losses if not v['near_plate']];evs=[('first_close',close,None)]+([('first_far_loss',far[0]['step'],far[0])] if far else [])
  inventory.append({'id':id,'level':row['level'],'success':row['success'],'steps':N,'stable_grasp_runs':len(stable),'persistent_loss_events':losses,'has_far_loss':bool(far)})
  for etype,step,loss in evs:
   after=[q for q in qsteps if q>=step];before=[q for q in qsteps if q<step];q0=after[0] if after else None;qb=before[-1] if before else None
   event={'id':id,'type':etype,'level':row['level'],'episode_success':row['success'],'event_step':step,'next_query_step':q0,'last_query_before':qb,'delay_steps':q0-step if q0 is not None else None,'loss':loss,'has_post_query':q0 is not None,'observed_steps_after_event':N-step}
   if q0 is not None:
    futuregrasp=[z for z in stable_starts if z>step];event['next_stable_grasp_step']=futuregrasp[0] if futuregrasp else None
    # Event-specific 90-step follow-up; early stable task completion is an observed positive terminal outcome.
    event['window90_observed']=bool(N>=q0+90 or row['success'])
    event['stable_grasp_by_query_plus90_including_prequery_recovery']=bool(any(step<z<=min(N,q0+90) for z in stable_starts))
    event['grasp_restored_before_next_query']=bool(any(step<z<=q0 for z in stable_starts))
    event['goal_within90_after_query']=bool(np.any(goal[q0:min(N,q0+90)]))
    chosen=[(-1,qb)]+[(j,qsteps[qsteps.index(q0)+j]) for j in range(3) if qsteps.index(q0)+j<len(qsteps)]
    for offset,q in chosen:
     if q is None:continue
     Q=f[f'queries/{q:04d}'];phys=Q['physical'];mq=phys['mug_xyz'][()];eq=phys['eef_xyz'][()];gq=bool(phys['grasped'][()]);goalq=bool(phys['goal'][()]);pred=Q['actions/original_control'][()];end=min(q+len(pred),N);new=[s for s in stable_starts if q<s<=end];stop=new[0] if new else end
     gap=np.linalg.norm(eq[:2]-mq[:2])*100;predgap=np.linalg.norm(pred[-1,:2]-mq[:2])*100;actualgap=np.linalg.norm(eef[stop-1,:2]-mug[stop-1,:2])*100;fixedgap=np.linalg.norm(eef[stop-1,:2]-mq[:2])*100
     entry={'id':id,'type':etype,'level':row['level'],'episode_success':row['success'],'event_step':step,'offset':offset,'query_step':q,'query_grasped':gq,'query_goal':goalq,'eligible_approach':not gq and not goalq,'current_xy_gap_cm':float(gap),'predicted_endpoint_xyz_m':pred[-1,:3].tolist(),'predicted_endpoint_gap_cm':float(predgap),'predicted_approach_cm':float(gap-predgap),'planned_toward_gt_0_5cm':bool(gap-predgap>.5),'actual_gap_at_stop_cm':float(actualgap),'actual_approach_cm':float(gap-actualgap),'actual_fixed_target_approach_cm':float(gap-fixedgap),'actual_toward_gt_0_5cm':bool(gap-actualgap>.5),'block_end_step':end,'evaluated_stop_step':stop,'stopped_at_new_stable_grasp':bool(new),'cup_motion_to_stop_cm':float(np.linalg.norm(mug[stop-1,:2]-mq[:2])*100),'block_new_stable_grasp':bool(new),'full30_available':end-q==30,'actual_3d_gap_at_stop_cm':float(np.linalg.norm(eef[stop-1]-mug[stop-1])*100)}
     if qb is not None:
      old=f[f'queries/{qb:04d}/actions/original_control'][()];entry['plan_endpoint_shift_from_pre_cm']=float(np.linalg.norm(pred[-1,:2]-old[-1,:2])*100)
      entry['feature_cosine_change_from_pre']={k:cos(F[k][lookup[(id,qb)]].astype(float),F[k][lookup[(id,q)]].astype(float)) for k in keys}
     queries.append(entry)
   events.append(event)
print('events',len(events),'queries',len(queries),flush=True)
(P/'events.json').write_text(json.dumps(events,indent=2));(P/'query_metrics.json').write_text(json.dumps(queries,indent=2));(P/'event_inventory.json').write_text(json.dumps(inventory,indent=2))
# Paired same-episode correction from first post-event inference to next inference, only if still ungrasped.
paired=[]
for e in events:
 a=next((v for v in queries if v['id']==e['id'] and v['type']==e['type'] and v['offset']==0),None);b=next((v for v in queries if v['id']==e['id'] and v['type']==e['type'] and v['offset']==1),None)
 if a and b and a['eligible_approach'] and b['eligible_approach']:
  paired.append({'id':e['id'],'type':e['type'],'level':e['level'],'gap_reduction_cm':a['current_xy_gap_cm']-b['current_xy_gap_cm'],'first_gap_cm':a['current_xy_gap_cm'],'second_gap_cm':b['current_xy_gap_cm'],'first_plan_approach_cm':a['predicted_approach_cm'],'second_plan_approach_cm':b['predicted_approach_cm'],'episode_success':e['episode_success']})
(P/'paired_ungrasped_queries.json').write_text(json.dumps(paired,indent=2))
summary={}
for typ in ['first_close','first_far_loss']:
 out=[]
 for l in range(4):
  es=[e for e in events if e['type']==typ and e['level']==l];qs=[v for v in queries if v['type']==typ and v['level']==l and v['offset']==0 and v['eligible_approach']];obs=[e for e in es if e.get('window90_observed')];pa=[v for v in paired if v['type']==typ and v['level']==l]
  out.append({'level':l,'events':len(es),'post_query_available':sum(e['has_post_query'] for e in es),'median_delay_steps':float(np.median([e['delay_steps'] for e in es if e['has_post_query']])) if any(e['has_post_query'] for e in es) else None,'eligible_first_post_queries':len(qs),'planned_toward_n':sum(v['planned_toward_gt_0_5cm'] for v in qs),'actual_toward_n':sum(v['actual_toward_gt_0_5cm'] for v in qs),'new_grasp_in_block_n':sum(v['block_new_stable_grasp'] for v in qs),'median_plan_approach_cm':float(np.median([v['predicted_approach_cm'] for v in qs])) if qs else None,'median_actual_approach_cm':float(np.median([v['actual_approach_cm'] for v in qs])) if qs else None,'window90_n':len(obs),'stable_grasp_by90_n':sum(e['stable_grasp_by_query_plus90_including_prequery_recovery'] for e in obs),'restored_before_query_n':sum(e.get('grasp_restored_before_next_query',False) for e in es),'final_successes':sum(e['episode_success'] for e in es),'recent_open_command_n':sum(bool(e['loss'] and e['loss']['open_command_recent']) for e in es),'paired_ungrasped_n':len(pa),'paired_gap_reduced_gt_0_5cm_n':sum(v['gap_reduction_cm']>.5 for v in pa),'paired_median_gap_reduction_cm':float(np.median([v['gap_reduction_cm'] for v in pa])) if pa else None})
 summary[typ]=out
(P/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
