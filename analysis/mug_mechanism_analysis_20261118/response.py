import os
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent;r=json.loads((P/'episodes.json').read_text());rng=np.random.default_rng(20261118)
def regression(x,z):
 A=np.c_[np.ones(len(x)),x];b=np.linalg.lstsq(A,z,rcond=None)[0];return b[1:].T,b[0]
summary=[];responses=[]
for level in range(4):
 a=[v for v in r if v['level']==level];c=[v for v in a if v['first_close_step'] is not None];e=np.array([v['first_close_xy_error_cm'] for v in c]);z=np.array([v['first_close_z_offset_cm'] for v in c]);closexy=np.array([v['first_close_eef_xy_cm'] for v in c]);target=np.array([v['first_close_mug_xy_cm'] for v in c])
 summary.append({'level':level,'n':len(a),'n_first_close':len(c),'median_first_close_xy_gap_cm':float(np.median(e)),'q25_q75_first_close_xy_gap_cm':np.quantile(e,[.25,.75]).tolist(),'median_first_close_z_offset_cm':float(np.median(z)),'median_final_plate_error_cm':float(np.median([v['final_xy_error_cm'] for v in a])),'no_close_ids':[v['id'] for v in a if v['first_close_step'] is None],'success_close_gap_median_cm':float(np.median([v['first_close_xy_error_cm'] for v in c if v['success']])) if any(v['success'] for v in c) else None})
 for kind in ['first_chunk_endpoint_xy_cm','first_close_eef_xy_cm','first_close_command_xy_cm']:
  aa=[v for v in a if v[kind] is not None];x=np.array([v['initial_xy_cm'] for v in aa]);z=np.array([v[kind] for v in aa]);B,b=regression(x,z);boot=[]
  for _ in range(1000):
   ix=rng.integers(len(x),size=len(x));bb,_=regression(x[ix],z[ix]);boot.append(np.diag(bb))
  responses.append({'level':level,'response':kind,'n':len(aa),'gain_matrix_outputs_xy_inputs_xy':B.tolist(),'intercept_cm':b.tolist(),'diagonal_gain_bootstrap95':np.quantile(boot,[.025,.975],axis=0).tolist(),'note':'Descriptive affine gain, not a required ideal controller gain; first chunk is approach, not a completed grasp.'})
(P/'action_summary.json').write_text(json.dumps(summary,indent=2));(P/'action_response_regressions.json').write_text(json.dumps(responses,indent=2));print(json.dumps(summary,indent=2))
