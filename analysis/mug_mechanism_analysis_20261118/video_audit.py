import os
from pathlib import Path
import json,av,numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent;S=Path(os.environ.get('XVLA_DATA_DIR', str(P.parents[1]/'data'/'mug_features_replay_20261118')));rows=json.loads((P/'episodes.json').read_text());plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
stages=['no_grasp','brief_grasp','lost_grasp_before_near_plate','held_without_near_plate','near_plate_unfinished'];names=['未检测到抓持','短暂抓持','持续抓持后丢失','末尾仍抓持','到盘边但未稳定完成'];picked=[];fig,axs=plt.subplots(5,3,figsize=(12,18),layout='constrained')
for i,(st,name) in enumerate(zip(stages,names)):
 r=next(r for r in rows if r['stage']==st);picked.append(r['id'])
 with h5py.File(S/'episodes'/r['id']/'features.h5') as f:xyz=f['steps/mug_xyz'][()]
 steps=[max(0,r['first_close_step']-1),int(np.argmax(xyz[:,2])),r['steps']-1];images={}
 with av.open(str(Path(r['source'])/'rollout.mp4')) as v:
  for k,fr in enumerate(v.decode(video=0)):
   if k in steps:images[k]=fr.to_ndarray(format='rgb24')[60:420,:360]
 for j,k in enumerate(steps):axs[i,j].imshow(images[k]);axs[i,j].set_title(f'{r["id"]}｜{name}\n'+['首次闭合指令','杯子最高时刻','轨迹末尾'][j]+f'：第{k+1}步');axs[i,j].axis('off')
fig.suptitle('失败阶段的代表性视频帧抽查（每类按编号取首条）\n类别由完整状态序列判定，画面用于检查明显矛盾，不代替全量人工标注',fontsize=14);fig.savefig(P/'stage_video_audit_zh.png',dpi=150);plt.close(fig)
(P/'video_audit_selection.json').write_text(json.dumps({'selection':'first episode by id in each failure category','ids':picked,'frames':'first close, highest mug body position, final','scope':'representative visual cross-check, not exhaustive manual ground truth'},indent=2));print(picked)
