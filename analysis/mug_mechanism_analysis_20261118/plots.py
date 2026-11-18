import os
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent;plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
rows=json.loads((P/'episodes.json').read_text());counts=json.loads((P/'stage_counts.json').read_text());pr=json.loads((P/'probe_results.json').read_text());ar=json.loads((P/'action_response_regressions.json').read_text());sm=json.loads((P/'action_summary.json').read_text())
regions=['5×5 内部','5→10 新增环','10→15 新增环','15→20 新增环'];cols=['#2781b5','#21a48b','#df9b36','#9962b9']
def save(fig,n):fig.savefig(P/(n+'_zh.png'),dpi=180);fig.savefig(P/(n+'_zh.pdf'));plt.close(fig)
stages=['success','no_grasp','brief_grasp','lost_grasp_before_near_plate','held_without_near_plate','near_plate_unfinished'];names=['成功','从未检测到抓持','抓持检测仅短暂为真（不足 5 步）','持续抓持检测后转假，未到盘边','结束时抓持检测为真，未到盘边','曾到盘边或目标成立，未稳定完成'];colors=['#168567','#bf373d','#e38548','#e9b944','#6e70ab','#5396b9']
fig,ax=plt.subplots(figsize=(11,7),layout='constrained');bottom=np.zeros(4)
for key,n,c in zip(stages,names,colors):
 v=np.array([counts[str(l)][key] for l in range(4)]);ax.bar(regions,v,bottom=bottom,color=c,label=n)
 for i,num in enumerate(v):
  if num>=5:ax.text(i,bottom[i]+num/2,str(num),ha='center',va='center',color='white' if key in ['success','no_grasp','held_without_near_plate'] else '#222')
 bottom+=v
ax.set(ylabel='轨迹数（各区域 100 条）',ylim=(0,104),title='任务结果与失败阶段分解\n按状态检测分解；未检测到抓持不等于杯子未被带起')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2,fontsize=10);save(fig,'failure_stages')
keys=['constant','vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean'];titles=['训练集均值基线','主视角视觉','腕部视角视觉','融合视觉','动作第 1 层','动作第 12 层','动作第 24 层']
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
for ax,proto,title,levels in [(axs[0],'spatial','覆盖各区域训练 → 留出空间块测试',range(4)),(axs[1],'extrapolation','仅默认 5×5 训练 → 外圈测试',range(1,4))]:
 a=np.array([[next(v['mean_euclidean_error_cm'] for v in pr if v['feature']==k and v['protocol']==proto and v['level']==l) for l in levels] for k in keys]);im=ax.imshow(a,cmap='YlOrRd',vmin=0,vmax=10,aspect='auto')
 for i in range(len(keys)):
  for j in range(len(list(levels))):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color='white' if a[i,j]>6 else '#222')
 ax.set_xticks(range(len(list(levels))),[regions[l] for l in levels],rotation=15);ax.set_yticks(range(len(keys)),titles);ax.set_title(title);fig.colorbar(im,ax=ax,label='平均二维位置误差（厘米）',shrink=.8)
fig.suptitle('位置解码探针：岭回归；目标为首次推理时的杯子 XY\n空间块边长 2.5 厘米；左图测试点周围 0.5 厘米不用于训练；探针训练不改变 XVLA',fontsize=13);save(fig,'position_probe')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,proto,levels in [(axs[0],'spatial',range(4)),(axs[1],'extrapolation',range(1,4))]:
 for k,n,c in [('constant','均值基线','#888'),('vision_wrist_mean','腕部视觉','#239a94'),('fusion_visual_mean','融合视觉','#377fbd'),('block_23_all_mean','动作第 24 层','#b66b2e')]:
  v=[next(z for z in pr if z['feature']==k and z['protocol']==proto and z['level']==l) for l in levels];mean=np.array([z['mean_euclidean_error_cm'] for z in v]);ci=np.array([z['mean_error_tile_bootstrap95'] for z in v]);ax.errorbar(list(levels),mean,yerr=[np.maximum(0,mean-ci[:,0]),np.maximum(0,ci[:,1]-mean)],label=n,color=c,marker='o',capsize=3)
 ax.set_xticks(list(levels),[regions[l] for l in levels],rotation=15);ax.set_ylabel('平均二维位置误差（厘米）');ax.set_title('空间块留出测试' if proto=='spatial' else '默认区域训练，外圈测试');ax.grid(alpha=.2)
axs[0].legend();fig.suptitle('位置探针误差与 95% 区间\n按空间块重采样，仅反映当前划分下的抽样变化，不涵盖所有训练不确定性');save(fig,'position_probe_intervals')
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
vals=[np.array([r['first_close_xy_error_cm'] for r in rows if r['level']==l]) for l in range(4)]
axs[0].boxplot(vals,tick_labels=regions,showfliers=False);axs[0].tick_params(axis='x',rotation=20);axs[0].set(ylabel='闭合前末端—杯子水平距离（厘米）',title='首次闭合指令发出前的距离')
for i,a in enumerate(vals):axs[0].text(i+1,np.quantile(a,.75)+.5,f'中位数 {np.median(a):.2f}',ha='center',fontsize=9)
for ax,kind,title in [(axs[1],'first_chunk_endpoint_xy_cm','首个动作块终点：位置响应'),(axs[2],'first_close_eef_xy_cm','首次闭合前末端：位置响应')]:
 for j,n,c in [(0,'X 对 X 的响应','#377fbd'),(1,'Y 对 Y 的响应','#db8b26')]:
  vv=[next(v for v in ar if v['level']==l and v['response']==kind) for l in range(4)];gain=np.array([v['gain_matrix_outputs_xy_inputs_xy'][j][j] for v in vv]);ci=np.array([v['diagonal_gain_bootstrap95'] for v in vv]);ax.errorbar(range(4),gain,yerr=[gain-ci[:,0,j],ci[:,1,j]-gain],marker='o',color=c,label=n,capsize=3)
 ax.set_xticks(range(4),regions,rotation=20);ax.set(title=title,ylabel='仿射拟合的同轴斜率');ax.axhline(0,c='#888',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('位置到动作响应：全部 400 条均出现过闭合指令\n距离按末端参考点计算，不是指尖接触误差；首块仍可能在接近，斜率不要求等于 1',fontsize=13);save(fig,'action_response')
fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained')
for l in range(4):
 a=[r for r in rows if r['level']==l];xy=np.array([r['initial_xy_cm'] for r in a]);ep=np.array([r['first_chunk_endpoint_xy_cm'] for r in a]);cl=np.array([r['first_close_eef_xy_cm'] for r in a])-np.array([-10,-15]);axs[0,l].scatter(xy[:,0],xy[:,1],s=12,c=cols[l],label='杯子初始位置');axs[0,l].scatter(ep[:,0],ep[:,1],s=12,c='#222',marker='x',label='首块末端指令');axs[1,l].scatter(xy[:,0],xy[:,1],s=12,c=cols[l]);axs[1,l].scatter(cl[:,0],cl[:,1],s=12,c='#222',marker='x',label='首次闭合前末端')
 for ax in axs[:,l]:ax.set(xlim=(-11,11),ylim=(-11,11),aspect='equal',xlabel='相对区域中心 X（厘米）');ax.grid(alpha=.15)
 axs[0,l].set_title(regions[l])
axs[0,0].set_ylabel('首块动作：相对中心 Y（厘米）');axs[1,0].set_ylabel('首次闭合：相对中心 Y（厘米）');axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8);fig.suptitle('杯子采样范围扩大，初始动作与首次闭合末端分布仍较集中\n这是早期动作的描述性比较，不代表完整轨迹的可达范围');save(fig,'action_spatial_distribution')
print('PLOTS_DONE')
