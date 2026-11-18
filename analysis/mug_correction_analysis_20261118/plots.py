import os
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent;plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
s=json.loads((P/'summary.json').read_text());events=json.loads((P/'events.json').read_text());q=json.loads((P/'query_metrics.json').read_text());pairs=json.loads((P/'paired_ungrasped_queries.json').read_text());regions=['5×5 内部','5→10 环','10→15 环','15→20 环'];colors=['#2c7fba','#1c9a86','#e69535','#9861b5']
def save(fig,n):fig.savefig(P/(n+'_zh.png'),dpi=180);fig.savefig(P/(n+'_zh.pdf'));plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained')
for ax,typ,title in zip(axs,['first_close','first_far_loss'],['首次闭合后的第一次推理','远盘抓持检测丢失后的第一次推理']):
 for j,(key,label,c) in enumerate([('planned_toward_n','预测终点更靠近杯子 >0.5cm','#377fbd'),('actual_toward_n','实际水平距离缩小 >0.5cm','#e49b34'),('new_grasp_in_block_n','本块内新出现持续抓持检测','#21966f')]):
  a=s[typ];n=np.array([v['eligible_first_post_queries'] for v in a]);v=np.array([v[key] for v in a]);x=np.arange(4)+(j-1)*.25;ax.bar(x,v/n*100,width=.24,label=label,color=c)
  for xx,num,den in zip(x,v,n):ax.text(xx,num/den*100+1,f'{num}/{den}',ha='center',fontsize=9)
 ax.set_xticks(range(4),regions);ax.set(ylim=(0,105),ylabel='比例（%）',title=title);ax.grid(axis='y',alpha=.15)
handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside lower center',ncol=3,fontsize=10)
fig.suptitle('模型有没有朝杯子调整？\n只统计查询时尚未检测到抓持且目标未成立的样本；实际距离评估止于新持续抓持开始或动作块结束',fontsize=14);save(fig,'correction_rates')
fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
a=s['first_far_loss']
for j,(key,label,c) in enumerate([('stable_grasp_by90_n','后续出现持续抓持检测','#20966e'),('final_successes','最终完成任务','#377fbd')]):
 n=np.array([v['events'] for v in a]);v=np.array([v[key] for v in a]);x=np.arange(4)+(j-.5)*.32;axs[0].bar(x,v/n*100,width=.31,label=label,color=c)
 for xx,num,den in zip(x,v,n):axs[0].text(xx,num/den*100+1,f'{num}/{den}',ha='center',fontsize=10)
axs[0].set_xticks(range(4),regions);axs[0].set(ylim=(0,105),ylabel='事件轨迹中的比例（%）',title='再次抓持不等于任务恢复');axs[0].legend(fontsize=10)
n=np.array([v['events'] for v in a]);op=np.array([v['recent_open_command_n'] for v in a]);axs[1].bar(regions,n-op,color='#a466b4',label='事件当步及前4步无张开指令');axs[1].bar(regions,op,bottom=n-op,color='#eaa251',label='事件当步及前4步有张开指令')
for i in range(4):axs[1].text(i,(n[i]-op[i])/2,str(n[i]-op[i]),ha='center',va='center');axs[1].text(i,n[i]-op[i]+op[i]/2,str(op[i]),ha='center',va='center')
axs[1].set(ylabel='轨迹数',title='先区分主动张开与其他接触丢失');axs[1].legend(fontsize=9)
fig.suptitle('远盘抓持丢失后的恢复结果\n每条轨迹仅取首次事件；再次抓持窗口为丢失后至首次后续推理+90步，包含推理前已恢复的情况',fontsize=13);save(fig,'loss_recovery')
fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
for rr,typ in enumerate(['first_close','first_far_loss']):
 for l in range(4):
  a=[v for v in pairs if v['type']==typ and v['level']==l];ax=axs[rr,l]
  for ok,c,m in [(False,'#cc4747','x'),(True,'#178967','o')]:
   b=[v for v in a if v['episode_success']==ok];ax.scatter([v['first_gap_cm'] for v in b],[v['second_gap_cm'] for v in b],s=20,c=c,marker=m,label='最终成功' if ok else '最终失败')
  limit=max([v[k] for v in a for k in ['first_gap_cm','second_gap_cm']]+[1])*1.08;ax.plot([0,limit],[0,limit],'--',c='#888',lw=1);ax.set(xlim=(0,limit),ylim=(0,limit),aspect='equal',title=f'{regions[l]}，n={len(a)}',xlabel='事件后第1次推理距离（厘米）')
  if l==0:ax.set_ylabel(('首次闭合后' if rr==0 else '远盘丢失后')+'\n第2次推理距离（厘米）')
  if a:ax.text(.03,.95,f'缩小>0.5cm：{sum(v["gap_reduction_cm"]>.5 for v in a)}/{len(a)}',transform=ax.transAxes,va='top',fontsize=9)
fig.suptitle('同一条轨迹：后续两次推理时的实际距离有没有缩小？\n仅纳入两次查询均未检测抓持且目标未成立的配对；虚线下方表示距离缩小，不同区域不是相同样本难度',fontsize=13);save(fig,'paired_distance')
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,typ,title in zip(axs,['first_close','first_far_loss'],['首次闭合后','远盘抓持丢失后']):
 vals=[[e['delay_steps'] for e in events if e['type']==typ and e['level']==l and e['has_post_query']] for l in range(4)];ax.boxplot(vals,tick_labels=regions,showfliers=False);ax.set(ylabel='事件发生后，到下次推理的等待步数',title=title,ylim=(-1,31));ax.grid(axis='y',alpha=.15)
 for l,v in enumerate(vals):ax.text(l+1,30,f'n={len(v)}\n中位数 {np.median(v):g}',ha='center',va='top',fontsize=9)
fig.suptitle('事件之后并非立即重新推理\n原实验每30步查询一次，20Hz下最多等待29步（1.45秒）；这段时间继续执行已生成动作');save(fig,'query_latency')
fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for rr,typ in enumerate(['first_close','first_far_loss']):
 for col,(key,name) in enumerate([('vision_wrist_mean','腕部视觉'),('fusion_visual_mean','融合视觉'),('block_23_all_mean','动作第24层')]):
  ax=axs[rr,col]
  for l,color in enumerate(colors):
   ids=[e['id'] for e in events if e['type']==typ and e['level']==l and all(any(v['id']==e['id'] and v['type']==typ and v['offset']==o for v in q) for o in [0,1,2])]
   vals=[[v['feature_cosine_change_from_pre'][key] for v in q if v['id'] in ids and v['type']==typ and v['offset']==o] for o in [0,1,2]]
   if ids:ax.plot([1,2,3],[np.median(v) for v in vals],marker='o',color=color,label=regions[l]+f'（n={len(ids)}）')
  ax.set(title=('闭合后：' if rr==0 else '丢失后：')+name,xlabel='事件后第几次推理',ylabel='相对事件前最后推理的余弦距离');ax.set_xticks([1,2,3]);ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.suptitle('内部特征是否更新：同一固定样本队列的变化\n特征变化不是纠偏成功证据；动作层还受每次查询不同的生成噪声影响',fontsize=13);save(fig,'feature_evolution')
print('PLOTS_DONE')
