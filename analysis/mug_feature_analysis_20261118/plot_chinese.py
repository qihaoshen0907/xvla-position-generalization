import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
D=np.load(P/'sample_labels.npz');y=D['success'];rings=D['ring'];xy=(D['xy_m']-[-.1,-.15])*100
keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean']
names=['主视角视觉特征','腕部视角视觉特征','图文融合后的视觉特征','动作模块第 1 层','动作模块第 12 层','动作模块第 24 层']
colors=['#2580bb','#23a58b','#e89832','#9b63bd'];ringnames=['5×5 厘米内部','5→10 厘米新增环','10→15 厘米新增环','15→20 厘米新增环']
def points(ax,Z):
 for v,c,m,label in [(False,'#d44848','x','失败（267 条）'),(True,'#168567','o','成功（133 条）')]:ax.scatter(Z[y==v,0],Z[y==v,1],c=c,marker=m,s=17,linewidths=.8,alpha=.85,label=label)
 ax.set_xticks([]);ax.set_yticks([])
def save(fig,name):
 fig.savefig(P/(name+'_zh.png'),dpi=180);fig.savefig(P/(name+'_zh.pdf'));plt.close(fig)
for method,methodname,paramname in [('tsne','t-SNE','困惑度'),('umap','UMAP','邻居数')]:
 fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
 for ax,k,n in zip(axs.flat,keys,names):points(ax,np.load(P/f'{method}__{k}__p30__s0.npy'));ax.set_title(n,fontsize=14)
 hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=2,fontsize=12)
 fig.suptitle(f'{methodname}：400 条轨迹首次推理的内部特征\n{paramname}=30，随机种子=0；各面板独立降维，图中距离不是实际空间距离',fontsize=14)
 save(fig,f'{method}_outcome')
 fig,axs=plt.subplots(6,3,figsize=(12,20),layout='constrained')
 for i,(k,n) in enumerate(zip(keys,names)):
  Z=np.load(P/f'{method}__{k}__p30__s0.npy')
  for l,c in enumerate(colors):axs[i,0].scatter(Z[rings==l,0],Z[rings==l,1],c=c,s=13,label=ringnames[l])
  axs[i,0].set_ylabel(n)
  for col,coord in [(1,0),(2,1)]:
   sc=axs[i,col].scatter(Z[:,0],Z[:,1],c=xy[:,coord],cmap='coolwarm',vmin=-10,vmax=10,s=13);fig.colorbar(sc,ax=axs[i,col],shrink=.7,label='厘米')
  for ax in axs[i]:ax.set_xticks([]);ax.set_yticks([])
 for ax,t in zip(axs[0],['采样区域','相对区域中心的初始 X','相对区域中心的初始 Y']):ax.set_title(t)
 hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=2);fig.suptitle(f'{methodname} 位置对照：与成功／失败图使用相同降维坐标')
 save(fig,f'{method}_position_context')
 for k,n in [('fusion_visual_mean','融合视觉特征'),('block_23_all_mean','动作模块第 24 层')]:
  fig,axs=plt.subplots(3,3,figsize=(11,11),layout='constrained')
  for i,p in enumerate([15,30,50]):
   for j,s in enumerate([0,1,2]):points(axs[i,j],np.load(P/f'{method}__{k}__p{p}__s{s}.npy'));axs[i,j].set_title(f'{paramname}={p}，随机种子={s}')
  hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=2);fig.suptitle(f'{methodname} 稳定性检查：{n}\n每个面板单独拟合，比较局部邻域而非坐标方向')
  save(fig,f'{method}_robustness_{k}')
fig,axs=plt.subplots(2,4,figsize=(17,8),layout='constrained')
for i,(k,n) in enumerate([('fusion_visual_mean','融合视觉特征'),('block_23_all_mean','动作模块第 24 层')]):
 Z=np.load(P/f'tsne__{k}__p30__s0.npy');points(axs[i,0],Z);axs[i,0].set_ylabel(n)
 sc=axs[i,1].scatter(Z[:,0],Z[:,1],c=rings,cmap=matplotlib.colors.ListedColormap(colors),vmin=-.5,vmax=3.5,s=15)
 cb=fig.colorbar(sc,ax=axs[i,1],ticks=[0,1,2,3],shrink=.7);cb.ax.set_yticklabels(['5×5 内部','5→10 环','10→15 环','15→20 环'])
 for col,coord in [(2,0),(3,1)]:
  sc=axs[i,col].scatter(Z[:,0],Z[:,1],c=xy[:,coord],cmap='coolwarm',vmin=-10,vmax=10,s=15);fig.colorbar(sc,ax=axs[i,col],shrink=.7,label='厘米')
 for ax in axs[i]:ax.set_xticks([]);ax.set_yticks([])
for ax,t in zip(axs[0],['最终任务结果','采样区域','初始 X（相对中心）','初始 Y（相对中心）']):ax.set_title(t)
hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=2);fig.suptitle('t-SNE：同一降维坐标，分别按任务结果和位置着色\n困惑度=30，随机种子=0；位置与任务结果相关，不能据此认定失败原因',fontsize=14)
save(fig,'tsne_outcome_position_comparison')
fig,ax=plt.subplots(figsize=(7,7),layout='constrained');points(ax,xy);ax.set(xlabel='相对区域中心的初始 X（厘米）',ylabel='相对区域中心的初始 Y（厘米）',title='400 条轨迹的实际初始位置与任务结果',aspect='equal');ax.set_xticks([-10,-5,0,5,10]);ax.set_yticks([-10,-5,0,5,10]);ax.legend();save(fig,'physical_positions')
print('完成：10 张中文图及对应 PDF；使用已有降维坐标，未重新拟合。')
