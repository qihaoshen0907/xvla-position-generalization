import os,sys
os.environ['MPLCONFIGDIR']='/tmp/v2-analysis-mpl';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
P=Path(sys.argv[1]);R=Path(os.environ.get('XVLA_ROOT', './xvla_local_test'))
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
r=json.loads((P/'episodes.json').read_text());y=np.array([x['v2_success'] for x in r]);b=np.array([x['baseline_success'] for x in r]);lev=np.array([x['level'] for x in r]);xy=np.array([[x['x_m'],x['y_m']] for x in r]);xycm=(xy-[-.1,-.15])*100;d=np.array([x['distance_cm'] for x in r]);weights=np.array([x['area_weight_m2'] for x in r]);names=['中心 5×5','外环 5→10','外环 10→15','外环 15→20'];C=['#377eb8','#e67e22'];green='#198b74';red='#cf505b';figures=[]
def save(fig,n):
 for ext in ['png','svg','pdf']:fig.savefig(P/(n+'.'+ext),dpi=175,bbox_inches='tight')
 figures.append(n);plt.close(fig);print('PLOT',n,flush=True)
def wilson(k,n):
 p=k/n;z=1.96;cen=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return (cen-h)*100,(cen+h)*100
def spatial(ax):
 ax.set(xlabel='相对采样中心 X（cm）',ylabel='相对采样中心 Y（cm）',xlim=(-10.6,10.6),ylim=(-10.6,10.6),aspect='equal');ax.grid(alpha=.15)
 for width in [5,10,15,20]:ax.add_patch(plt.Rectangle((-width/2,-width/2),width,width,fill=False,lw=.65,ec='#888',alpha=.6))
 ax.plot(0,0,'k+',ms=9)
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for j,(vals,name,c) in enumerate([(b,'原始模型',C[0]),(y,'v2 LoRA',C[1])]):
 ks=np.array([vals[lev==i].sum() for i in range(4)]);ci=np.array([wilson(k,100) for k in ks]);axs[0].bar(np.arange(4)+(j-.5)*.35,ks,width=.35,color=c,label=name,yerr=np.maximum(0,np.array([ks-ci[:,0],ci[:,1]-ks])),capsize=3)
 for i,k in enumerate(ks):axs[0].text(i+(j-.5)*.35,ci[i,1]+3,str(k)+'%',ha='center',fontsize=10)
axs[0].set(xticks=range(4),xticklabels=names,ylabel='稳定释放成功率（%）',ylim=(0,106),title='同一批坐标：分区域对比（每区 100 点）');axs[0].legend()
counts=[int(((b==1)&(y==1)).sum()),int(((b==0)&(y==1)).sum()),int(((b==1)&(y==0)).sum()),int(((b==0)&(y==0)).sum())]
axs[1].barh(['持续成功','失败→成功','成功→失败','持续失败'],counts,color=[green,'#67b9a4',red,'#aaa']);axs[1].invert_yaxis()
for i,k in enumerate(counts):axs[1].text(k+2,i,str(k),va='center')
axs[1].set(xlim=(0,max(counts)+30),xlabel='采样点数量',title='逐点变化：改善 90 点，退步 59 点')
fig.suptitle('400 点重测：33.25% → 41.00%，增加 7.75 个百分点\n区间为未加权 Wilson 95% 描述区间；固定单场景、单种子',fontsize=14);save(fig,'01_success_comparison')
fig,axs=plt.subplots(1,3,figsize=(17,5.6),layout='constrained')
for ax,v,title in [(axs[0],b,'原始模型：133/400'),(axs[1],y,'v2 LoRA：164/400')]:
 for z,col,mark,lab in [(0,red,'x','失败'),(1,green,'o','成功')]:
  mask=v==z;ax.scatter(*xycm[mask].T,c=col,marker=mark,s=19,label=lab,alpha=.85)
 spatial(ax);ax.set_title(title);ax.legend(loc='upper left',fontsize=9)
for mask,col,mark,lab in [((b==y),'#bbb','.', '结果不变'),((b==0)&(y==1),green,'o','改善 90'),((b==1)&(y==0),red,'x','退步 59')]:axs[2].scatter(*xycm[mask].T,c=col,marker=mark,s=25,label=lab)
spatial(axs[2]);axs[2].set_title('改善和退步发生在哪里');axs[2].legend(fontsize=9)
fig.suptitle('400 个实际采样位置：不插值，不用平滑掩盖局部差异');save(fig,'02_spatial_outcomes')
edges=np.arange(-10,10.1,2);bx=np.clip(np.digitize(xycm[:,0],edges)-1,0,9);by=np.clip(np.digitize(xycm[:,1],edges)-1,0,9);maps=[];ns=np.zeros((10,10),int)
for v in [b,y,y-b]:
 z=np.full((10,10),np.nan)
 for i in range(10):
  for j in range(10):
   mask=(bx==i)&(by==j);ns[j,i]=mask.sum()
   if mask.any():z[j,i]=np.average(v[mask],weights=weights[mask])*100
 maps.append(z)
fig,axs=plt.subplots(1,3,figsize=(17,5.7),layout='constrained')
for ax,z,title,delta in zip(axs,maps,['原始模型','v2 LoRA','v2 − 原始（百分点）'],[False,False,True]):
 im=ax.imshow(z,origin='lower',extent=[-10,10,-10,10],vmin=-100 if delta else 0,vmax=100,cmap='RdBu' if delta else 'YlGnBu');spatial(ax);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75)
 for j in range(10):
  for i in range(10):
   if ns[j,i]:ax.text(-9+i*2,-9+j*2,str(ns[j,i]),ha='center',va='center',fontsize=7,color='#fff' if (abs(z[j,i])>60 if delta else z[j,i]>60) else '#222')
fig.suptitle('2×2 cm 空间网格：按原采样区域有效面积加权\n格内数字是样本量 n；空白为无样本，小样本格的不确定性较大');save(fig,'03_spatial_heatmaps')
radial=[];fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for v,label,c in [(b,'原始模型',C[0]),(y,'v2 LoRA',C[1])]:
 xx=[];raw=[];weighted=[];ci=[]
 for lo in np.arange(0,18,2):
  m=(d>=lo)&(d<lo+2);n=m.sum()
  if not n:continue
  k=v[m].sum();xx.append(lo+1);raw.append(k/n*100);weighted.append(np.average(v[m],weights=weights[m])*100);ci.append(wilson(k,n));radial.append(dict(model=label,lo_cm=float(lo),hi_cm=float(lo+2),n=int(n),successes=int(k),raw_percent=float(raw[-1]),area_weighted_percent=float(weighted[-1])))
 raw=np.array(raw);ci=np.array(ci);axs[0].errorbar(xx,raw,yerr=np.maximum(0,[raw-ci[:,0],ci[:,1]-raw]),color=c,marker='o',capsize=3,label=label);axs[1].plot(xx,weighted,'o-',c=c,label=label)
for ax,title in zip(axs,['样本成功率及 Wilson 95% 描述区间','面积加权成功率（保留原采样权重）']):ax.set(title=title,xlabel='距官方示范白杯初始位置均值（cm）',ylabel='稳定释放成功率（%）',ylim=(-3,103));ax.grid(alpha=.2);ax.legend()
fig.suptitle('距离—成功率：互不重叠 2 cm 环带；圆心沿用首次分析\n官方示范均值不等于已确认的该 checkpoint 训练集均值');save(fig,'04_distance_curves');(P/'radial_bins.json').write_text(json.dumps(radial,indent=2))
fig,axs=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
for ax,width in zip(axs,[1,2,3]):
 for v,label,c in [(b,'原始',C[0]),(y,'v2',C[1])]:
  vals=[];cent=[]
  for lo in np.arange(0,18,width):
   m=(d>=lo)&(d<lo+width)
   if m.any():cent.append(lo+width/2);vals.append(np.average(v[m],weights=weights[m])*100)
  ax.plot(cent,vals,'o-',c=c,label=label)
 ax.set(title=f'环带宽度 {width} cm',ylim=(-3,103),xlabel='距离（cm）',ylabel='面积加权成功率（%）');ax.grid(alpha=.2);ax.legend()
fig.suptitle('分组敏感性：不同环带宽度下的曲线\n不强制单调，不用分组选择证明“距离越远必然越差”');save(fig,'05_distance_sensitivity')
stages=['success','no_grasp','brief_grasp','lost_grasp_before_near_plate','held_without_near_plate','near_plate_unfinished'];sn=['成功','未检测到抓持','抓持不足 5 步','持续抓持后丢失，未到盘边','仍抓持但未到盘边','到盘边/目标成立，未稳定完成'];sc=[green,red,'#ec9a55','#e8c34a','#8680b6','#539dbd']
fig,axs=plt.subplots(1,2,figsize=(15,6.6),layout='constrained');stcounts={}
for ax,key,title in zip(axs,['baseline_stage','stage'],['原始模型','v2 LoRA']):
 bottom=np.zeros(4);stcounts[key]={}
 for st,n,c in zip(stages,sn,sc):
  val=np.array([sum(x['level']==l and x[key]==st for x in r) for l in range(4)]);stcounts[key][st]=val.tolist();ax.bar(names,val,bottom=bottom,label=n,color=c)
  for i,k in enumerate(val):
   if k>=5:ax.text(i,bottom[i]+k/2,str(k),ha='center',va='center',fontsize=10)
  bottom+=val
 ax.set(title=title,ylabel='轨迹数',ylim=(0,103));ax.tick_params(axis='x',rotation=15)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
fig.suptitle('结果与失败阶段：沿用首次分析的互斥分类\n这是观测状态分类，未检测到抓持不等于物体未被带起');save(fig,'06_failure_stages');(P/'stage_counts.json').write_text(json.dumps(stcounts,indent=2))
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
for ax,key,basekey,title in [(axs[0],'steps','baseline_steps','执行步数（失败回合含 520 步截断）'),(axs[1],'final_error_cm','baseline_final_error_cm','最终杯子—目标盘水平距离'),(axs[2],'first_close_error_cm','baseline_first_close_error_cm','首次闭合前末端—杯子水平距离')]:
 for j,(k,c) in enumerate([(basekey,C[0]),(key,C[1])]):
  a=[[x[k] for x in r if x['level']==l and x[k] is not None] for l in range(4)];bp=ax.boxplot(a,positions=np.arange(4)+(j-.5)*.32,widths=.27,patch_artist=True,showfliers=False)
  for patch in bp['boxes']:patch.set_facecolor(c);patch.set_alpha(.7)
 ax.set_xticks(range(4),names,rotation=20);ax.set_title(title,fontsize=11);ax.set_ylabel('步' if key=='steps' else 'cm');ax.grid(axis='y',alpha=.15)
fig.suptitle('轨迹与动作诊断：蓝色原始模型，橙色 v2\n箱线图隐藏离群点仅为便于阅读；末端参考点距离不是指尖接触误差');save(fig,'07_trajectory_diagnostics')
fig,axs=plt.subplots(2,4,figsize=(16,11),layout='constrained')
for l in range(4):
 rr=[x for x in r if x['level']==l];points=np.array([[x['x_m'],x['y_m']] for x in rr])*100-[-10,-15]
 for ax,key,title in [(axs[0,l],'first_endpoint_xy_cm','首动作块终点'),(axs[1,l],'first_close_eef_xy_cm','首次闭合前末端')]:
  valid=[x for x in rr if x[key] is not None];z=np.array([x[key] for x in valid])-[-10,-15];ax.scatter(*points.T,s=12,c='#bbb',label='杯子初始位置');ax.scatter(*z.T,s=15,c=C[1],marker='x',label=title);spatial(ax);ax.set(xlim=(-12,12),ylim=(-18,18));ax.set_title(names[l]+' / '+title,fontsize=10)
axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8);fig.suptitle('v2 的早期动作空间分布\n首块输出不等于最终抓取点；统一坐标轴覆盖全部动作点，灰框标示采样范围');save(fig,'08_action_spatial_distribution')
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
steps=np.array([x['steps'] for x in r]);bs=np.array([x['baseline_steps'] for x in r]);xx=np.arange(0,521)
for v,s,label,c in [(b,bs,'原始',C[0]),(y,steps,'v2',C[1])]:axs[0].plot(xx,[(v.astype(bool)&(s<=t)).sum()/400*100 for t in xx],c=c,label=label)
axs[0].set(xlabel='仿真控制步数',ylabel='400 点中累计稳定成功比例（%）',title='在给定步数预算内完成的比例');axs[0].legend();axs[0].grid(alpha=.2)
q=json.loads((P/'queries.json').read_text());qt=[x['seconds'] for x in q if not x['pilot']];axs[1].hist(qt,bins=35,color=C[1],alpha=.85);axs[1].set(xlabel='秒 / 模型查询及特征落盘',ylabel='查询数',title=f'采集延迟（不含四个双跑校验回合）\n中位数 {np.median(qt):.3f} 秒');fig.suptitle('预算效率与本次采集开销：延迟包含特征记录，不是纯模型推理基准');save(fig,'09_completion_and_latency')
fig,ax=plt.subplots(figsize=(7,5),layout='constrained');mat=np.array([[sum((b==i)&(y==j)) for j in [0,1]] for i in [0,1]]);im=ax.imshow(mat,cmap='Blues')
for i in range(2):
 for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=25,color='white' if mat[i,j]>120 else '#222')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['v2 失败','v2 成功'],yticklabels=['原始失败','原始成功'],title='逐点配对结果：同样的 400 个初始位置');save(fig,'10_paired_outcomes')
summary={'n':400,'baseline_successes':int(b.sum()),'v2_successes':int(y.sum()),'delta_percentage_points':float((y-b).mean()*100),'area_weighted_baseline_percent':float(np.average(b,weights=weights)*100),'area_weighted_v2_percent':float(np.average(y,weights=weights)*100),'figures':figures}
rng=np.random.default_rng(20261119);boot=[]
for _ in range(4000):
 ii=np.concatenate([rng.choice(np.where(lev==j)[0],100,replace=True) for j in range(4)]);boot.append(np.mean((y-b)[ii])*100)
summary['paired_stratified_bootstrap95_delta_pp']=np.percentile(boot,[2.5,97.5]).tolist();(P/'summary.json').write_text(json.dumps(summary,indent=2))
print('BASIC_PLOTS_COMPLETE',flush=True)
