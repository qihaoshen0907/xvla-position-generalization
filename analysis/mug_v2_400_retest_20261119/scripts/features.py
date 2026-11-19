import os,sys
os.environ['MPLCONFIGDIR']='/tmp/v2-analysis-mpl';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
R=Path(os.environ.get('XVLA_ROOT', './xvla_local_test'));sys.path.append(str(R/'feature_analysis_deps'))
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE,trustworthiness
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
import umap
P=Path(sys.argv[1]);plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':10})
r=json.loads((P/'episodes.json').read_text());ids=[x['id'] for x in r];y=np.array([x['v2_success'] for x in r]);lev=np.array([x['level'] for x in r]);xy=np.array([[x['x_m'],x['y_m']] for x in r])*100;xyrel=xy-[-10,-15]
keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean'];titles=['主视角视觉','腕部视角视觉','融合视觉','动作第 1 层','动作第 12 层','动作第 24 层'];X=np.load(P/'first_query_features.npz');E={};diagnostics=[];colors=['#377eb8','#22a38d','#e59734','#9463b7']
def save(fig,n):
 for ext in ['png','svg','pdf']:fig.savefig(P/(n+'.'+ext),dpi=175,bbox_inches='tight')
 plt.close(fig);print('PLOT',n,flush=True)
def embedding(key,method,param=30,seed=0):
 name=f'{method}_{key}_p{param}_s{seed}';path=P/(name+'.npy');xx=normalize(X[key].astype(np.float64))
 if path.exists():z=np.load(path)
 elif method=='tsne':z=TSNE(perplexity=param,init='random',learning_rate='auto',max_iter=1500,random_state=seed,n_jobs=4).fit_transform(xx);np.save(path,z)
 else:z=umap.UMAP(n_neighbors=param,min_dist=.1,init='random',n_epochs=500,random_state=seed,n_jobs=1).fit_transform(xx);np.save(path,z)
 assert np.isfinite(z).all();return z,xx
for method in ['tsne','umap']:
 for key in keys:
  z,xx=embedding(key,method);E[method,key]=z;diagnostics.append({'method':method,'feature':key,'trustworthiness_k10':float(trustworthiness(xx,z,n_neighbors=10))});print('EMBED',method,key,flush=True)
 for coloring in ['outcome','region']:
  fig,axs=plt.subplots(2,3,figsize=(14,9),layout='constrained')
  for ax,key,title in zip(axs.flat,keys,titles):
   z=E[method,key]
   if coloring=='outcome':
    for label,col,marker,name in [(0,'#cf505b','x','失败 236'),(1,'#198b74','o','成功 164')]:
     m=y==label;ax.scatter(*z[m].T,s=16,c=col,marker=marker,alpha=.75,label=name)
   else:
    for j,col in enumerate(colors):
     m=lev==j;ax.scatter(*z[m].T,s=14,c=col,alpha=.8,label=f'区域 {j}')
   ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
  axs[0,0].legend(fontsize=9);fig.suptitle(f'v2 首次动作前特征 · {method.upper()} · '+('按最终结果着色' if coloring=='outcome' else '按采样区域着色')+'\n每点一条样本；逐向量 L2 归一化；参数 30 / seed 0；标签不参与降维，坐标无物理单位',fontsize=13)
  save(fig,('11' if method=='tsne' else '12')+'_'+method+'_'+coloring)
 # Last layer robustness, same predetermined grid as first analysis.
 fig,axs=plt.subplots(3,3,figsize=(11,10),layout='constrained')
 for i,param in enumerate([15,30,50]):
  for j,seed in enumerate([0,1,2]):
   z,_=embedding(keys[-1],method,param,seed);ax=axs[i,j]
   for label,col,marker in [(0,'#cf505b','x'),(1,'#198b74','o')]:
    m=y==label;ax.scatter(*z[m].T,s=12,c=col,marker=marker,alpha=.65)
   ax.set_title(f'参数 {param} / seed {seed}');ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle(f'动作第 24 层：{method.upper()} 参数和随机种子敏感性\n绿=成功，红=失败；不同面板的方向、形状及坐标不能直接比较',fontsize=13);save(fig,'13_'+method+'_robustness')
(P/'embedding_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
index=[json.loads(s) for s in (R/'mug_features_replay_20261118/query_index.jsonl').read_text().splitlines()];lookup={x['episode_id']:i for i,x in enumerate(index) if x['policy_step']==0};ii=[lookup[id] for id in ids];B=np.load(R/'mug_features_replay_20261118/pooled_features.npz');tiles=np.floor((xyrel+10)/2.5).astype(int);groups=tiles[:,0]*8+tiles[:,1]
splits=[]
for tr,te in GroupKFold(n_splits=5).split(xy,groups=groups):
 dist=pairwise_distances(xy[tr],xy[te]);tr=tr[dist.min(1)>=.5];splits.append((tr,te))
(P/'probe_splits.json').write_text(json.dumps([{'train':a.tolist(),'test':b.tolist()} for a,b in splits]))
probe=[];drift=[]
for key in keys:
 bx=B[key][ii].astype(float);vx=X[key].astype(float);bn=normalize(bx);vn=normalize(vx);cos=np.clip(1-(bn*vn).sum(1),0,2);rel=np.linalg.norm(vx-bx,axis=1)/np.maximum(np.linalg.norm(bx,axis=1),1e-12);drift.append({'feature':key,'cosine_distance_median':float(np.median(cos)),'relative_l2_median':float(np.median(rel)),'max_abs_difference':float(np.max(abs(vx-bx)))})
 for name,xx in [('baseline',bn),('v2',vn)]:
  pred=np.zeros_like(xy)
  for tr,te in splits:pred[te]=Ridge(alpha=1.).fit(xx[tr],xy[tr]).predict(xx[te])
  errs=np.linalg.norm(pred-xy,axis=1)
  for l in range(4):probe.append({'feature':key,'model':name,'level':l,'mean_error_cm':float(errs[lev==l].mean())})
constpred=np.zeros_like(xy)
for tr,te in splits:constpred[te]=xy[tr].mean(0)
consterr=np.linalg.norm(constpred-xy,axis=1)
for name in ['baseline','v2']:
 for l in range(4):probe.append({'feature':'constant','model':name,'level':l,'mean_error_cm':float(consterr[lev==l].mean())})
pkeys=['constant']+keys;ptitles=['训练均值基线']+titles
fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for ax,name,title in zip(axs,['baseline','v2'],['原始模型','v2 LoRA']):
 vals=np.array([[next(x['mean_error_cm'] for x in probe if x['feature']==k and x['model']==name and x['level']==l) for l in range(4)] for k in pkeys]);im=ax.imshow(vals,cmap='YlOrRd',vmin=0,vmax=max(x['mean_error_cm'] for x in probe),aspect='auto')
 for i in range(7):
  for j in range(4):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=10)
 ax.set(yticks=range(7),yticklabels=ptitles,xticks=range(4),xticklabels=['中心','外环 1','外环 2','外环 3'],title=title);fig.colorbar(im,ax=ax,label='XY 误差（cm）')
fig.suptitle('首查询特征的位置解码：两模型使用同一空间留出划分\n5 折、2.5 cm 空间块、0.5 cm 测试邻域排除；L2 特征 + 岭回归 α=1 固定；不是原报告的调参探针',fontsize=12);save(fig,'14_position_probe_comparison')
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
axs[0].barh(titles,[x['relative_l2_median']*100 for x in drift],color='#e59734');axs[0].set(xlabel='相对 L2 差异中位数（%）',title='同坐标、同首查询：特征改变量');axs[0].invert_yaxis()
axs[1].barh(titles,[x['cosine_distance_median'] for x in drift],color='#377eb8');axs[1].set(xlabel='余弦距离中位数',title='特征方向的变化');axs[1].invert_yaxis()
fig.suptitle('原始模型 → v2 的特征变化\n基于相同初始观测和策略种子；变化幅度本身不能说明性能改善或因果关系',fontsize=13);save(fig,'15_feature_change')
(P/'probe_results.json').write_text(json.dumps(probe,indent=2));(P/'feature_change.json').write_text(json.dumps(drift,indent=2));print('FEATURE_ANALYSIS_COMPLETE',flush=True)
