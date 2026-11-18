import os
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent; B=P.parent; S=Path(os.environ.get('XVLA_DATA_DIR', B/'mug_features_replay_20261118'))
rows=[r for r in json.loads((B/'mug_close_embedding_20261118/samples.json').read_text()) if r['offset']==0 and not r['query_grasped']]
ix={r['row']:r for r in map(json.loads,(S/'query_index.jsonl').read_text().splitlines())}
a=[r for r in rows if r['episode_success']];b=[r for r in rows if not r['episode_success']]
keys=['vision_wrist_mean','fusion_visual_mean','block_23_all_mean']; names=['腕部视觉','融合视觉','动作模块末层']
F=np.load(S/'pooled_features.npz'); features={k:F[k] for k in keys}
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
def position(r,field):return np.array(ix[r['feature_row']][field][:2])*100
def match(field,t):
 d=cdist([position(r,field) for r in a],[position(r,field) for r in b]); cost=np.concatenate([np.where(d<=t,d,1e6),np.full((len(a),len(a)),1000.)],axis=1)
 u,v=linear_sum_assignment(cost)
 return [(a[i],b[j]) for i,j in zip(u,v) if j<len(b)]
def save(fig,name):fig.savefig(P/(name+'.png'),dpi=180);fig.savefig(P/(name+'.pdf'));plt.close(fig)
def evaluate(pairs,key):
 X=features[key][[r['feature_row'] for pair in pairs for r in pair]].astype(float);X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
 D=X[::2]-X[1::2];n=len(D);G=D@D.T;np.fill_diagonal(G,0)
 alignment=G.sum(1)/(n-1);obs=float(np.mean(alignment>0));coherence=float(G.sum()/(n*(n-1)))
 signs=np.random.default_rng(195).choice([-1,1],size=(5000,n));scores=signs*(signs@G)/(n-1);null=(scores>0).mean(1);cn=scores.mean(1)
 return X,{'n_pairs':n,'leave_one_pair_out_ranking_fraction':obs,'correct_pairs':int(sum(alignment>0)),'ranking_permutation_p':float((1+sum(null>=obs))/5001),'mean_cross_pair_dot':coherence,'coherence_permutation_p':float((1+sum(cn>=coherence))/5001),'null_ranking_mean':float(null.mean())}
summary={'eligible':len(rows),'success':len(a),'failure':len(b),'query_goal_true':sum(r['query_goal'] for r in rows),'analyses':[]};cohorts={};embeds={}
for field in ['initial_xy_m','mug_xyz']:
 for t in [.5,1.,2.]:
  pairs=match(field,t);record={'match_field':field,'caliper_cm':t,'n_pairs':len(pairs),'metrics':{}}
  for k in keys: _,record['metrics'][k]=evaluate(pairs,k)
  record['median_initial_distance_cm']=float(np.median([np.linalg.norm(position(s,'initial_xy_m')-position(f,'initial_xy_m')) for s,f in pairs]))
  record['median_query_cup_distance_cm']=float(np.median([np.linalg.norm(position(s,'mug_xyz')-position(f,'mug_xyz')) for s,f in pairs]))
  record['median_query_eef_distance_cm']=float(np.median([np.linalg.norm(position(s,'eef_xyz')-position(f,'eef_xyz')) for s,f in pairs]))
  summary['analyses'].append(record)
  if t!=1.:continue
  cohorts[field]=pairs
  (P/f'pairs_{field}.json').write_text(json.dumps([{'pair':i+1,'success':s,'failure':f,'success_index':ix[s['feature_row']],'failure_index':ix[f['feature_row']]} for i,(s,f) in enumerate(pairs)],ensure_ascii=False,indent=2))
  for k in keys:
   X,_=evaluate(pairs,k)
   for seed in [0,1]:
    perplexity=min(10,max(3,(len(X)-1)//3));Z=TSNE(perplexity=perplexity,init='random',learning_rate='auto',max_iter=1500,random_state=seed,n_jobs=4).fit_transform(X)
    embeds[field,k,seed]=Z;np.save(P/f'tsne_{field}_{k}_seed{seed}.npy',Z)
for seed in [0,1]:
 fig,axs=plt.subplots(3,2,figsize=(12,13),layout='constrained')
 for j,(field,title) in enumerate([('initial_xy_m','初始杯子位置匹配'),('mug_xyz','推理时杯子位置匹配')]):
  n=len(cohorts[field])
  for i,(k,name) in enumerate(zip(keys,names)):
   ax=axs[i,j];Z=embeds[field,k,seed]
   for p in range(n):ax.plot(Z[2*p:2*p+2,0],Z[2*p:2*p+2,1],color='#b0b0b0',lw=.65,alpha=.7,zorder=0)
   ax.scatter(Z[::2,0],Z[::2,1],c='#168a76',s=35,label='最终成功');ax.scatter(Z[1::2,0],Z[1::2,1],c='#d75a52',marker='x',s=35,label='最终失败')
   ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{name}｜{title}：{n} 对',fontsize=12)
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
 fig.suptitle(f'闭合后首次推理仍未检测抓持：位置相近的成功／失败配对\nt-SNE，种子 {seed}；配对距离 ≤1 厘米；灰线连接配对样本，各面板独立拟合',fontsize=14)
 save(fig,f'matched_tsne_seed{seed}_zh')
fig,axs=plt.subplots(2,2,figsize=(11,10),layout='constrained')
for i,(field,title) in enumerate([('initial_xy_m','按初始位置配对'),('mug_xyz','按推理时位置配对')]):
 for j,(view,label) in enumerate([('initial_xy_m','初始杯子位置'),('mug_xyz','推理时杯子位置')]):
  ax=axs[i,j]
  for s,f in cohorts[field]:
   z=np.array([position(s,view),position(f,view)])-[-10,-15];ax.plot(z[:,0],z[:,1],c='#bbb',lw=.8);ax.scatter(*z[0],c='#168a76',s=22);ax.scatter(*z[1],c='#d75a52',marker='x',s=22)
  ax.set(xlabel='相对默认中心 X（厘米）',ylabel='相对默认中心 Y（厘米）',title=f'{title}｜检查{label}',aspect='equal');ax.grid(alpha=.2)
fig.suptitle('配对质量检查：绿色成功，红色失败，灰线为同一对\n初始位置相近不保证闭合后位置仍相近；未匹配高度、姿态及机械臂状态',fontsize=14)
save(fig,'position_matching_zh')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for j,field in enumerate(cohorts):
 rec=next(r for r in summary['analyses'] if r['match_field']==field and r['caliper_cm']==1)
 rates=[rec['metrics'][k]['leave_one_pair_out_ranking_fraction']*100 for k in keys];axs[j].bar(names,rates,color=['#5588bb','#55aa99','#bb88aa']);axs[j].axhline(50,c='gray',ls='--')
 for i,k in enumerate(keys):
  m=rec['metrics'][k];axs[j].text(i,rates[i]+2,f"{m['correct_pairs']}/{m['n_pairs']}\np={m['ranking_permutation_p']:.3f}",ha='center')
 axs[j].set(ylim=(0,115),ylabel='留一对排序正确比例（%）',title=['初始位置匹配','推理时位置匹配'][j])
fig.suptitle('原始高维特征检查：其余配对的平均差异能否判断留出对？\n固定方法，无调参；p 为配对标签翻转检验（5000 次，未经多重比较校正）',fontsize=13)
save(fig,'highdim_pair_ranking_zh')
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
