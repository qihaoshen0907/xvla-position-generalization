from pathlib import Path
import os
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import json,hashlib,time,importlib.metadata as im
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,trustworthiness
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import umap
P=Path(__file__).resolve().parent;SRC=Path(os.environ.get('XVLA_DATA_DIR', str(P.parents[1]/'data'/'mug_features_replay_20261118')))
rows=[json.loads(s) for s in (SRC/'query_index.jsonl').read_text().splitlines()];idx=np.array([i for i,r in enumerate(rows) if r['policy_step']==0]);r=[rows[i] for i in idx]
assert len(r)==400 and len({x['episode_id'] for x in r})==400
labels=np.array([x['episode_success'] for x in r]);rings=np.array([x['level'] for x in r]);xy=np.array([x['initial_xy_m'] for x in r]);xycm=(xy-[-.1,-.15])*100
keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean'];titles=['Main-view vision','Wrist-view vision','Fused visual tokens','Action block 00','Action block 11','Action block 23']
config={'sample':'first pre-action query per episode','n':400,'successes':int(labels.sum()),'features':keys,'normalization':'row L2; no PCA, no label-based preprocessing','tsne':{'perplexity':[15,30,50],'seeds':[0,1,2],'init':'random','metric':'euclidean on L2-normalized vectors','max_iter':1500,'learning_rate':'auto','method':'barnes_hut'},'umap':{'n_neighbors':[15,30,50],'seeds':[0,1,2],'min_dist':.1,'metric':'euclidean on L2-normalized vectors','init':'random','n_epochs':500},'main_plot':'parameter30 seed0 selected before seeing embeddings','diagnostics':'10 nearest neighbors; 1000 global/within-ring label permutations; exploratory, unadjusted p values; position confounding not removed by ring conditioning','versions':{x:im.version(x) for x in ['numpy','scipy','scikit-learn','umap-learn']},'input_sha256':{n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['pooled_features.npz','query_index.jsonl']}}
(P/'config.json').write_text(json.dumps(config,indent=2));(P/'samples.json').write_text(json.dumps(r,indent=2));np.savez(P/'sample_labels.npz',success=labels,ring=rings,xy_m=xy,source_row=idx)
features=np.load(SRC/'pooled_features.npz');emb={};metrics=[];high=[]
Dxy=pairwise_distances(xycm);tri=np.triu_indices(400,1)
def neighbors(d):
 dd=d.copy();np.fill_diagonal(dd,np.inf);return np.argsort(dd,axis=1,kind='stable')[:,:10]
def mix(nn,y):return float(np.mean(y[:,None]==y[nn]))
def overlap(a,b):return float(np.mean([len(set(x)&set(y))/10 for x,y in zip(a,b)]))
for key in keys:
 X=normalize(features[key][idx].astype(np.float64));assert np.isfinite(X).all();D=pairwise_distances(X);nn=neighbors(D)
 observed=mix(nn,labels);rng=np.random.default_rng(20261118);global_null=[];ring_null=[]
 for _ in range(1000):
  global_null.append(mix(nn,rng.permutation(labels)));yy=labels.copy()
  for level in range(4):yy[rings==level]=rng.permutation(yy[rings==level])
  ring_null.append(mix(nn,yy))
 h={'feature':key,'unique_vectors':len(np.unique(X,axis=0)),'same_outcome_neighbor_fraction':observed,'global_shuffle_mean':float(np.mean(global_null)),'within_ring_shuffle_mean':float(np.mean(ring_null)),'within_ring_excess':observed-float(np.mean(ring_null)),'within_ring_p_one_sided':float((1+sum(v>=observed for v in ring_null))/1001),'same_ring_neighbor_fraction':float(np.mean(rings[:,None]==rings[nn])),'neighbor_position_distance_cm':float(np.mean(Dxy[np.arange(400)[:,None],nn])),'all_pair_position_distance_cm':float(Dxy[tri].mean()),'feature_vs_position_distance_spearman':float(spearmanr(D[tri],Dxy[tri]).statistic)}
 high.append(h);print('HIGH',key,h,flush=True)
 for method in ['tsne','umap']:
  for param in [15,30,50]:
   for seed in [0,1,2]:
    name=f'{method}__{key}__p{param}__s{seed}';fp=P/(name+'.npy');t=time.monotonic()
    if fp.exists():Y=np.load(fp)
    elif method=='tsne':Y=TSNE(n_components=2,perplexity=param,init='random',learning_rate='auto',max_iter=1500,random_state=seed,n_jobs=4).fit_transform(X);np.save(fp,Y)
    else:Y=umap.UMAP(n_neighbors=param,min_dist=.1,metric='euclidean',init='random',random_state=seed,n_epochs=500,n_jobs=1).fit_transform(X);np.save(fp,Y)
    assert Y.shape==(400,2) and np.isfinite(Y).all();emb[name]=Y;enn=neighbors(pairwise_distances(Y))
    metrics.append({'method':method,'feature':key,'parameter':param,'seed':seed,'trustworthiness_k10':float(trustworthiness(X,Y,n_neighbors=10)),'high_low_neighbor_overlap_k10':overlap(nn,enn),'embedded_same_outcome_neighbor_fraction':mix(enn,labels),'seconds':time.monotonic()-t})
    (P/'progress.json').write_text(json.dumps({'embeddings_completed':len(metrics),'total':108,'latest':name}));print('EMBED',name,flush=True)
(P/'high_dimensional_diagnostics.json').write_text(json.dumps(high,indent=2));(P/'embedding_metrics.json').write_text(json.dumps(metrics,indent=2))
stability=[]
for method in ['tsne','umap']:
 for key in keys:
  seednn=[neighbors(pairwise_distances(emb[f'{method}__{key}__p30__s{s}'])) for s in [0,1,2]]
  stability.append({'method':method,'feature':key,'seed_pair_neighbor_overlap_k10':[overlap(seednn[a],seednn[b]) for a,b in [(0,1),(0,2),(1,2)]]})
(P/'seed_stability.json').write_text(json.dumps(stability,indent=2))
colors=['#2580bb','#23a58b','#e89832','#9b63bd']
def outcome(ax,Y):
 for label,col,mk,name in [(False,'#d44848','x','Failure (267)'),(True,'#168567','o','Success (133)')]:
  sel=labels==label;ax.scatter(Y[sel,0],Y[sel,1],c=col,marker=mk,s=15,linewidths=.75,alpha=.8,label=name)
 ax.set_xticks([]);ax.set_yticks([])
for method in ['tsne','umap']:
 fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
 for ax,key,title in zip(axs.flat,keys,titles):
  Y=emb[f'{method}__{key}__p30__s0'];outcome(ax,Y);ax.set_title(title)
 handles,labs=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside lower center',ncol=2)
 fig.suptitle(f'{method.upper()} | First query of 400 episodes | parameter=30, seed=0\nEach panel fits a separate feature space; cluster distances are not physical distances',fontsize=13)
 fig.savefig(P/f'{method}_outcome.png',dpi=180);fig.savefig(P/f'{method}_outcome.pdf');plt.close(fig)
 fig,axs=plt.subplots(6,3,figsize=(12,20),layout='constrained')
 for rownum,(key,title) in enumerate(zip(keys,titles)):
  Y=emb[f'{method}__{key}__p30__s0']
  for level,col in enumerate(colors):
   sel=rings==level;axs[rownum,0].scatter(Y[sel,0],Y[sel,1],c=col,s=13,label=['5 cm square','5 to 10 cm ring','10 to 15 cm ring','15 to 20 cm ring'][level])
  axs[rownum,0].set_ylabel(title)
  for colnum,coordinate in [(1,0),(2,1)]:
   sc=axs[rownum,colnum].scatter(Y[:,0],Y[:,1],c=xycm[:,coordinate],cmap='coolwarm',vmin=-10,vmax=10,s=13);fig.colorbar(sc,ax=axs[rownum,colnum],shrink=.7,label='cm')
  for ax in axs[rownum]:ax.set_xticks([]);ax.set_yticks([])
 for ax,title in zip(axs[0],['Sampling region','Initial X relative to center','Initial Y relative to center']):ax.set_title(title)
 hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=4);fig.suptitle(f'{method.upper()} | Same coordinates as outcome panels | Position context')
 fig.savefig(P/f'{method}_position_context.png',dpi=160);plt.close(fig)
 for key in ['fusion_visual_mean','block_23_all_mean']:
  fig,axs=plt.subplots(3,3,figsize=(10,10),layout='constrained')
  for a,param in enumerate([15,30,50]):
   for b,seed in enumerate([0,1,2]):
    outcome(axs[a,b],emb[f'{method}__{key}__p{param}__s{seed}']);axs[a,b].set_title(f'parameter={param}, seed={seed}')
  fig.suptitle(f'{method.upper()} robustness | {key}');fig.savefig(P/f'{method}_robustness_{key}.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,6),layout='constrained');outcome(ax,xycm);ax.set(xlabel='Initial X relative to center (cm)',ylabel='Initial Y relative to center (cm)',title='Physical position of the same 400 episodes',aspect='equal');ax.set_xticks([-10,-5,0,5,10]);ax.set_yticks([-10,-5,0,5,10]);ax.legend();fig.savefig(P/'physical_positions.png',dpi=180);plt.close(fig)
(P/'complete.json').write_text(json.dumps({'samples':400,'embeddings':108,'no_pca':True,'features':6,'source_unchanged':True},indent=2));print('DONE',flush=True)
