import os
from pathlib import Path
import json,time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
P=Path(__file__).resolve().parent;S=Path(os.environ.get('XVLA_DATA_DIR', str(P.parents[1]/'data'/'mug_features_replay_20261118')));rows=json.loads((P/'episodes.json').read_text());sp=json.loads((P/'spatial_groups.json').read_text());groups=np.array(sp['groups']);y=np.array(sp['xy_cm']);rings=np.array([r['level'] for r in rows]);q=[json.loads(s) for s in (S/'query_index.jsonl').read_text().splitlines()];ix={r['episode_id']:i for i,r in enumerate(q) if r['policy_step']==0};idx=[ix[r['id']] for r in rows];features=np.load(S/'pooled_features.npz')
keys=['vision_main_mean','vision_wrist_mean','fusion_visual_mean','block_00_all_mean','block_11_all_mean','block_23_all_mean'];alphas=[.001,.01,.1,1,10,100,1000]
outer=[]
for train,test in GroupKFold(n_splits=5,shuffle=True,random_state=195).split(y,groups=groups):
 train=train[cdist(y[train],y[test]).min(axis=1)>=.5]
 assert not set(groups[train])&set(groups[test]);assert cdist(y[train],y[test]).min()>=.5
 outer.append((train,test))
(P/'probe_splits.json').write_text(json.dumps({'description':'5 outer spatial-group folds,2.5cm tiles; remove training points within0.5cm of held-out test points; alpha selected by3 inner spatial-group folds on training only','folds':[{'train':[rows[i]['id'] for i in a],'test':[rows[i]['id'] for i in b],'n_train':len(a),'n_test':len(b)} for a,b in outer],'alphas':alphas,'preprocessing':'row L2; feature-column StandardScaler fitted on training only; no PCA; targets centered XY cm','targets':'cup initial XY; relative-to-EEF XY differs by fixed offset because initial EEF is constant'},indent=2))
def fitpredict(X,train,test,alpha):
 sc=StandardScaler().fit(X[train]);xx=sc.transform(X[train]);xt=sc.transform(X[test]);model=Ridge(alpha=alpha,solver='cholesky').fit(xx,y[train]);return model.predict(xt)
def choose(X,train):
 folds=list(GroupKFold(n_splits=min(3,len(np.unique(groups[train])))).split(train,groups=groups[train]));loss=[]
 for alpha in alphas:
  residual=[]
  for aa,bb in folds:
   a=train[aa];b=train[bb];p=fitpredict(X,a,b,alpha);residual.extend(np.sum((p-y[b])**2,axis=1))
  loss.append(float(np.mean(residual)))
 return alphas[int(np.argmin(loss))],loss
predictions={};results=[];selections={}
# Mean predictor corresponds also to state-only baseline: initial state is fixed.
constant=np.zeros_like(y)
for a,b in outer:constant[b]=y[a].mean(0)
predictions['spatial__constant']=constant
for key in keys:
 X=normalize(features[key][idx].astype(np.float64));pred=np.zeros_like(y);selections[key]=[]
 for fold,(a,b) in enumerate(outer):
  alpha,loss=choose(X,a);pred[b]=fitpredict(X,a,b,alpha);selections[key].append({'fold':fold,'alpha':alpha,'inner_mse_cm2':loss});print('FOLD',key,fold,alpha,flush=True)
 predictions['spatial__'+key]=pred
 # Second experiment: tune exclusively inside default 5x5; test all three outer rings.
 a=np.flatnonzero(rings==0);b=np.flatnonzero(rings>0);alpha,loss=choose(X,a);out=fitpredict(X,a,b,alpha);ex=np.full_like(y,np.nan);ex[b]=out;predictions['extrapolation__'+key]=ex;selections[key].append({'protocol':'train_default_test_outer','alpha':alpha,'inner_mse_cm2':loss})
ex=np.full_like(y,np.nan);ex[rings>0]=y[rings==0].mean(0);predictions['extrapolation__constant']=ex
rng=np.random.default_rng(20261118)
for name,pred in predictions.items():
 protocol,key=name.split('__')
 for level in range(4):
  ii=np.flatnonzero((rings==level)&np.isfinite(pred).all(axis=1))
  if not len(ii):continue
  err=np.linalg.norm(pred[ii]-y[ii],axis=1);ug=np.unique(groups[ii]);boots=[]
  for _ in range(1000):
   sel=rng.choice(ug,len(ug),replace=True);j=np.concatenate([np.flatnonzero(groups[ii]==g) for g in sel]);boots.append(err[j].mean())
  results.append({'protocol':protocol,'feature':key,'level':level,'n':len(ii),'n_spatial_groups':len(ug),'mean_euclidean_error_cm':float(err.mean()),'median_euclidean_error_cm':float(np.median(err)),'rmse_euclidean_cm':float(np.sqrt(np.mean(err**2))),'mean_error_tile_bootstrap95':np.quantile(boots,[.025,.975]).tolist()})
np.savez_compressed(P/'probe_predictions.npz',target_xy_cm=y,**predictions);(P/'probe_results.json').write_text(json.dumps(results,indent=2));(P/'probe_alpha_selection.json').write_text(json.dumps(selections,indent=2));print('PROBES_DONE',flush=True)
