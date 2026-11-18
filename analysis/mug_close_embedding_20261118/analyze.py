import os
import json
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

P=Path(__file__).resolve().parent
S=Path(os.environ.get('XVLA_DATA_DIR', P.parent/'mug_features_replay_20261118'))
C=P.parent/'mug_correction_analysis_20261118'
events={e['id']:e for e in json.loads((C/'events.json').read_text()) if e['type']=='first_close'}
index={(r['episode_id'],r['policy_step']):r for r in map(json.loads,(S/'query_index.jsonl').read_text().splitlines())}
rows=[r for r in json.loads((C/'query_metrics.json').read_text()) if r['type']=='first_close']
for r in rows:
    e=events[r['id']]
    r['outcome']=2 if e['episode_success'] else int(e['stable_grasp_by_query_plus90_including_prequery_recovery'])
    r['feature_row']=index[r['id'],r['query_step']]['row']
assert len({(r['id'],r['offset']) for r in rows})==len(rows)
(P/'samples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
phase=np.array([r['offset'] for r in rows]); y=np.array([r['outcome'] for r in rows]); ring=np.array([r['level'] for r in rows]); ids=np.array([r['id'] for r in rows])
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
colors=['#d75a52','#dba32e','#168a76']; labels=['窗口内无新持续抓持，最终失败','窗口内有新持续抓持，最终失败','最终成功']
titles=['闭合前最后一次推理','闭合后第 1 次推理','闭合后第 2 次推理','闭合后第 3 次推理']
keys=['vision_wrist_mean','fusion_visual_mean','block_23_all_mean']; names=['腕部视觉','融合视觉','动作模块末层']
data=np.load(S/'pooled_features.npz'); metrics=[]; embeds={}
for key in keys:
    X=data[key][[r['feature_row'] for r in rows]].astype(np.float32)
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    for seed in [0,1]:
        out=P/f'tsne_{key}_seed{seed}.npy'
        if out.exists(): Z=np.load(out)
        else:
            Z=TSNE(n_components=2,perplexity=30,init='random',learning_rate='auto',max_iter=1500,random_state=seed,n_jobs=4).fit_transform(X)
            np.save(out,Z)
        embeds[key,seed]=Z
        metrics.append({'feature':key,'seed':seed,'trustworthiness_10':float(trustworthiness(X,Z,n_neighbors=10))})
        print(key,seed,metrics[-1],flush=True)
    for off in [-1,0,1,2]:
        m=phase==off; xx=X[m]; yy=y[m]; rr=ring[m]
        candidates=NearestNeighbors(n_neighbors=11).fit(xx).kneighbors(xx,return_distance=False)
        nn=np.array([v[v!=i][:10] for i,v in enumerate(candidates)])
        observed=float(np.mean(yy[nn]==yy[:,None])); rng=np.random.default_rng(195); null=[]
        for _ in range(500):
            yp=yy.copy()
            for g in range(4): yp[rr==g]=rng.permutation(yp[rr==g])
            null.append(np.mean(yp[nn]==yp[:,None]))
        metrics.append({'feature':key,'offset':off,'n':int(m.sum()),'high_dim_same_outcome_10nn':observed,'within_ring_shuffle_mean':float(np.mean(null)),'permutation_p_uncorrected':float((1+sum(v>=observed for v in null))/501)})

def save(fig,name):
    fig.savefig(P/(name+'.png'),dpi=170);fig.savefig(P/(name+'.pdf'));plt.close(fig)
def scatter(ax,Z,m):
    for g in range(3):
        v=m&(y==g);ax.scatter(Z[v,0],Z[v,1],s=11,c=colors[g],alpha=.72,label=labels[g])
    ax.set_xticks([]);ax.set_yticks([])
for seed in [0,1]:
    fig,axs=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for i,key in enumerate(keys):
        Z=embeds[key,seed]
        for j,off in enumerate([-1,0,1,2]):
            ax=axs[i,j];m=phase==off;scatter(ax,Z,m)
            ax.set_xlim(Z[:,0].min()-2,Z[:,0].max()+2);ax.set_ylim(Z[:,1].min()-2,Z[:,1].max()+2)
            if i==0: ax.set_title(titles[j]+f'（{m.sum()} 条）')
            if j==0: ax.set_ylabel(names[i],fontsize=14)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=10)
    fig.suptitle(f'首次夹爪闭合前后的特征：t-SNE，种子 {seed}\n每行所有时刻联合拟合；各列共享坐标；不跨行比较距离；缺失后续推理不补点',fontsize=15)
    save(fig,f'close_stages_seed{seed}_zh')

fig,axs=plt.subplots(3,3,figsize=(13,11),layout='constrained')
for i,key in enumerate(keys):
    Z=embeds[key,0];m=phase==0
    scatter(axs[i,0],Z,m)
    for g,c in enumerate(['#2580bb','#23a58b','#e89832','#9b63bd']):
        v=m&(ring==g);axs[i,1].scatter(Z[v,0],Z[v,1],s=14,c=c,label=['5×5 内部','5→10 环','10→15 环','15→20 环'][g])
    grasp=np.array([r['query_grasped'] for r in rows])
    for g,c,label in [(False,'#8f8f8f','当前未检测到抓持'),(True,'#386cb0','当前检测到抓持')]:
        v=m&(grasp==g);axs[i,2].scatter(Z[v,0],Z[v,1],s=14,c=c,label=label)
    for ax in axs[i]:ax.set_xticks([]);ax.set_yticks([])
    axs[i,0].set_ylabel(names[i],fontsize=14)
for ax,title in zip(axs[0],['按后续结果着色','按初始采样区域着色','按查询当时抓持检测着色']):ax.set_title(title)
for ax in axs[-1]:ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.02))
fig.suptitle('闭合后第 1 次推理：同一坐标的三种解释\n位置与当前抓持状态均可能造成聚类，不能把分群直接解释为失败原因',fontsize=15)
save(fig,'close_context_zh')

complete=[eid for eid in events if sum(ids==eid)==4]
selected=[]
for g in range(3):
    selected.extend(sorted([eid for eid in complete if next(r['outcome'] for r in rows if r['id']==eid)==g])[:2])
fig,axs=plt.subplots(3,3,figsize=(13,11),layout='constrained')
for i,key in enumerate(keys):
    Z=embeds[key,0]
    for g in range(3):
        ax=axs[i,g];ax.scatter(Z[:,0],Z[:,1],s=3,c='#cccccc',alpha=.22)
        for eid in selected:
            ix=np.where(ids==eid)[0];ix=ix[np.argsort(phase[ix])]
            if y[ix[0]]!=g:continue
            z=Z[ix];ax.plot(z[:,0],z[:,1],'-o',ms=4,lw=1,label=eid)
            for k in range(3):ax.annotate('',xy=z[k+1],xytext=z[k],arrowprops={'arrowstyle':'->','color':'#333333','lw':.7})
            for k,pt in enumerate(z):ax.annotate(['前','后1','后2','后3'][k],pt,fontsize=8)
        ax.set_xticks([]);ax.set_yticks([]);ax.legend(fontsize=8)
        if i==0:ax.set_title(labels[g],fontsize=11)
        if g==0:ax.set_ylabel(names[i])
fig.suptitle('首次闭合前后的特征路径示例\n每组取完整随访中编号最小的两条，非典型性筛选；箭头长度不代表纠偏强弱',fontsize=14)
save(fig,'close_paths_zh')
summary={'episodes':len(events),'points':len(rows),'phase_counts':{str(o):int(sum(phase==o)) for o in [-1,0,1,2]},'complete_four_query_episodes':len(complete),'outcomes':{labels[g]:sum(next(r['outcome'] for r in rows if r['id']==eid)==g for eid in events) for g in range(3)},'selected_examples':selected,'metrics':metrics}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},ensure_ascii=False),flush=True)
