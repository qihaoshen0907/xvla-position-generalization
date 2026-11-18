import os
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;S=Path(os.environ.get('XVLA_DATA_DIR', P.parent/'mug_features_replay_20261118'));es=json.loads((P/'events.json').read_text());q=json.loads((P/'query_metrics.json').read_text());summary=json.loads((P/'summary.json').read_text());pairs=json.loads((P/'paired_ungrasped_queries.json').read_text());inventory=json.loads((P/'event_inventory.json').read_text())
assert len(inventory)==400 and sum(e['type']=='first_close' for e in es)==400
assert len({(v['id'],v['type'],v['offset']) for v in q})==len(q)
assert all(v['evaluated_stop_step']>v['query_step'] for v in q)
assert all(e['next_query_step']>=e['event_step'] for e in es if e['has_post_query'])
thresholds=[];subgroups=[];stagnation=[]
for typ in ['first_close','first_far_loss']:
 for l in range(4):
  qq=[v for v in q if v['type']==typ and v['level']==l and v['offset']==0 and v['eligible_approach']]
  thresholds.append({'type':typ,'level':l,'n':len(qq),'thresholds_cm':{str(t):{'planned':sum(v['predicted_approach_cm']>t for v in qq),'actual_tracking':sum(v['actual_approach_cm']>t for v in qq),'actual_fixed_target':sum(v['actual_fixed_target_approach_cm']>t for v in qq),'both_plan_and_actual_tracking':sum(v['predicted_approach_cm']>t and v['actual_approach_cm']>t for v in qq)} for t in [0,.5,1]},'median_cup_motion_cm':float(np.median([v['cup_motion_to_stop_cm'] for v in qq]))})
  if typ=='first_far_loss':
   for op in [False,True]:
    ee=[e for e in es if e['type']==typ and e['level']==l and e['loss']['open_command_recent']==op];subgroups.append({'level':l,'recent_open_command':op,'n':len(ee),'stable_grasp_by90':sum(e.get('stable_grasp_by_query_plus90_including_prequery_recovery',False) for e in ee),'final_successes':sum(e['episode_success'] for e in ee)})
 for e in [e for e in es if e['type']==typ]:
  aa=[next((v for v in q if v['id']==e['id'] and v['type']==typ and v['offset']==i),None) for i in [0,1,2]]
  if all(v and v['eligible_approach'] and v['current_xy_gap_cm']>=3 for v in aa):
   gaps=[v['current_xy_gap_cm'] for v in aa];ends=np.array([v['predicted_endpoint_xyz_m'][:2] for v in aa])*100
   stagnant=bool(gaps[0]-gaps[-1]<.5 and np.max(np.linalg.norm(np.diff(ends,axis=0),axis=1))<1)
   stagnation.append({'id':e['id'],'type':typ,'level':e['level'],'gaps_cm':gaps,'max_adjacent_endpoint_shift_cm':float(np.max(np.linalg.norm(np.diff(ends,axis=0),axis=1))),'stagnant_by_rule':stagnant})
(P/'threshold_sensitivity.json').write_text(json.dumps(thresholds,indent=2));(P/'loss_subgroups.json').write_text(json.dumps(subgroups,indent=2));(P/'stagnation_candidates.json').write_text(json.dumps(stagnation,indent=2))
regions=['5×5内部','5→10新增环','10→15新增环','15→20新增环']
lines=['# 事件对齐后的再推理与纠偏分析','', '全部使用原来的400条轨迹、5,526个推理点及其特征/动作记录；未运行新仿真、未修改模型、未更改原成功率。本实验是任务开始前的位置扩展，不是额外施加中途扰动的对照。','', '## 结论','', '有可观察到的局部靠近动作和再次抓持，但不能等同于可靠任务恢复。尤其最外环：首次闭合后仍未检测抓持的89条中，44条预测终点更接近杯子、53条实际动态水平距离缩小，均以>0.5cm为阈值；当块仅9条新出现持续抓持。最终最外环仍0/100成功。','', '远盘抓持丢失后，最外环26条中的22条在随访窗口重新出现持续抓持检测，但0条最终成功；其中23条在事件当步及前4步出现过张开命令。这个统计不能描述为“22次意外滑脱被成功挽救”。','', '状态/特征更新与动作变化并不等于模型识别了错误。没有冻结观测或固定生成噪声的反事实对照，无法把变化唯一归因于反馈纠偏；这里的“纠偏”仅为明确的行为代理指标。','', '## 事件与时序','', '- 首次闭合：夹爪命令首次由非正变正的动作步，非物理闭合完成时刻。','- 远盘抓持丢失：至少连续5步抓持检测为真，随后至少连续3步为假；第一步为假的位置距目标盘中心XY>3cm且目标未成立。每条轨迹仅取首次符合条件事件。正常盘边释放不计入。','- 丢失事件按“当步及前4步是否出现负值张开指令”分组。无张开指令也不自动等于滑脱；抓持检测是接触代理。','- query_step表示已执行步数；取事件之前最后一次查询，以及事件发生后可观察到该状态的第1、2、3次查询。事件确认用后续3步，因此属于回顾性事件标注，不是在线检测器。','- 原实验每30步推理一次；事件后0–29步可能仍执行旧动作，20Hz下最长1.45秒，不能把这段直接叫“已重新推理但不反应”。','', '## 首次闭合后的第一次推理','', '下表仅纳入该次查询时抓持检测为假、目标未成立的轨迹；已抓住/已完成者不计入“还需接近”的分母。','', '| 区域 | 纳入数 | 预测终点更近>0.5cm | 实际动态距离缩小>0.5cm | 本块新持续抓持 | 到下次推理等待中位数 |','|---|---:|---:|---:|---:|---:|']
for l,v in enumerate(summary['first_close']):lines.append(f'| {regions[l]} | {v["eligible_first_post_queries"]} | {v["planned_toward_n"]} | {v["actual_toward_n"]} | {v["new_grasp_in_block_n"]} | {v["median_delay_steps"]:g}步 |')
lines+=['', '预测更近：查询时末端到杯子的XY距离减去预测动作块末端到查询时杯子位置的距离。它不检查姿态、接触、安全或放置能力。实际动态距离：块结束或首次新持续抓持开始（两者较早）时，末端与当时杯子的XY距离变化。提前停止避免把正常抓住后搬运误记成没有接近。','', '杯子自身移动也能改变动态距离，因此另存固定查询时杯子位置的距离变化。两者不能替代因果验证；另有0/0.5/1cm阈值敏感性表。Z、姿态调整不被XY指标覆盖，所以“XY没改善”不等于完全没有纠偏。','', '## 远盘抓持丢失后的第一次推理与恢复','', '| 区域 | 首次丢失事件数 | 仍需接近的查询数 | 预测更近 | 实际动态距离缩小 | 本块新持续抓持 | 随访窗口再次持续抓持 | 最终成功 | 附近有张开命令 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for l,v in enumerate(summary['first_far_loss']):lines.append('| '+regions[l]+' | '+' | '.join(str(v[k]) for k in ['events','eligible_first_post_queries','planned_toward_n','actual_toward_n','new_grasp_in_block_n','stable_grasp_by90_n','final_successes','recent_open_command_n'])+' |')
lines+=['', '随访窗口从丢失事件之后持续到首次后续查询+90步，包含下一次推理之前已经恢复的情况；总长度取决于等待时间，最长约119步。所有123条事件均有窗口随访或提前稳定完成，不将无后续观测当失败。再次抓持必须连续≥5步，不等于杯子成功放到盘上。','', '严格无近期张开命令的子组：','', '| 区域 | 子组数量 | 窗口再次持续抓持 | 最终成功 |','|---|---:|---:|---:|']
for v in subgroups:
 if not v['recent_open_command']:lines.append(f'| {regions[v["level"]]} | {v["n"]} | {v["stable_grasp_by90"]} | {v["final_successes"]} |')
lines+=['', '最外环严格子组只有3条，不能据此估计一般的意外滑脱恢复率。','', '## 物体运动与未改善样本','', '最外环丢失后仍需接近的23条中：实际末端相对“查询时杯子位置”更近17条，相对“随后杯子位置”更近只有7条；该窗口杯子水平移动中位数1.38cm。需要区分追向旧位置与真正缩短当前距离；物体移动可能由机器人动作本身引起，不是本实验额外施加的扰动。','', 'paired_distance_zh.png只比较两次查询均未检测抓持、目标未成立的同一轨迹。其间可能短暂抓持后再丢失；它是困难样本子集，不代表全体平均恢复能力，尤其默认区仅8条配对。','', f'重复计划/XY停滞候选规则：事件后连续3个查询均未抓持、目标未成立且XY距离≥3cm；第一到第三查询距离改善<0.5cm，且相邻预测末端XY位移均<1cm。符合完整观察条件{len(stagnation)}条事件轨迹，候选{sum(x["stagnant_by_rule"] for x in stagnation)}条。候选名单保存在stagnation_candidates.json，仅是待检查线索，不自动判定网络进入死循环。','', '## 内部特征','', '相对事件前最后一次推理，比较腕部视觉、融合视觉、动作末层的余弦距离。每个区域/事件类型只用同时有3次后续查询的固定队列；每条轨迹一次。特征确有变化并不表明纠偏正确，机器人姿态、遮挡和物体位置都在变；动作层还受每次查询不同生成噪声影响。早完成而不足3次后续查询的轨迹不进入这张图，样本量标注在图例中。','', '## 文件与复现','', '- correction_rates_zh.png/.pdf：计划靠近、实际靠近与新抓持的比例及分母。','- loss_recovery_zh.png/.pdf：再次抓持、任务完成和张开指令分组。','- paired_distance_zh.png/.pdf：两次后续查询的配对距离。','- query_latency_zh.png/.pdf：事件到下一次推理的等待。','- feature_evolution_zh.png/.pdf：固定队列的内部表示变化。','- events.json、query_metrics.json、event_inventory.json：事件时刻、查询对齐及逐条指标。','- loss_subgroups.json、threshold_sensitivity.json、stagnation_candidates.json：敏感性和候选样本。','- analyze.py → report.py → plots.py；数据来源为相邻mug_features_replay_20261118目录。','', '本轮分析结果已纳入本公开仓库；原始特征数据未发布。']
(P/'README.md').write_text('\n'.join(lines)+'\n')
(P/'complete.json').write_text(json.dumps({'episodes':400,'close_events':400,'far_loss_events':sum(e['type']=='first_far_loss' for e in es),'aligned_event_query_rows':len(q),'unique_event_query_keys_verified':True,'no_simulation_run':True,'source_hashes':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['manifest.json','query_index.jsonl','pooled_features.npz']}},indent=2));print('REPORT_DONE',len(stagnation),sum(v['stagnant_by_rule'] for v in stagnation))
