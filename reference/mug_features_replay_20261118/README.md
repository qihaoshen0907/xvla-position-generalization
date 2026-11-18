# XVLA 400条原模型轨迹：一致性重放与内部特征

**状态：已恢复并全部完成。400条轨迹、5,526个推理点均已采集并通过文件核验。原始记录与重放的状态/速度及重算动作最大误差全部为0。**

15组池化向量每组形状[5526,1024]，与query_index.jsonl逐行对齐。完整HDF5约12,631.6MB，池化文件约252.8MB。原实验的成功轨迹133条、失败267条；不要按帧数等同统计轨迹数。

原始输入像素没有历史基准可比；这里只验证保存状态与重算动作一致。尚未进行降维拟合或失败原因推断。

数据来源：默认5×5 cm的100条轨迹，以及5→10、10→15、15→20 cm三个新增方环各100条。原实验成功数89、36、8、0。只执行原始实际动作，重新推理仅用于采集与核对，不控制重放。

## 验证与准入

`manifest.json` 预先固定400条来源及原结果/动作/指标文件SHA256，以及12条试采集名单。每个区域先取第一条成功（若有）、第一条失败，再按距已选点的最小距离最大化补足3条。最外环全为失败。

12条每个推理点都进行关闭采集/打开采集对照，要求动作完全相同；全部轨迹的重算动作块与原保存块最大绝对误差不超过1e-6。每一步末端位置、杯子free-joint位置、杯子body位置与速度误差不超过1e-8（分别m与m/s），抓持/目标布尔值必须一致。首次不一致即停止。通过12条准入并确认容量后才扩展。

原实验未保存原始无损图像，不能宣称与原始图像逐像素验证一致；动作与保存状态的一致性是重建依据。MuJoCo body_xpos与free-joint qpos分别对齐原记录中的同一字段，不混用。

## 文件

- `episodes/<ID>/features.h5`：每条轨迹完整特征和重建观测。
- `episodes/<ID>/validation.json`：逐查询核对、最大误差、文件大小。
- `pilot_gate.json`：12条准入结果。
- `collection_complete.json`：400条采集汇总。
- `verified_complete.json`：离线逐数据集读取、形状/完整性与有限值核对结果，生成后才代表交付完成。
- `query_index.jsonl`：与池化向量逐行对齐的标签、坐标、阶段及HDF5定位。
- `pooled_features.npz`：供后续PCA/UMAP使用的float32向量。未拟合降维，未将结果标签作为特征。
- `feature_file_manifest.json`：400个HDF5文件大小和SHA256。
- `read_features.py`：无损读取BF16张量。

## HDF5结构

`queries/0000` 等编号是**已执行的策略步数**，保存下一动作执行前的观测。每30步查询一次，末尾仅保留原实际执行长度，未延长失败或成功轨迹。

- `inputs`：实际送入模型的input_ids、image_input、image_mask、domain_id、proprio。
- `raw_images`：两路未压缩uint8原始图像，保持仿真返回方向；不要使用视频朝向替代。
- `raw_observation`：其他数值观测。
- `vision/valid_view_tokens`：2×50×1024，顺序为主视角、腕部视角。
- `fusion/vlm_features`：1×100×1024；前50主视角视觉tokens、后50文本序列tokens。
- `fusion/aux_visual_inputs`：1×150×1024；腕部视角50tokens，其后两个空视角各50tokens。
- `projected/vlm_proj`、`projected/aux_visual_proj`：最后生成迭代的视觉投影，保留完整tokens。
- `action_layers/block_00`、`block_11`、`block_23`：最后生成迭代、相应block输出，1×30×1024；保留全部动作tokens，未池化。层编号从0开始。
- `generation/initial_noise_raw`：原始采样噪声；`initial_noise_after_action_preprocess`是清零夹爪通道后的噪声，两者不可混淆。
- `generation/iteration_00..09`：各生成迭代的t和Transformer原始20维预测。最后迭代索引9，t=0.1。
- `actions/model_output_20d`：动作空间后处理后的模型输出；另外保存原控制动作块、重算控制动作块及实际执行部分。
- `physical`：查询时刻杯子/末端/盘子位置、末端旋转、杯子四元数、抓持、速度、误差、sim_state、qpos/qvel及动态相机位姿。
- `steps`：每一步执行后的物理状态和原执行动作。

实际输入含4个图像槽位，image_mask=[True,True,False,False]。两个空视角也在辅助序列中，投影有bias，不能假定投影后依旧全零。只池化真实腕部的前50个tokens，不把空槽位混入腕部向量。

存储：BF16按uint16原始位模式无损保存，dataset属性torch_dtype标记类型；**不能直接把uint16数值作为特征**。使用read_features.array()恢复float32，或tensor()恢复BF16。其他类型保持原精度，gzip无损压缩。

## 后续分析注意事项

- `episode_success`是整条轨迹最终结果；当前`goal`、`grasped`是当下状态，二者分开。
- `stage_heuristic`只提供placed_or_releasing、grasped_low、transport、ungrasped_approach_or_recovery粗标签；不是人工标注。未抓持可能是接近或失败后恢复，不强行标为接近。
- 完整动作/物理状态可用于离线细分首次闭合、首次抓持与释放事件。
- 文本池化为实际50token序列均值，包括保存的padding位置；并非仅有效自然语言词均值。可利用input_ids另做敏感性分析。
- `pooled_features.npz`保留主视觉/腕视觉、融合视觉/文本、投影融合/腕部，以及每个动作层的全30token均值、前5token均值、首token。
- 以轨迹ID分组划分训练和测试，不能让同轨迹相邻帧跨集合。首次推理是机器人初态匹配的位置对照，后续需按阶段匹配。
- 两次查询之间没有模型推理，因而没有对应内部特征。物理状态逐步保存。
- sim_state不包含全部控制器/包装器隐状态；任意快照续跑需额外验证。本采集从原始初始化完整重放。
- 未保存全部层/全部注意力；不需要为基础降维制造巨大注意力矩阵。

`initial_schema_trial`仅为补齐噪声字段前的开发试采集，不纳入400条交付数据。

## 用户暂停记录

已按用户要求暂停。已验证完成175/400条、1678个推理点。详见paused_checkpoint.json。未完成文件保留为.partial，不计入结果；恢复时该条从头重放，已完成轨迹跳过。运行resume.sh即可继续采集与最终汇总。
