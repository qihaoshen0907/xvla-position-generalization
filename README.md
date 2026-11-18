# XVLA 位置泛化与内部特征分析

原版 XVLA 在 LIBERO 衍生仿真场景中的位置泛化实验，以及基于已保存动作重放的内部特征分析。报告和图表以中文为主。

**范围：同一场景、机器人初态、杯子朝向和推理种子。不是完整 LIBERO 基准成绩，也不构成失败原因的因果证明。**

## 实验结果

开始前移动白杯，目标为将白杯放到左盘；场景中的其他杯子保留。原模型，无 LoRA，无抬高或回原位恢复策略；每次最多520步，30步动作块。

| 采样区域 | 次数 | 稳定成功 |
|---|---:|---:|
| 5×5 cm 内部 | 100 | 89% |
| 10×10 cm，排除内部5×5 | 100 | 36% |
| 15×15 cm，排除内部10×10 | 100 | 8% |
| 20×20 cm，排除内部15×15 | 100 | 0% |

区域中心为世界XY=(-0.10,-0.15)m。先过滤初始碰撞等无效布局，再均匀采样；未按模型成功结果筛选。成功要求连续20步目标成立、未被抓持且速度低于0.01m/s。

## 分析结果

![中文 t-SNE](analysis/mug_feature_analysis_20261118/tsne_outcome_zh.png)

- **降维**：400条轨迹首次推理，每条一个点；6组1024维特征，t-SNE和UMAP各3组参数、3个种子，共108个嵌入。没有PCA。主图参数在运行前固定；成功/失败的聚集同时与位置相关。
- **位置探针**：覆盖各区域训练、留出空间块测试时，最外环融合视觉和动作末层的平均二维误差分别为0.89cm和0.99cm。只用默认区域训练线性探针时，最外环误差分别为8.06cm和7.85cm。探针能力不等于原策略利用位置的能力。
- **动作响应**：首次闭合指令发出前，末端参考点到杯子的水平距离中位数为2.83、4.26、5.58、7.92cm。这不是指尖接触误差；首次闭合也不等于实际抓取时刻。
- **阶段分解**：未触发抓持检测的轨迹随范围扩大而增多；但其中21条仍出现杯子抬高，不能将检测标签直接当作物理抓取/掉落真值。

![中文位置探针](analysis/mug_mechanism_analysis_20261118/position_probe_zh.png)

## 报告导航

- [t-SNE / UMAP 方法、参数扫描与局限](analysis/mug_feature_analysis_20261118/README.md)
- [失败阶段、位置解码与动作响应](analysis/mug_mechanism_analysis_20261118/README.md)
- [中文失败阶段图](analysis/mug_mechanism_analysis_20261118/failure_stages_zh.png)
- [中文动作响应图](analysis/mug_mechanism_analysis_20261118/action_response_zh.png)
- [中文位置着色对照](analysis/mug_feature_analysis_20261118/tsne_outcome_position_comparison_zh.png)

## 首次夹爪闭合前后的 t-SNE

分析400条轨迹首次夹爪闭合前一次、闭合后前三次推理的腕部视觉、融合视觉和动作末层特征，共1,578个推理点。每行各时刻共享同一降维坐标；绿色为最终成功，黄色为观察窗口内出现持续抓持但最终失败，红色为窗口内无新持续抓持且最终失败。

![首次夹爪闭合前后的中文 t-SNE](analysis/mug_close_embedding_20261118/close_stages_seed0_zh.png)

闭合指首次闭合命令，不代表物理接触完成。第三次后续推理有378条轨迹；位置、当前抓持状态和任务阶段均可能影响分群，不能仅凭二维分离判断纠偏能力。

[查看完整分析报告与方法](analysis/mug_close_embedding_20261118/README.md)

## 本次公开内容

包含分析代码、中文PNG/PDF、数值汇总、探针预测、采样坐标、108组二维嵌入、运行设置，以及实验和特征采集的参考脚本。没有令牌、本机用户目录、模型权重、完整隐藏特征、原始观测图像或视频。

**完整特征数据已在 [Hugging Face 数据集](https://huggingface.co/datasets/QihaoShen/xvla-position-generalization-features) 发布。** 包含400条轨迹、5,526个推理点，完整HDF5约12.63GB、池化特征约253MB。GitHub保留代码和分析结果，大文件请从Hugging Face下载。数据含重放观测图像、动作和状态，不含模型权重或完整仿真安装资源。

```bash
hf download QihaoShen/xvla-position-generalization-features --repo-type dataset --local-dir ./xvla-features
```

发布副本的本机绝对路径已替换为可移植引用；HDF5仅修改来源路径属性，数值数组与图像不变。数据集包含文件校验清单与BF16读取工具，使用前请阅读其数据说明。

12条试采集先通过无采集/有采集动作一致性验证，再扩展到400条。所有轨迹的保存位置、速度和重算动作与原记录最大误差为0。原始历史输入图像没有无损基准可比，因此没有声称原始像素逐像素验证一致。

## 重绘公开图表

```bash
python -m venv .venv
# 激活新环境后：
python -m pip install -r requirements-analysis.txt
python analysis/mug_feature_analysis_20261118/plot_chinese.py
python analysis/mug_mechanism_analysis_20261118/plots.py
```

中文绘图使用 Noto Sans CJK。脚本默认字体路径 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`；其他系统请在绘图脚本中改为已安装的中文字体。

上述重绘使用仓库内的二维坐标、标签和统计结果，不需要完整模型或特征。代码没有重新生成模拟轨迹。

若已有完整特征数据，可设置 `XVLA_DATA_DIR` 指向包含 `pooled_features.npz`、`query_index.jsonl`、`manifest.json`、`episodes/` 的目录，再运行分析脚本。完整数据可从上述Hugging Face仓库获取；下载完成后可从特征重新拟合。

`reference/` 是原实验/采集脚本的脱敏快照，依赖额外的 LIBERO-MAX、LIBERO-plus、LeRobot、模型及本地兼容启动器；不是独立的一键仿真安装包。未打包上游软件或权重，参考脚本中的workspace/data路径为占位路径。

## 来源与版本

- [XVLA checkpoint](https://huggingface.co/lerobot/xvla-libero)，revision `12e8783e996944f5c97e490d37d4c145484ed70a`；模型卡标注Apache-2.0。
- [X-VLA](https://github.com/2toinf/X-VLA)
- [LeRobot](https://github.com/huggingface/lerobot)，本地版本 `dd5313e305ff7aaa5b6c7b4769d1b26650ce250b`
- [LIBERO-MAX](https://github.com/liberomax/LIBERO-MAX)，`01625e290b1aa07511bdc820781e635e851353cc`
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)，`4976dc30028e805ff8094b55501d532c48fec182`
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

请引用相应上游工作。项目间的许可证相互独立；本次公开不授予上游资产或模型的额外权利。详见 [使用范围](USAGE.md)。
