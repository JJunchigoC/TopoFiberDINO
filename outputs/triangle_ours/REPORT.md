# TopoFiberDINO Coordinate (ours) 实验结果

数据集：训练 134，验证 39，测试 19。

所有模型完整训练 500 个 epoch；关闭早停。best.pt 为验证最优轮次，last.pt 为最后一轮；最佳轮次不是停止轮次。

DINO 系列使用官方 DINOv2 ViT-S/14 Registers 冻结骨干与轻量解码器；另有从零训练的小型 RGB U-Net 对照。无人工标签时为伪标签自训练，不是标准的有标签 few-shot 基准。

**结果尚未经真实标注验证。** 伪标签 IoU 表示与自动候选的一致性；不能当作识别准确率或与顶会论文的真实 IoU 数值比较。毫米字段为空。

| 模型 | 验证伪标签 IoU | 测试伪标签 IoU | 测试时序 MAE |
|---|---:|---:|---:|
| baseline | 0.5000 | 0.4747 | 0.0196 |
| directional | 0.4975 | 0.4902 | 0.0138 |
| TopoFiberDINO Coordinate (ours) | 0.5246 | 0.5273 | 0.0129 |
| unet_small | 0.5045 | 0.4427 | 0.0232 |
| dino_linear | 0.4820 | 0.4758 | 0.0107 |
| dino_pyramid | 0.4950 | 0.4794 | 0.0118 |

本地结构对照和消融实验，不是对原论文的完整复现。更多模型及逐帧诊断见 [交互式复核页](review.html) 和 [质量说明](quality/README.md)。

[论文格式对比图](figures/qualitative_comparison.png) · [训练曲线](figures/learning_curves.png) · [消融诊断](figures/ablation_diagnostics.png) · [模型结构](figures/architecture.png)

所选模型直接回归 A/B/C 三点坐标，再连接 AB、BC、CA 三条直边。叠加图不再显示弯曲轮廓或区域填色。`triangles/` 提供纯三角形 SVG，`masks/` 仅为同一三角形的栅格化结果。

训练使用三点伪标签、坐标损失和可微三角形栅格化损失。B/C 的自动目标仍来自最大可见展开截面，尚非已确认的物理钳口；三角形形状正确不代表三点位置准确。无效区外预测明确标记失败，不裁切成额外顶点。

7:2:1 为同视频内时间分块；相邻边界帧仍有相关性，且这些图片在以前开发中已被查看，不能宣称独立盲测。

新增质量输出：6 个模型 × 192 帧，共 1152 条逐帧诊断。真实标注评估条数：0。

[全部模型对比图](figures/all_models_comparison.png) · [优先复核帧](figures/review_priority_examples.png) · [尺寸时间曲线](figures/measurement_timeseries.png) · [逐帧CSV](quality/frame_diagnostics.csv) · [模型汇总](quality/model_summary.csv)

复核页初始模型为 `full`，按配置指定的验证指标选择，不代表真实准确率最高。尺寸仍为像素，复核优先级不是准确率。

## TopoSlide-inspired adaptation

Train-only DINO clusters, DTM cubical H0/H1 histograms, conditional histogram/CDF/proportion supervision and a shared global context branch. MRF and three-point output are retained. This is an independent ROI adaptation, not the published pathology model.

[Topology diagnostics](figures/topology_diagnostics.png) · [Descriptor metrics](topology/metrics.json) · [Method and differences](../../docs/TOPOLOGY.md)

## 500 epoch comparison

Same split, pseudo targets, epoch budget and validation vertex-error selection. No human accuracy established.

| Model | Best epoch | Val pseudo IoU | Test pseudo IoU | Test vertex error | Test flow MAE |
|---|---:|---:|---:|---:|---:|
| baseline | 34 | 0.5000 | 0.4747 | 0.0831 | 0.0196 |
| directional | 9 | 0.4975 | 0.4902 | 0.0797 | 0.0138 |
| TopoFiberDINO Coordinate (ours) | 5 | 0.5246 | 0.5273 | 0.0748 | 0.0129 |
| unet_small | 75 | 0.5045 | 0.4427 | 0.0867 | 0.0232 |
| dino_linear | 64 | 0.4820 | 0.4758 | 0.0789 | 0.0107 |
| dino_pyramid | 110 | 0.4950 | 0.4794 | 0.0821 | 0.0118 |

Ours ranks first on all reported quality metrics: False
[All metrics and ranks](comparison/study500.json) | [Every frame](comparison/study500_frames.json)
