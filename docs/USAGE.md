# 使用与参数说明

当前统一预算 500 轮，默认六模型；数据、指标和结论以 [根 README](../README.md) 及 [本次协议](STUDY500.md) 为准。历史 1000 轮配置已压缩至 archive/history_before_study500.zip，仅供历史查阅。

## 常用命令

```powershell
python main.py
python main.py --config configs/ours.json --output outputs/triangle_ours
python main.py --predict-only
python main.py --config configs/ours.json --output outputs/triangle_ours --predict-only
python main.py --models baseline full
python main.py --retrain-models full
python main.py --prepare-only
```

`--models` 选择子集，仍统一 500 轮；`--epochs N` 可显式覆盖。选择性重训要求其他模型配置兼容，否则拒绝混用。输出目录必须是 outputs 的直接子目录，由项目管理并整目录替换。`--predict-only` 复用兼容 best 权重，只重建预测与图表。

## 修改 ours

修改 `configs/ours.json`，再运行带该 `--config` 的命令；不会自动同步 default.json。当前参数：

| 参数 | 值 | 作用 |
|---|---|---|
| epochs / batch_size | 500 / 8 | 固定预算，无早停 |
| learning_rate / weight_decay | 0.0003 / 0.001 | AdamW 与余弦学习率 |
| checkpoint_selection | vertex_error | 验证三点误差选权重 |
| coordinate_head | calibrated_spatial | 仅 full 使用，其他模型池化头 |
| distribution_weight | 0.6 | 独立坐标证据的边缘分布 KL；必须配 calibrated_spatial |
| coordinate_weight | 5 | 三点 Smooth L1 |
| feature_dropout | 0.15 | 特征通道屏蔽 |
| translation_weight | 0.5 | 同步平移监督 |
| translation_range | [0.02, 0.025] | 归一化水平与垂直位移 |
| bounded_translation | true | 防止平移目标超出几何范围 |
| boundary_weight / vertex_weight | 0 / 0 | 关闭预测三角形派生的重复热图项 |
| mrf.weight / color_sigma | 0.002 / 0.1 | 八邻域 Potts 正则 |
| temporal_weight | 0.02 | EMA 光流一致性 |
| topology.embedding_dim | 32 | 小型全局分支 |
| topology.weight / proportion_weight | 0.02 / 0.01 | 拓扑直方图与占比监督 |
| topology_residual_scale | 0.15 | 有界坐标残差 |

原空间头是 `spatial_integral`，校正映射头是 `calibrated_spatial`。分布监督的高斯宽度为归一化坐标 0.025；它监督独立空间证据，未使用目标坐标替换推理结果。修改头类型后必须重新训练。

## 输出与少标注入口

`review.html` 复核所有帧；各模型 `predictions.json`、`overlays/`、`triangles/` 提供原始三点和图像；`measurements.csv` 为尺寸；`comparison/study500.json` 为完整排名；`quality/` 为逐帧诊断、时序和真实标注评估入口。

在 dataset/annotations/train 放入与 ROI 同名 JSON：

```json
{"vertices_roi": [[120,180],[450,150],[450,280]]}
```

以上只是格式示例，不是实际标注。提供至少五张真实训练标注后可运行 `python main.py --config configs/ours.json --shots 5 --output outputs/triangle_fewshot_5`。验证和测试真值放到各自目录。少样本精度尚未验证；当前伪标签不计作真实样本标注。
