"""Refresh current documentation from completed 500-epoch evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
load=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
study=load('outputs/triangle_ours/comparison/study500.json'); cfg=study['config']; models=study['models']
selection=load('outputs/triangle_ours/selection_protocol.json')
assert len(models)==6 and cfg['epochs']==500
short={'baseline':'DINO base','directional':'DINO direction','full':'TopoFiberDINO（ours）','unet_small':'Small RGB U-Net','dino_linear':'DINO linear','dino_pyramid':'DINO pyramid'}
lines=['| 模型 | 最佳轮 | 验证伪 IoU ↑ | 测试伪 IoU ↑ | 测试伪顶点误差 ↓ | 测试光流 MAE ↓ |','|---|---:|---:|---:|---:|---:|']
for n,m in models.items(): lines.append(f'| {short[n]} | {m["best_epoch"]} | {m["val"]["pseudo_mask_iou"]:.4f} | {m["test"]["pseudo_mask_iou"]:.4f} | {m["test"]["pseudo_vertex_error_normalized"]:.4f} | {m["test"]["flow_warp_mae"]:.4f} |')
result_table='\n'.join(lines)
iou_rank=study['rankings']['test_pseudo_mask_iou'].index('full')+1
point_rank=study['rankings']['test_pseudo_vertex_error_normalized'].index('full')+1
conclusion=f'ours 的测试伪 IoU 排名 {iou_rank}/6，测试伪顶点误差排名 {point_rank}/6。'
conclusion+='两项测试任务指标均排名第一，但不代表真实精度或全部工况最优。' if iou_rank==point_rank==1 else '当前实测不支持 ours 在全部任务指标上最优，保留完整不利结果。'
conclusion+=f'测试光流一致性排名 {study["rankings"]["test_flow_warp_mae"].index("full")+1}/6，未达到所有指标最优。'
README=f'''# 纺纱三角区三点定位

输入已处理 ROI，输出恰好 **A、B、C 三点、三条直边与像素尺寸**。现有六模型全部按 **500 轮**完成训练，无早停。方法为 {cfg['name']}，使用自动三点伪标签，没有人工定位真值。

**{conclusion}**

## 查看报告与结果

- [给老师的图文汇报 PDF](reports/纺纱三角区三点定位阶段汇报.pdf)
- [交互式逐帧复核](outputs/triangle_ours/review.html)
- [全部六模型固定帧对比](outputs/triangle_ours/figures/all_models_comparison.png)
- [本次完整指标与排名](outputs/triangle_ours/comparison/study500.json)
- [每一帧三点误差](outputs/triangle_ours/comparison/study500_frames.json)
- [参数与详细命令](docs/USAGE.md) · [模型说明](docs/MODEL_CARD.md) · [实验协议](docs/STUDY500.md)

## 一键训练

在项目根目录运行，或双击 `run.bat`：

```powershell
# baseline directional full unet_small dino_linear dino_pyramid，各 500 轮
python main.py

# 单独训练 ours：先修改 configs/ours.json
python main.py --config configs/ours.json --output outputs/triangle_ours

# 对当前六模型权重重新导出，不训练
python main.py --predict-only

# 上次只训练 ours 时，使用同一配置重新导出
python main.py --config configs/ours.json --output outputs/triangle_ours --predict-only

# 仅检查数据
python main.py --prepare-only
```

成功生成并核验后，整个 `outputs/triangle_ours/` 原子替换，失败保留上次完整结果。单独训练 ours 会替换该目录上一轮的其他模型结果。锁文件用于并发保护。

`last.pt` 是第 500 轮，`best.pt` 按验证三点误差选择；最佳轮较早不代表提前停止。当前 batch size 8、训练帧 134，每模型 8500 次更新。修改参数后从头训练；旧 1000 轮权重与本次不同预算配置不兼容。

## 六模型实测

{result_table}

所有分数都来自当前 500 轮结果。伪 IoU 是与自动三角形目标的像素交并比，三点误差为按 ROI 宽高归一化后的平均距离；光流 MAE 只是时序一致性。它们不是人工准确率或毫米误差。

选定候选为 `{selection['selected']}`，坐标头 `{cfg['coordinate_head']}`，分布监督权重 `{cfg.get('distribution_weight',0)}`。ours 通过四个各 500 轮的验证候选选型，所选一次计入最终六模型，其他三次为额外开发成本；控制模型没有同等规模调参。最终轮数相同不等于总搜索算力相同。失败候选与完整选择依据保存在 [验证记录](docs/records/study500_selection.json)。

## 数据与科学边界

192 张 ROI 来自两段视频，时间分块为 134/39/19，约 7:2:1。训练集拟合原型与拓扑聚类。19 张测试帧没有传入本次试验训练器，但在历史开发中已经被查看，也不是独立视频测试。

尚无人工三点，B/C 的自动最大展开截面是否对应物理测量截面仍需确认。缺少同平面长度参照，毫米字段为 null。不能把规整三角形、论文风格插图或某项伪指标领先当作真实测量准确。

## 项目导航

```text
main.py / run.bat        当前六模型入口
configs/                当前 500 轮配置
spinning/               模型、训练、测量、导出
dataset/                唯一正式数据集
outputs/triangle_ours/  本次 500 轮六模型结果
reports/                正式 PDF 与真实数据插图
docs/                   使用、模型和实验说明
docs/records/           诊断、选型和核验记录
tools/experiments/      当前 500 轮验证选参工具
tools/report/           汇报生成工具
tests/                  实现核验
weights/ third_party/   官方骨干及来源
.cache/study500/        本次验证试验、日志与拟合状态
archive/                历史实验压缩归档，不参与当前运行
```

环境安装：`python -m pip install -r requirements.txt`。核验：`python -m unittest discover -s tests -v`。报告不会因再次训练自动更新；重训后需重新生成报告。
'''
(ROOT/'README.md').write_text(README,encoding='utf-8')
manual=f'''# 使用与参数说明

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
| coordinate_head | {cfg['coordinate_head']} | 仅 full 使用，其他模型池化头 |
| distribution_weight | {cfg.get('distribution_weight',0)} | 独立坐标证据的边缘分布 KL；必须配 calibrated_spatial |
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
{{"vertices_roi": [[120,180],[450,150],[450,280]]}}
```

以上只是格式示例，不是实际标注。提供至少五张真实训练标注后可运行 `python main.py --config configs/ours.json --shots 5 --output outputs/triangle_fewshot_5`。验证和测试真值放到各自目录。少样本精度尚未验证；当前伪标签不计作真实样本标注。
'''
(ROOT/'docs/USAGE.md').write_text(manual,encoding='utf-8')
protocol=f'''# 500 轮六模型实验协议

{conclusion}

{result_table}

## 选择过程

四个 ours 候选各训练 500 轮：old_spatial、corrected、distribution015、distribution060。全部只将训练与验证的 173 张帧传给试验训练器，按验证伪顶点误差选择，当前锁定 `{selection['selected']}`。其完整 500 轮权重直接用于最终评分；其他五模型各训练 500 轮。

ours 使用了额外验证搜索，不能宣称所有模型总开发预算相同。测试在历史开发中已被查看，当前只有两段相关视频和一个种子，不能宣称独立盲测或统计显著优势。旧 1000 轮结果不进入本次排名。

## 改进的实际含义

新坐标头把空间期望解释为实际 A/B/C，再经过场景几何逆映射。旧头可学习相对高度补偿，因此映射诊断不是旧预测误差的直接测量。新分布监督借鉴 [SimCC，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660088.pdf) 的坐标分类思想，使用本地联合空间图的轴边缘分布 KL，并非官方 SimCC 复现。空间积分参考 ECCV 2018，DINOv2 Registers 关联 ICLR 2024，拓扑参考 CVPR 2026 TopoSlide。未使用论文精度或图像冒充本项目结果。

MRF 与 TopoSlide 分支保留；它们的独立收益没有由本轮组合试验证明。新版纯拓扑损失消融配置为 500 轮，但尚未完成该额外消融。旧消融已移入 archive/history_before_study500.zip，不能混入当前排名。

## 复做

```powershell
# 四组验证试验，再完成固定六模型对照
python tools/experiments/study500.py --stage all

# 或分开执行；publish 会核对选型与权重哈希
python tools/experiments/study500.py --stage select
python tools/experiments/study500.py --stage publish
```

日常只需 `python main.py`，不必重复搜索。查看 [选型记录](records/study500_selection.json)、[坐标映射诊断](records/coordinate_mapping_diagnosis.json) 和 [全部指标](../outputs/triangle_ours/comparison/study500.json)。验证试验中间权重在 .cache/study500/trials，上一轮 ours 的权重证据在 .cache/study500/previous_ours。
'''
(ROOT/'docs/STUDY500.md').write_text(protocol,encoding='utf-8')
print('Current README, usage and study protocol refreshed from actual completed metrics.')
