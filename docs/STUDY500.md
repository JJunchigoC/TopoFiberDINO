# 500 轮六模型实验协议

ours 的测试伪 IoU 排名 1/6，测试伪顶点误差排名 1/6。两项测试任务指标均排名第一，但不代表真实精度或全部工况最优。测试光流一致性排名 3/6，未达到所有指标最优。

| 模型 | 最佳轮 | 验证伪 IoU ↑ | 测试伪 IoU ↑ | 测试伪顶点误差 ↓ | 测试光流 MAE ↓ |
|---|---:|---:|---:|---:|---:|
| DINO base | 34 | 0.5000 | 0.4747 | 0.0831 | 0.0196 |
| DINO direction | 9 | 0.4975 | 0.4902 | 0.0797 | 0.0138 |
| TopoFiberDINO（ours） | 5 | 0.5246 | 0.5273 | 0.0748 | 0.0129 |
| Small RGB U-Net | 75 | 0.5045 | 0.4427 | 0.0867 | 0.0232 |
| DINO linear | 64 | 0.4820 | 0.4758 | 0.0789 | 0.0107 |
| DINO pyramid | 110 | 0.4950 | 0.4794 | 0.0821 | 0.0118 |

## 选择过程

四个 ours 候选各训练 500 轮：old_spatial、corrected、distribution015、distribution060。全部只将训练与验证的 173 张帧传给试验训练器，按验证伪顶点误差选择，当前锁定 `distribution060`。其完整 500 轮权重直接用于最终评分；其他五模型各训练 500 轮。

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
