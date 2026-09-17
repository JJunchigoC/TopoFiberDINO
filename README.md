
# 纺纱三角区三点定位 (TopoFiberDINO)

> **无人工标注学习框架**：基于深度特征融合与拓扑几何先验，实现纺纱三角区 A、B、C 三点定位、三条直边拟合与像素尺寸测量。

> [!WARNING]
> **科学边界与适用范围声明**
> * **非绝对物理精度**：目前指标均为**像素级伪指标**（基于算法自动生成的伪标签评估），不代表真实物理测量精度或毫米级精度。
> * **算力预算差异**：`ours` 模型经过了 4 组候选调参选型（消耗额外开发算力），而对比模型未进行同等规模调参。
> * **未达全指标最优**：`ours` 在测试伪 IoU 与顶点误差上排名 **1/6**，但在测试光流一致性（MAE）上排名 **3/6**。
> 
> 

---

## 📌 核心结论与实测对比

所有模型均在统一数据集（192 张 ROI，时间分块 134/39/19）上按 **500 轮（8,500 次更新）** 完成训练，无早停。

| 模型名称 | 最佳轮次 (Best Epoch) | 验证伪 IoU ↑ | 测试伪 IoU ↑ | 测试伪顶点误差 ↓ | 测试光流 MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| DINO base | 34 | 0.5000 | 0.4747 | 0.0831 | 0.0196 |
| DINO direction | 9 | 0.4975 | 0.4902 | 0.0797 | 0.0138 |
| **TopoFiberDINO (ours)** | **5** | **0.5246** | **0.5273** | **0.0748** | 0.0129 |
| Small RGB U-Net | 75 | 0.5045 | 0.4427 | 0.0867 | 0.0232 |
| DINO linear | 64 | 0.4820 | 0.4758 | 0.0789 | **0.0107** |
| DINO pyramid | 110 | 0.4950 | 0.4794 | 0.0821 | 0.0118 |

> [!NOTE]
> * **伪 IoU**：与自动生成的三角形目标区域的像素交并比。
> * **伪顶点误差**：按 ROI 宽高归一化后的三点平均欧氏距离。
> * **光流 MAE**：衡量时序预测的平滑一致性。
> 
> 

---

## 🚀 快速开始

### 1. 环境准备与核验

```bash
# 安装依赖环境
python -m pip install -r requirements.txt

# 运行单元测试核验实现
python -m unittest discover -s tests -v

```

### 2. 一键运行与常用命令

你可以通过双击 `run.bat` 或在终端运行以下命令：

```powershell
# 训练/预测当前全部六个模型 (各 500 轮)
python main.py

# 仅训练/更新 ours 模型 (需先配置 configs/ours.json)
python main.py --config configs/ours.json --output outputs/triangle_ours

# 重新导出预测结果 (不触发重训)
python main.py --predict-only

# 仅检查并准备数据集
python main.py --prepare-only

```

> [!TIP]
> **目录替换与保护机制**：
> 训练成功后将原子替换 `outputs/triangle_ours/` 目录；若失败则保留上次完整结果。系统内置锁文件，支持并发保护。`best.pt` 依据验证集三点误差自动挑选。

---

## 🔗 报告与数据导航

* 🔍 **交互式可视化**：[逐帧复核网页 HTML](https://www.google.com/search?q=outputs/triangle_ours/review.html)
* 🖼️ **全模型对比**：[固定帧效果对比图](https://www.google.com/search?q=outputs/triangle_ours/figures/all_models_comparison.png)
* 📄 **指标数据**：[完整指标 JSON](https://www.google.com/search?q=outputs/triangle_ours/comparison/study500.json) | [逐帧误差数据 JSON](https://www.google.com/search?q=outputs/triangle_ours/comparison/study500_frames.json)
* 📝 **详细指南**：[使用说明 docs/USAGE.md](https://www.google.com/search?q=docs/USAGE.md) | [模型卡片 MODEL_CARD](https://www.google.com/search?q=docs/MODEL_CARD.md) | [实验协议 STUDY500](https://www.google.com/search?q=docs/STUDY500.md)

---

## 🗂️ 项目目录结构

```text
├── main.py / run.bat         # 主程序入口 / 一键运行脚本
├── configs/                  # 当前 500 轮训练配置文件
├── spinning/                 # 核心模块（模型架构、训练器、几何测量、数据导出）
├── dataset/                  # 唯一正式数据集（192 张 ROI，划分 134/39/19）
├── outputs/triangle_ours/    # 本次 500 轮六模型输出结果
├── reports/                  # 正式汇报 PDF 与真实数据可视化插图
├── docs/                     # 文档说明（使用指南、模型卡片、实验协议等）
│   └── records/              # 诊断、选型和核验记录
├── tools/                    # 工具集（实验选参工具、汇报自动生成器）
├── tests/                    # 单元测试与实现核验
├── weights/ third_party/     # 官方预训练骨干网络及第三方权重
└── .cache/study500/          # 验证试验、日志与拟合状态缓存

```

---

## ⚠️ 实验数据与局限性说明

1. **样本独立性**：192 张 ROI 选自 2 段视频。19 张测试帧虽未直接参与训练器拟合，但在历史开发迭代中已被查看过，**非严格独立视频测试**。
2. **物理标定缺失**：目前缺乏同平面长度参照物，无法将像素转换为实际物理毫米（`mm` 字段当前为 `null`）。
3. **几何截面定义**：自动找出的 B/C 端点最大展开截面是否与实际物理测量截面完全一致，仍需后续人工交互核对确认。
