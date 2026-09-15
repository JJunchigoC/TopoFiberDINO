# TopoFiberDINO Coordinate（ours）模型说明

当前优化使用 `architecture_revision=5`，输入已处理 ROI，输出恰好 A/B/C 三点及 AB、BC、CA 三条直边。掩膜、边长、面积均由同一组三点生成。版本 3/4 的加载分支保留用于复核历史实验，检查点不能混用。

## 诊断与几何修正

旧坐标头强制 `By <= Ay <= Cy` 且 `Bx/Cx >= 0.45`。训练集中共 42/134 张、验证集中 12/39 张自动目标因此不可表示。新版能表示全部训练/验证自动目标，证据见 [optimization_diagnosis.json](records/optimization_diagnosis.json)。可表示不代表自动目标就是正确的物理端点。

归一化参数 `q ∈ [0,1]^6` 生成：

```text
Ax = 0.10 + 0.25 q0       base = 0.37 + 0.41 q1
By = 0.24 + 0.34 q3       Cy = By + 0.06 + (0.84 - By - 0.06) q4
Ay = 0.26 + 0.34 q5
gap = base - Ax           height = Cy - By
bound = min(0.055, 0.5 gap height / (abs(By + Cy - 2 Ay) + height))
tilt = (2 q2 - 1) bound
Bx = base + tilt         Cx = base - tilt
```

A 的高度独立，底边可倾斜；有向面积严格为正，A 在左、B 在 C 上方。仍是本批已配准场景的先验，不能直接处理任意旋转或大幅倾斜底边。

## 空间定位与拓扑分支

当前选定 `coordinate_head=calibrated_spatial`，在 112×50 网格中预测三个独立空间证据图，积分得到实际 A/B/C 坐标，再通过 `scene_logits_from_points` 逆映射为六个场景参数。旧 `spatial_integral` 将 C 通道当成相对高度证据，热图上的 C 位置与输出 C 位置不一定相同；它可学习补偿，不意味着旧预测全都错误。新头使空间证据与坐标含义一致，便于独立分布监督。

当前分布监督权重 `distribution_weight=0.6`，分别比较空间图的水平、垂直边缘分布与三点高斯目标，使用 KL 损失；归一化高斯宽度为 0.025。目标截断在同一固定场景支持区。该设计借鉴 SimCC 的坐标分类思想，但从联合图取边缘分布，且继续使用几何约束与空间期望，不是官方 SimCC 复现。监督来自自动三点，推理不读取目标坐标。`calibrated` 仅表示坐标映射一致，不表示输出概率经过校准。

四个候选各完成 500 轮，按验证三点误差选定该方案；验证选择与控制组公平性边界见 [STUDY500.md](STUDY500.md)。新逆映射及分布监督有往返、梯度和推理隔离测试。

控制模型保持池化坐标头；revision 5 共享几何约束、特征 dropout 和平移训练设置。`coordinate_head` 仅控制 full。方向适配器、空间积分和拓扑分支都是 ours 的结构差异，不能将整套改进归因于单一模块。

拓扑分支仅使用训练帧的有效 DINO patch 拟合四个簇，DTM 持续同调产生条件 H0/H1 直方图和簇占比。32 维单层 Transformer 表示经零初始化线性层及 `0.15*tanh`，小幅修正局部坐标参数的 logits。初始化时不让随机全局表示扰动局部定位；残差开始学习后，坐标梯度也进入全局分支。拓扑辅助任务从第一轮起参与训练。详细差异见 [TOPOLOGY.md](TOPOLOGY.md)。

## 训练设置

- 冻结官方 DINOv2-S/14 Registers，输入 840×378；工作图 448×200。RGB U-Net 不读取 DINO 特征，但共享自动目标与置信度流程。
- 区域 BCE+Dice、坐标 Smooth L1（权重 5、beta=0.05）。关闭重复的几何边界/热图 BCE；它们原本也由预测三点派生，并非独立观测。
- DINO 通道 dropout 0.15；额外平移监督权重 0.5，水平/垂直最大位移 0.02/0.025。图像、缓存特征、有效区、目标、置信度与坐标同步变换。最终 `bounded_translation=true` 按目标限制平移，避免再次超出几何范围。这是缓存特征空间增强，不等同于重新提取变换图像的 DINO 特征。
- 有效像素内 8 邻域 Potts MRF，颜色 sigma=0.1、权重 0.002；EMA 光流权重 0.02。两者在原始批次计算，不错误复用到变换后的样本。MRF 仍可能产生收缩偏置。
- AdamW 学习率 0.0003、weight decay 0.001，余弦调度；每个模型 500 epoch、batch 8，共 8,500 次更新。无早停，best.pt 按验证归一化三点误差选择（checkpoint_selection=vertex_error），last.pt 是第 500 轮。
- 关闭的边界/热图辅助项不再分配无用稠密图或计算无效 BCE。

## 来源与限制

- [Vision Transformers Need Registers，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html)；[官方 DINOv2](https://github.com/facebookresearch/dinov2)。本地来源和哈希见 weights/provenance.json。
- [TopoSlide，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.html)：借鉴全局条件拓扑学习，未移植病理模型。
- [SimCC，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660088.pdf)：坐标分布监督的来源，非官方模型复现。
- [Integral Human Pose Regression，ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.php)：空间分布积分到连续坐标的经典依据，不冒充近年算法。
- [On Regularized Losses，ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Meng_Tang_On_Regularized_Losses_ECCV_2018_paper.pdf)：空间正则进入训练损失的依据。

本地组合是研究原型，没有顶会性能或论文级新颖性证明。192 帧来自两段相关视频，按时间分为 134/39/19，并非独立视频盲测。自动 B/C 取自可见最大展开截面，尚未确认是物理钳口。无人工三点，无真实定位精度结论；无同平面标尺，毫米字段保持 null。少标注入口可用，效果仍需真实支持集验证。
