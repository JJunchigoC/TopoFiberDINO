# TopoFiberDINO Coordinate（ours）的 TopoSlide 借鉴

参考 [TopoSlide，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.html) 的训练集聚类、DTM、持续性直方图和全局条件预测思路。已核对主论文第 3 节及[作者实现](https://github.com/plevritis-lab/TopoSlide/blob/main/train/train_toposlide4.py)。本项目仍输出纺纱 ROI 的三个顶点，不是病理 WSI 模型移植或完整复现。

## 当前实现与差异

| 环节 | 本地实现 |
|---|---|
| 特征 | 已有冻结 DINOv2 Registers，不加载 CONCH 或病理权重 |
| 聚类 | 仅 134 张训练帧的有效 patch；球面聚类，最多 16,000 个 token，25 次更新，4 簇；未采用论文 PCA 初始化 |
| 网格 | 28×12，排除与无效填充相交的池化单元 |
| DTM | 到最近 3 个簇内 patch 中心的平均距离；按工作图宽度归一化，保留宽高比 |
| 同调 | GUDHI CubicalComplex 的 H0/H1；仅有限正持续区间，空簇返回零直方图 |
| 细节差异 | 本地 H0/H1 都取前景 DTM 下水平集，没有照搬论文逆图 H0，不能称为数值等价 |
| 直方图 | 每维 8 个 bin，末 bin 包含溢出；用训练帧的逐 bin 最大值归一化，分母至少 1，验证/测试不拟合统计 |
| 全局编码 | 384→32 投影、二维位置编码、1 层 4 头自注意力及有效 token 均值池化 |
| 条件预测 | 每簇冻结特征均值与全局表示构成通道门控，空簇使用训练中心；不是作者 CLP 的完整实现 |
| 三点联系 | 全局表示通过零初始化线性层和 0.15×tanh 残差修正局部坐标 logits；不再直接拼接随机全局特征替代局部头 |

H0/H1 描述的是图像特征簇的空间分布，不是最终三角形的连通域数。三角形本来由三点构成，重复规定一个连通域且无孔并不能提供有效位置监督。簇编号也不代表已确认的纤维、钳口或背景语义。没有实现原论文局部临界点与跨切片对比损失，不复制作者代码或加载其受限权重。

## 参数和损失

当前 `configs/ours.json` 的 topology 参数：enabled=true、clusters=4、grid_size=[28,12]、embedding_dim=32、dtm_neighbors=3、weight=0.02、cdf_weight=0.5、proportion_weight=0.01、warmup_epochs=10；bins=[0,0.01,0.02,0.04,0.08,0.12,0.2,0.35,1.2]。

`L_topology = min(epoch/10,1) × [0.02 × (Huber(hist) + 0.5 × Huber(cumsum(hist)/bins)) + 0.01 × Huber(proportion)]`。Huber 使用 beta=0.1 的 Smooth L1；累计计数除以 bin 数控制量级，不是严格 Wasserstein 距离。直方图 softplus 输出、簇占比 sigmoid 输出。当前结构为 revision 5，旧 revision 3/4 权重不能混用。

```powershell
python main.py --config configs/ours.json --output outputs/triangle_ours
python main.py --config configs/ours_ablation.json --output outputs/triangle_ours_ablation
python main.py --config configs/ours.json --output outputs/triangle_ours --predict-only
```

消融配置只关闭 topology.weight 和 proportion_weight，保留同样网络、初始化、MRF 与预算，检验拓扑辅助监督而不是整个全局分支。新版消融尚未完成，不能将整套优化收益归因于拓扑。

## 输出

- topology/fitted.pt：训练集中心、归一化值、训练 ID 和配置；预测模式复用。
- topology/targets.json：192 帧簇地图、原始持续性直方图和占比。
- full/topology_predictions.json：最佳权重的描述符预测和自监督目标。
- topology/metrics.json、figures/topology_diagnostics.png/.svg：重建 MAE 和固定四帧诊断，不是顶点准确率。
- full/history.json、verification.json：实际损失、统计来源和检查点中心一致性检查。

## 历史 revision 4 消融

两组均训练 1000 轮、175,536 个参数、最佳轮均为 5。加入拓扑辅助损失后，验证区域损失由 0.560696 降至 0.551933，但测试伪 IoU 从 0.473785 降至 0.462477，伪顶点误差由 0.082641 增至 0.083640。测试描述符 MAE 由 0.670111 降至 0.290629：辅助任务学得更好，没有带来该次测试定位收益。

旧对照已压缩至 archive/history_before_study500.zip 中的 outputs/triangle_ours_ablation/，旧完整方法的精简参考在 docs/records/optimization_reference.json。当前 ours 目录替换为新版结果，当前 ours_ablation 配置也已对应新版，需重新训练后才产生可匹配的消融。真实定位和毫米精度仍需要人工三点及同平面尺度参照。
