# 三点三角形质量说明

本次所选模型统一训练500个epoch，关闭早停。full为ours，具体版本以run.json和model_summary为准。model_summary中epochs_run/last_epoch是实际轮数，best_epoch是配置所指定验证指标的最优轮次；默认图像用best.pt，第500轮权重另存last.pt。两者不必是同一轮。

full 训练增加 8 邻域对比敏感 Potts MRF 正则，最终仍只有三个顶点。MRF 能量低不代表三点真实位置更准，平滑项还可能使三角形收缩。有兼容的旧无 MRF 参考时，comparison/mrf_ablation.json 和 figures/mrf_comparison.png 提供同预算前后比较；没有真实三点时不能据此声称实际测量精度提高。

输出契约为 A/B/C 三个坐标及 AB/BC/CA 三条直边。最终 SVG 和叠加图只画三点三边，二值 mask 是同一三角形的栅格化，尺寸只有三角形面积与周长。

打开 [三点复核页](../review.html)，对照原图检查 A 的汇聚位置、B/C 的端点定义和三条边的方向。切换模型保持当前帧，随后用连续视频和尺寸曲线查看时间变化。B/C 自动目标仍是最大展开截面代理，尚未验证为物理钳口。

## 指标如何理解

- pseudo_iou / pseudo_dice：三角形与自动三点目标的一致性，不是真实准确率。
- pseudo_vertex_error_normalized：模型三点与自动目标三点的平均欧氏距离，x/y 各自除以 ROI 宽/高后计算；不是像素或毫米误差。
- threshold_stability_iou：软三角形在阈值0.4/0.6下的重合程度。三角形结构已被强制约束，高稳定性不代表端点准确。
- foreground_boundary_entropy：软栅格化在前景及5×5边缘带中的二元熵。它主要反映边缘过渡，不能当作定位置信度。
- model_disagreement：4×前景投票率×(1−投票率)，在所有模型前景并集内平均；模型共享自动目标，错误可能相关。
- flow_warp_mae：同视频、同分组邻帧经光流对齐后的软三角形差异，无有效门控像素时为空。
- discarded_fraction / raw_components：检查阈值区域与有效掩膜相交后的连通性；新版不执行丢弃分量后处理。
- triangle_area_raw_px2：由原视频坐标三点计算的三角形面积。

复核优先级是排序启发式，不是合格分或准确率。公式：100×加权平均，0.45×(1−阈值稳定性)、0.25×模型分歧、0.20×min(时序MAE/0.1,1)、0.10×区域碎片比例。缺失项去除后归一；空预测或无效三角形为100。提示阈值分别是稳定性<0.8、分歧>0.35、MAE>0.05、碎片比例>0.1。这些权重尚未按人工误差校准。

## 真正的三点评估

在 dataset/annotations/val 或 test 放置同名 JSON：
{"vertices_roi":[[Ax,Ay],[Bx,By],[Cx,Cy]]}
使用原 ROI 像素坐标，顺序固定 A左 / B右上 / C右下。运行 python main.py --predict-only，无需重新训练即可计算三点平均欧氏误差、三角形 IoU/Dice、精确率/召回率、边界F1、ASSD和HD95。人工三角形PNG只用于兼容，必须确实是单一三角形；JSON优先。没有标注时真实指标为null。

边界容差2个原 ROI 像素，距离指标同样使用 ROI 像素。区域距离由导出的硬三角形计算；各指标按有定义的帧取宏平均，缺失值不当作0分。只有一方为空时IoU/Dice为0；两者为空为1；无边界可比较时距离为null。错误图绿色TP、红色FP、蓝色FN。

## 文件

frame_diagnostics.csv/json 为全部模型逐帧诊断；model_summary.csv/json 为模型汇总；ground_truth_metrics.json 为独立人工评估。各模型 probability/ 是工作尺寸200×448的uint16 PNG，除以65535还原；无效像素置0。entropy/ 和 disagreement/ 采用固定0–1色阶，黑紫低、橙黄高，无效像素灰色。

figures/all_models_comparison.* 为固定等间距4张测试帧；review_priority_examples.* 为诊断排序选取的8张不同测试帧，各自有选图记录，后者不是随机示例。measurement_timeseries.* 显示三角形宽度、高度和面积，分组边界用虚线。videos/ 按原30fps时间间隔输出，V01第8帧缺失以空白显示，视频从首个保留帧开始。

7:2:1来自同两段视频的时间分块，不能代表独立工况泛化。真实物理尺寸还需要标尺；当前毫米字段为空。
