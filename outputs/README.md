# 当前结果

`triangle_ours/` 是唯一保留在工作目录的正式结果，包含六模型各 500 轮的权重、指标、三点预测、图表和视频。

- [逐帧复核](triangle_ours/review.html)
- [完整指标与排名](triangle_ours/comparison/study500.json)
- [输出核验](triangle_ours/verification.json)

默认训练成功后替换此目录。当前权重与图文报告在项目整理中未改动。锁文件用于并发保护，并非残留错误。

历史 `triangle_v2` 和旧 `triangle_ours_ablation` 已逐文件校验并压缩至 [历史归档](../archive/history_before_study500.zip)。新版消融仍可通过 configs/ours_ablation.json 另行训练。
