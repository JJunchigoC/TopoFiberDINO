# 历史资料归档

`legacy_preprocessing_and_v1.zip` 包含重构前的 1607 个文件，包括原图、预处理产物、上一轮训练、报告和旧 README。压缩后逐文件 SHA256 校验通过，随后清除了工作目录内对应的旧副本。

恢复时将此 ZIP 解压到一个新的目录，以保留当前训练项目结构。当前程序只使用 `dataset`、`spinning`、`configs` 和官方模型资产。

`cleanup_plan.json` 保留归档条目及哈希；`cleanup_result.json` 记录实际移除的路径。旧预测中可由权重重算的大型中间张量没有额外保留副本。

## 500 轮项目整理

`history_before_study500.zip` 集中保存两套旧模型输出、旧实验配置和脚本、旧报告素材、旧优化缓存及上一轮 ours。逐文件 SHA256 校验通过后，已从工作目录删除原副本。ZIP 保留原相对路径，解压到独立目录查看；当前训练不读取归档。

`report_1000_source.zip` 保存上一版报告生成脚本。最新数据集、六模型结果和正式 PDF 保持原位置；本次四组验证试验仍保留在 `.cache/study500/`。

整理清单及保护文件哈希见 `docs/records/project_cleanup_manifest.json`，核验结果见 `docs/records/project_cleanup_verification.json`。
