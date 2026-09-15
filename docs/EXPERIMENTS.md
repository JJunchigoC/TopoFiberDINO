# 实验入口与证据

日常训练：`python main.py`；单独训练 ours：`python main.py --config configs/ours.json`。六模型默认均为 500 轮。

| 入口 | 用途 |
|---|---|
| configs/default.json | 六模型正式比较 |
| configs/ours.json | 单独训练 ours |
| configs/ours_ablation.json | 同结构关闭拓扑辅助损失，尚未完成该额外消融 |
| tools/experiments/study500.py | 四组验证候选选参，再发布六模型结果 |

复做完整研究流程：`python tools/experiments/study500.py --stage all`。会完整训练并替换当前结果；日常训练无须运行额外选参。

本次协议见 [STUDY500.md](STUDY500.md)，选参证据见 [study500_selection.json](records/study500_selection.json)，训练与报告核验见 [study500_verification.json](records/study500_verification.json)。完整候选权重和历史保留在 `.cache/study500/trials/`，供审计和重建。

旧 1000 轮配置、诊断/选参/发布脚本、固定历史比较模块、两套旧输出和旧报告素材已归档至 [history_before_study500.zip](../archive/history_before_study500.zip)。历史脚本不再作为当前入口；恢复到另一个目录后仅供历史检查，不能混入当前排名。

`records/` 中旧记录是历史事实，不代表本次验收，也不改写其中当时的路径。历史分析见 [OPTIMIZATION.md](OPTIMIZATION.md)。
