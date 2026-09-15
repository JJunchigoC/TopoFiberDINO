# 汇报生成工具

以下脚本只读取已完成的 500 轮六模型实验，不训练模型：

1. `study_figures.py` 在项目 Python 环境运行，用 matplotlib 绘制全部六模型及四张固定帧，并保存指标快照。
2. `study_report.py` 使用文档运行环境的 ReportLab，直接生成给老师的 PDF。
3. `study_docs.py` 根据完成的真实指标更新 README、使用说明与实验协议。

从项目根目录依次调用。第二步需要 ReportLab 和中文字体，使用 Codex 文档运行环境，不属于训练依赖。正式输出在 `reports/`，指标快照与插图在 `reports/assets/study500/`。改动图文后必须重新生成 PDF 并逐页检查。

本次文档运行环境为 `C:/Users/jjiang38/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。训练环境仍使用项目原来的 `python`。

历史 1000 轮报告的三份生成脚本已核对后归档至 `archive/report_1000_source.zip`，不再与当前入口混放。
