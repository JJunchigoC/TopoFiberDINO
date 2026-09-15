# 统一纺纱三角区数据集

原 `data/roi/V01`、`data/roi/V02` 合并，共 192 张，图片字节保持不变。新版目标是 A/B/C 三个点及三条直边。

- `images/`：原 ROI。
- `valid_masks/`：255 有效、0 无效填充，不是三角形标注。
- `metadata.json`：来源、帧号、时间戳、坐标变换与哈希。
- `splits/` 与 `splits.json`：训练 134、验证 39、测试 19。
- `annotations/train/`、`val/`、`test/`：真实三点标注，当前为空。

人工标注优先使用同名 JSON：`{"vertices_roi":[[Ax,Ay],[Bx,By],[Cx,Cy]]}`，坐标为原 ROI 像素。A 为左汇聚点，B/C 为右上/右下点，按原分组放置，不跨组复制。三点必须在有效 ROI 内且不共线。同名 JSON 优先于 PNG；旧 PNG 只接受同尺寸 0/255 的单一三角形，不接受弯曲包络。

默认无人工标签，三点自动目标存放在输出目录的 `pseudo_vertices.json`，不可冒充人工真值。少标注训练及评估命令见 [主 README](../README.md)。

7:2:1 按视频内时间分块，存在同视频相关性，不是独立视频盲测。训练不需解压 `archive/legacy_preprocessing_and_v1.zip`。
