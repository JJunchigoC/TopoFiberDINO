"""Build an honest eight-page report after the complete 500-epoch comparison."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white,black
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]; ASSETS=ROOT/'reports/assets/study500'
source=json.loads((ASSETS/'source_snapshot.json').read_text(encoding='utf-8'))
study=source['study']; models=study['models']; names=list(models); selected=source['selection']['selected']
assert len(names)==6 and all(m['epochs']==500 for m in models.values())
pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\msyh.ttc'))
pdfmetrics.registerFont(TTFont('YaHeiBold',r'C:\Windows\Fonts\msyhbd.ttc'))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
styles={
 'body':ParagraphStyle('Body',fontName='YaHei',fontSize=10.6,leading=17,spaceAfter=8,wordWrap='CJK'),
 'title':ParagraphStyle('Title',fontName='YaHeiBold',fontSize=24,leading=34,spaceAfter=15,wordWrap='CJK'),
 'h1':ParagraphStyle('H1',fontName='YaHeiBold',fontSize=17,leading=25,spaceAfter=12,keepWithNext=True),
 'h2':ParagraphStyle('H2',fontName='YaHeiBold',fontSize=12,leading=18,spaceBefore=8,spaceAfter=6,keepWithNext=True),
 'caption':ParagraphStyle('Caption',fontName='YaHei',fontSize=8.5,leading=13,spaceAfter=8,wordWrap='CJK'),
}
cell=ParagraphStyle('Cell',parent=styles['body'],fontSize=9.2,leading=14,spaceAfter=0)
head=ParagraphStyle('Head',parent=cell,fontName='YaHeiBold',textColor=white)
story=[]
def p(text,style='body',raw=False): story.append(Paragraph(text if raw else escape(text).replace('\n','<br/>'),styles[style]))
def h(text): p(text,'h1')
def sub(text): p(text,'h2')
def page(): story.append(PageBreak())
def fig(name,caption,width=480):
 im=Image(str(ASSETS/(name+'.png'))); im.drawHeight=im.imageHeight*width/im.imageWidth; im.drawWidth=width
 story.extend([Spacer(1,4),im,Spacer(1,5)]); p(caption,'caption')
def table(headers,rows,widths):
 values=[[Paragraph(escape(str(v)).replace('\n','<br/>'),head) for v in headers]]
 values += [[Paragraph(escape(str(v)).replace('\n','<br/>'),cell) for v in row] for row in rows]
 t=Table(values,colWidths=widths,repeatRows=1,hAlign='CENTER')
 cmds=[('BACKGROUND',(0,0),(-1,0),HexColor('#294B63')),('GRID',(0,0),(-1,-1),.4,HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
       ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]
 for i in range(1,len(values)): cmds.append(('BACKGROUND',(0,i),(-1,i),HexColor('#F1F5F7') if i%2 else white))
 t.setStyle(TableStyle(cmds)); story.extend([t,Spacer(1,10)])
label={'baseline':'DINO base','directional':'DINO direction','full':'TopoFiberDINO\n(ours)','unet_small':'RGB U-Net','dino_linear':'DINO linear','dino_pyramid':'DINO pyramid'}
ranks=study['rankings']; iou_rank=ranks['test_pseudo_mask_iou'].index('full')+1; point_rank=ranks['test_pseudo_vertex_error_normalized'].index('full')+1
ours=models['full']; test=ours['test']
conclusion=f'本次六模型均完成 500 轮训练。ours 的测试伪 IoU 为 {test["pseudo_mask_iou"]:.4f}，排名 {iou_rank}/6；测试伪顶点误差为 {test["pseudo_vertex_error_normalized"]:.4f}，排名 {point_rank}/6。'
conclusion += ('在本次对照的两项测试任务指标上均排名第一。' if iou_rank==point_rank==1 else '当前结果未支持 ours 在全部任务指标上最优，报告保留完整对比。')
conclusion += '这些指标来自自动目标，尚不能证明真实物理定位精度。'

p('纺纱三角区三点定位\n500 轮六模型对比汇报','title')
p('面向指导教师的模型改进与实验报告','h2')
p('<b>'+conclusion+'</b>',raw=True)
h('1 研究任务与统一设置')
p('输入已处理的纺纱三角区 ROI，输出恰好 A、B、C 三点及三条直边，并由同一组三点计算边长、高度、面积、周长和顶角。A 表示左侧汇聚方向，B/C 表示右侧上、下端点。当前自动目标取可见最大展开截面，物理截面定义仍需确认。')
table(['项目','本次设置'],[
 ['数据','192 张已处理 ROI  两段相关视频'],
 ['训练 验证 测试','134 / 39 / 19  按同视频时间分块  约 7∶2∶1'],
 ['训练预算','六个模型各 500 epoch  batch size 8  各 8500 次更新'],
 ['选权重规则','统一最小化验证集归一化三点误差  无早停'],
 ['标注与单位','无人工三点真值  输出像素与平方像素  毫米未标定'],
 ['数据来源边界','原型和聚类仅训练集拟合  测试不参与本次选型'],
 ],[118,376])
p('本次使用无人工标注的伪标签自训练，以及冻结的预训练图像特征。它包含数据先验和预训练知识，不是从零开始的纯无监督学习。测试帧曾在历史开发中被查看，且与训练帧来自相同视频，因此不是独立盲测。')
p('数据字节和划分保持不变。旧 1000 轮成绩只作历史背景，不混入本次 500 轮排名；训练轮数减少后已按新预算完成训练，而非直接截取旧报告的数字。')

page(); h('2 六个对照与方法来源')
table(['模型','本地实现与角色'],[
 [label['baseline'],'冻结 DINOv2 特征与普通适配器  池化坐标头'],
 [label['directional'],'增加方向卷积  其余使用池化坐标头'],
 [label['full'],'方向特征 空间坐标头 拓扑辅助 MRF 与时序一致性'],
 [label['unet_small'],'从头训练的小型 RGB U 形网络  共享三点输出约束'],
 [label['dino_linear'],'冻结 DINO 特征的线性像素分类与池化坐标头'],
 [label['dino_pyramid'],'冻结 DINO 特征的多尺度解码与池化坐标头'],
 ],[132,362])
fig('pipeline','图 1  本地方法与验证候选  虚线为辅助监督  推理不读取自动三点目标',width=468)
p('骨干来源为 DINOv2 Registers，关联 ICLR 2024 [1]；全局拓扑监督借鉴 TopoSlide，CVPR 2026 [2]；新增坐标分布监督参考 SimCC，ECCV 2022 [3]。空间积分本身是经典方法 [4]。图中模块为本地场景适配，不是完整原论文复现。')
p('六项对照是本项目实现之间的比较，不能把小型 RGB U-Net 或 DINO 多尺度解码器称作对应顶会官方模型。引用论文说明思想来源，不将论文中的精度或图像结果作为本项目实验成绩。','caption')

page(); h('3 改进假设与验证选型')
p('旧空间头的 C 通道表示相对高度参数：热图峰值即使落在自动候选 C 的位置，经过后续参数化后也不一定落回同一位置。旧模型可以学习补偿，但这种间接表示不适合直接施加坐标分布监督。新头先求三个实际空间期望，再通过场景映射的逆变换转换为几何参数。')
p('新增监督取独立空间证据图的水平与垂直边缘分布，与自动三点周围的高斯目标计算 KL 损失。它监督网络证据，而不是把预测三角形重新画成热图后监督自身。同步平移时，图像、特征、目标和分布监督保持同一坐标系。MRF 和拓扑分支保留，但本轮不单独宣称它们带来收益。')
rows=[]
for n,t in source['selection']['trials'].items():
 rows.append([n,t['config']['distribution_weight'],t['summary']['best_epoch'],f'{t["metrics"]["pseudo_vertex_error_normalized"]:.5f}',f'{t["metrics"]["pseudo_mask_iou"]:.4f}'])
table(['候选','分布权重','最佳轮','验证三点误差 ↓','验证伪 IoU ↑'],rows,[136,69,59,122,108])
p(f'锁定方案为 {selected}。选择依据仅为验证三点误差；所有候选各完成 500 轮，试验训练器只接收 134 张训练帧和 39 张验证帧。选定 ours 的 500 轮权重直接进入最终比较，未额外累计训练。')
fig('selection','图 2  验证试验的完整曲线与最终 ours 的分组结果')
p('公平性边界：ours 使用了四次结构或权重试验，其他模型沿用统一配置，没有相同规模的超参数搜索。因此最终训练预算相同，但总开发算力不相同。本次选型不能证明全局最优或排除调参优势。','caption')

page(); h('4 六模型的统一结果')
table(['模型','最佳轮','验证伪 IoU ↑','测试伪 IoU ↑','测试三点误差 ↓','测试光流 MAE ↓'],[
 [label[n],m['best_epoch'],f'{m["val"]["pseudo_mask_iou"]:.4f}',f'{m["test"]["pseudo_mask_iou"]:.4f}',f'{m["test"]["pseudo_vertex_error_normalized"]:.4f}',f'{m["test"]["flow_warp_mae"]:.4f}'] for n,m in models.items()
 ],[111,48,83,83,88,81])
fig('metrics','图 3  全部六模型的测试指标  固定顺序  ours 仅用颜色标识  不隐藏不利结果')
p('<b>'+conclusion+'</b>',raw=True)
p('伪 IoU 按有效像素累计交集与并集；三点误差是 A/B/C 对应坐标按 ROI 宽高归一化后的平均欧氏距离。光流 MAE 衡量预测与邻帧光流变换后的差异，只是时序一致性，不是定位准确率。所有指标均由文件自动汇总。')
p(f'ours 的测试光流一致性排名 {ranks["test_flow_warp_mae"].index("full")+1}/6；本次结果不支持所有指标最优的结论。参数量和训练时间另列，不能把区域重叠、点定位、平滑程度与计算效率合称一个“准确率”。')

for group,section in [(0,'5 视频 V01 的固定帧对照'),(1,'6 视频 V02 的固定帧对照')]:
 page(); h(section)
 p('同一帧展示全部六模型，使用相同图像范围和画线规则。青色为三条直边，橙色为 A/B/C。帧来自既有固定选择清单，未按本次表现重新挑图；没有人工真值叠加，也未人工调整模型顶点。')
 for k in [group*2+1,group*2+2]: fig(f'frame_{k}',f'图 {k+3}  {source["selected_ids"][k-1]}  六模型原始预测',width=468)
 p('比较时应关注汇聚点和右侧截面端点是否稳定、是否落在需要测量的物理位置。三角形边线规则仅说明输出形式正确；反光或纤维弱边缘下的偏移需要结合人工三点才能判定。','caption')

page(); h('7 训练完成情况与计算代价')
fig('learning','图 8  六模型均完成 500 轮  最佳权重统一按验证三点误差选择')
table(['模型','实际轮数','最佳轮','可训练参数','训练耗时 秒'],[
 [label[n],m['epochs'],m['best_epoch'],f'{m["trainable_parameters"]:,}',f'{m["training_seconds"]:.1f}'] for n,m in models.items()
 ],[132,74,70,110,108])
p('每个模型的 history.json 都应含连续 1—500 轮，last.pt 为第 500 轮，best.pt 为验证三点误差最小的权重。最佳轮数较小不是提前停止。训练耗时为本机单次墙钟时间，包含训练、每轮验证及环境负载波动，不等于推理速度；未做独立速度基准。')
v=source['verification']
p(f'输出核验共 {v["quality_records"]} 条模型与帧记录，有效三角形 {v["valid_triangles"]} 个、非法三角形 {v["invalid_triangles"]} 个。数据原图哈希保持一致，三点与直边掩膜对应，毫米字段保持为空。工程核验通过不代表真实精度已经验证。')

page(); h('8 阶段结论与下一步')
p('<b>'+conclusion+'</b>',raw=True)
p('当前限制主要是无人工三点、视频数量少、历史测试已被查看和单随机种子。对 MRF、拓扑和分布监督的组合收益，需要同设置消融与独立数据验证。即使某指标领先，也不应外推为所有工况、所有指标或顶会水平的最优模型。')
p('建议先由老师确认 A/B/C 的物理定义，补充少量真实三点和独立视频；在固定保留测试上比较多随机种子的误差与波动，再决定是否保留复杂辅助分支。实际尺寸需要与纤维同平面的已知长度参照，完成标定后再与人工测量对照。')
sub('复现与查看')
p('默认六模型训练：python main.py\n单独 ours：python main.py --config configs/ours.json --output outputs/triangle_ours\n查看结果：outputs/triangle_ours/review.html\n完整排名：outputs/triangle_ours/comparison/study500.json','caption')
p('每次成功导出替换固定输出目录。本报告及指标快照保存在 reports，作为本次结果记录，不会因再次训练而自动改写。历史 1000 轮资料与本次 500 轮成绩分开保存。','caption')
sub('参考文献与证据')
refs=[
 ('[1] Darcet T 等. Vision Transformers Need Registers. ICLR 2024.','https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html'),
 ('[2] Abousamra S 等. TopoSlide: Topologically-Informed Histopathology Whole Slide Image Representation Learning. CVPR 2026.','https://openaccess.thecvf.com/content/CVPR2026/html/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.html'),
 ('[3] Li Y 等. SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation. ECCV 2022.','https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660088.pdf'),
 ('[4] Sun X 等. Integral Human Pose Regression. ECCV 2018.','https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.php'),
 ]
for text,url in refs: p(escape(text)+f' <link href="{url}" color="#245B75">论文来源</link>','caption',True)
p('本地证据：selection_protocol.json、各模型 history.json、verification.json、comparison/study500.json。报告快照：reports/assets/study500/source_snapshot.json。所有数字和配图来自本次真实实验，未引用他人论文成绩冒充本地结果。','caption')

def header(canvas,doc):
 canvas.saveState(); canvas.setFont('YaHei',8); canvas.setFillColor(black)
 canvas.drawString(52,766,'纺纱三角区三点定位    500 轮六模型实验')
 canvas.drawRightString(560,26,f'2026年9月15日    第 {doc.page} 页'); canvas.restoreState()
path=ROOT/'reports/纺纱三角区三点定位阶段汇报.pdf'
SimpleDocTemplate(str(path),pagesize=letter,leftMargin=52,rightMargin=52,topMargin=48,bottomMargin=44,
 title='纺纱三角区三点定位 500轮六模型对比汇报',author='').build(story,onFirstPage=header,onLaterPages=header)
print(path)
