# CUB 鸟类识别基础项目

所有文件、运行环境、下载缓存、数据和模型保存在本目录。已有 pro_proposal.docx 保留不变。

## 运行

重新完整训练：双击 Start_Training.bat。若 baseline_full 已存在，会创建带时间戳的新实验目录，不覆盖已有结果。继续最近一次基础训练：双击 Resume_Training.bat。训练已完成 10 轮时，继续操作不会额外增加轮数；如需延长训练，请用 --resume --epochs 20。

PowerShell 命令（在本目录执行）：

    .\.venv\Scripts\python.exe train.py --run runs/baseline_full
    .\.venv\Scripts\python.exe train.py --resume --run runs/baseline_full
    .\.venv\Scripts\python.exe train.py --crop --margin 0.1 --run runs/crop_margin010

仅在最终方法确定后评估官方测试集：

    .\.venv\Scripts\python.exe train.py --evaluate --run runs/baseline_full

训练默认 10 epochs，前 2 轮冻结主干、只训练分类层，其后微调全部参数。ResNet-18 ImageNet 预训练；输入 224×224；batch size 32；AdamW；主干学习率 0.0001，分类层 0.001；混合精度。显存不足可用 --batch-size 16，数据加载遇到问题可用 --workers 0。Ctrl+C 停止后，可以从最后一个完整 epoch 继续。继续训练不保证与不中断运行逐步一致，因为数据加载器随机状态会重置。

## 数据与评估

官方训练集 5994 张按类别划分为 4794 张训练、1200 张验证，随机种子 42；官方测试集 5794 张留作最终评估。splits.json 保存实际图片 ID，environment.json 保存划分哈希与环境信息。整图和裁剪实验使用同一划分。裁剪依赖真实边界框，属于额外标注条件，不能宣称是无需定位标注的自动识别改进。--margin 0.1 表示边界框的每一侧扩展原宽/高的 10%。训练与验证增强不同，但两个实验的增强规则相同。

数据集官方 MD5: 97eceeb196236b17998738112f37df78。prepare_data.py 校验压缩包、解压并逐张验证 11788 张图片。

## 输出

runs/实验名/ 中：config.json 参数；environment.json 环境和划分；splits.json 图片划分；history.csv 每轮 loss、Top-1、Top-5、Macro-F1 和时间；learning_curves.png 曲线；best.pt 验证准确率最佳模型；last.pt 用于断点继续；completed.json 训练完成记录。显式测试后另外生成 test_metrics.json、test_confusion_matrix.csv、test_predictions.csv。分类编号为 0 到 199。

smoke 实验只使用极少量图片检查流程，其指标不能作为正式项目结果。

## 来源与限制

数据：https://www.vision.caltech.edu/datasets/cub_200_2011/
下载：https://data.caltech.edu/records/65de6-vp158
引用：Wah, Branson, Welinder, Perona, Belongie. The Caltech-UCSD Birds-200-2011 Dataset. CNS-TR-2011-001, 2011.
模型：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
ResNet：He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
迁移学习参考：https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

数据限非商业研究和教育使用。CUB 官方指出与 ImageNet 可能存在图片重叠，报告必须说明预训练数据重叠限制。不要将其他数据集和 CUB 随意混合。模型文件仅在本地保存；按课程要求不要把 checkpoint 放入最终补充材料。项目脚本由 Codex 生成，请按课程 AI 使用政策披露并自行理解、核验。

## 本机已验证结果（2026-11-03）

Python 3.14.6；PyTorch 2.11.0+cu128；torchvision 0.26.0+cu128；CUDA 12.8；RTX 5070 12GB。已验证 sm_120 支持、真实 GPU 前向/反向计算、依赖完整性、断点继续、裁剪训练和单图预测。

200 类整图正式基准已完成 10 轮，最佳为第 10 轮：验证集 Top-1 71.50%，Top-5 91.67%，Macro-F1 0.7093。官方测试集未进行指标评估。结果来自单个随机种子，属于初步基准。训练准确率约 99.81%，高于验证准确率，后续可以用增强、正则化和裁剪对照研究泛化。

最佳模型：runs/baseline_full/best.pt
结果摘要：runs/baseline_full/validation_summary.json
曲线：runs/baseline_full/learning_curves.png
逐轮结果：runs/baseline_full/history.csv
首次正式训练日志：runs/baseline_train.log
数据样例图：data/dataset_preview.png
数据完整性：data/verification.json
硬件验证：gpu_check.json
预测示例：runs/baseline_full/prediction_example.json

识别自己的图片（整图模型，默认读取首次正式训练的 best.pt）：

    .\.venv\Scripts\python.exe predict.py "你的图片完整路径.jpg"

模型输出的 score 为 softmax 分数，未做概率校准。项目只识别训练中定义的 200 个类别。

主下载来源为 CaltechDATA。由于连接缓慢，实际图片包使用 https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz 镜像下载；其 MD5 与官方完全一致，并已逐张验证全部图片。download_files.py 支持分段与续传；prepare_data.py 负责最终校验。镜像模式先生成 data/CUB_mirror.tgz，校验解压前需要将其改名为 data/CUB_200_2011.tgz；本次已完成这些步骤，无需重复下载。

## 整图与裁剪配对实验

双击 Run_Comparison.bat 可运行/继续实验。已经完成的训练会跳过，最后重新验证并生成分析；不要在已有对照训练尚未结束时重复启动。

设计：训练随机种子 42、43、44，每个种子分别训练整图和鸟区域裁剪模型；数据划分统一固定 split-seed=42。训练 10 轮，前 2 轮冻结主干，其他设置一致。裁剪边界框的每一侧扩展原宽/高的 10%。首次整图 seed42 使用 runs/baseline_full，crop42 使用 runs/crop_margin010，其余目录为 runs/comparison_模式_seed种子。

train.py 现在将 --seed（训练随机性）与 --split-seed（数据划分，默认42）分开；兼容原先未记录 split_seed 的 checkpoint。旧模型和原始结果没有改动。

compare_experiments.py 会检查 6 个实验的数据划分哈希、验证图片 ID 和受控参数一致，重新加载每次验证 Top-1 最佳模型，仅在同一验证集上评估。结果保存在 runs/comparison：

- comparison_report.txt：中文实验报告与限制。
- summary.json / validation_results.csv：汇总与逐次指标。
- paired_differences.csv：每组提升幅度、纠正和新增错误数量。
- bird_size_analysis.csv：边界框占图比例分组比较。
- per_class_results.csv：类别级结果（每类仅6张验证图片，需谨慎解释）。
- validation_predictions.csv：可复核的逐图预测。
- performance_comparison.png：三次训练均值与样本标准差。
- validation_curves.png：训练过程中的验证准确率。
- bird_size_comparison.png：不同目标大小下的表现。
- qualitative_examples.png：种子42的改善与退步案例，按结果选择，用于解释。

误差条和阴影代表训练种子间样本标准差，不是置信区间。裁剪依赖额外真实位置标注，并同时改变目标尺度和背景，因此不能单凭结果推断背景的因果作用。官方测试集继续留作最终评估。

本次配对实验结果（2026-11-03）：整图 Top-1 70.31% ± 1.13%，裁剪 67.44% ± 0.05%（3 个训练种子的均值 ± 样本标准差）。整图平均高 2.86 个百分点；三个种子方向一致。边界框占图不足 10% 的 33 张图片中，裁剪平均高 14.14 个百分点，属于小样本探索结果。官方测试集未评估。

案例复核提示，Resize(256)+CenterCrop(224) 对狭长鸟区域可能截去头、尾。后续可以单独比较等比例缩放和填充方式；本次裁剪结果不能视为对所有裁剪方法的结论，也不能单独判断背景作用。

## 完整鸟区域的等比例缩放填充实验

运行 Run_Letterbox.bat（训练进行中不要重复启动）。新结果保存在 runs/letterbox_seed42、letterbox_seed43、letterbox_seed44；三方案对比保存在 runs/letterbox_comparison。旧整图和中心裁剪模型及结果保持不变。

新命令示例：

    .\.venv\Scripts\python.exe train.py --crop --resize-mode letterbox --seed 42 --split-seed 42 --run runs/letterbox_seed42

保持相同的标注边界框及每侧10%扩展。最长边等比例缩放至224，居中填充到224×224，填充色RGB=(124,116,104)，接近预训练归一化均值。训练和验证都不再裁掉区域；训练仅保留水平翻转，因此同时取消了随机缩放裁剪。网络、训练轮数、冻结轮数、学习率、batch size、随机种子与固定数据划分不变。

这是一项预处理方案比较：训练增强和验证几何处理同时变化，不能单独把差异解释为填充带来的因果效果。新方案使用真实鸟框这一额外监督条件。最长边缩放可能让狭长目标占用较少的有效像素，不能预先保证优于中心裁剪。

letterbox_checks.json 保存了狭长输入两端保留检查和16项旧输入逐像素一致检查结果。更新前 train.py 快照保存在 cache/source_snapshots/train_before_letterbox.py。旧 checkpoint 未记录 resize_mode 时仍按 legacy 行为读取。

三方案汇总文件为 experiment_report.txt、summary.json、validation_results.csv、validation_predictions.csv、paired_differences.csv、bird_size_analysis.csv、performance_comparison.png、bird_size_comparison.png、validation_curves.png、qualitative_examples.png。旧两组结果从此前已保存的验证记录读取；新三组重新加载最佳模型评估，检查划分和受控参数一致；不评估官方测试集。

2026-11-04 完成：缩放填充三次验证Top-1为74.25%、75.08%、73.58%；均值74.31%±0.75%，Top-5 93.06%，Macro-F1 0.7397。相对整图平均提升4.00个百分点、中心裁剪提升6.86个百分点，三个种子均提升。小于10%边界框占图组（n=33）平均准确率为82.83%，需注意样本量。以上均为验证集结果，未评估官方测试集。完整记录见 runs/letterbox_comparison/experiment_report.txt。
