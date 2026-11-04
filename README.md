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

## 本机整图与鸟框 letterbox 控制实验

本实验用于排除“仅取消随机裁剪和中心裁剪即可带来提升”的解释。两种条件使用完全相同的 letterbox 几何处理和增强，仅保留水平翻转；唯一输入差异是鸟框条件使用真实边界框并每侧扩展10%，整图条件直接使用完整原图。

运行 Run_Full_Letterbox_Control.bat，或在项目目录运行：

    powershell -ExecutionPolicy Bypass -File run_full_letterbox_control.ps1

该实验在 NVIDIA GeForce RTX 3060 Laptop GPU 6GB、Python 3.9.20、PyTorch 2.5.1+cu124 上完成，训练 batch size 为32、workers为0。固定 split-seed=42，训练 seed=42、43、44；每项训练10轮，前2轮冻结主干。

2026-11-04 本机配对结果（3个训练种子的均值±样本标准差）：

- 整图 letterbox：Top-1 65.25% ± 0.44%，Top-5 88.28% ± 0.76%，Macro-F1 64.82% ± 0.55%。
- 鸟框 letterbox：Top-1 74.00% ± 0.36%，Top-5 93.44% ± 0.42%，Macro-F1 73.91% ± 0.33%。
- 鸟框相对整图的逐种子 Top-1 差值分别为 +9.33、+8.08、+8.83 个百分点，平均 +8.75 个百分点。

鸟框占原图面积不足10%时（n=33），整图 letterbox 与鸟框 letterbox 的平均 Top-1 分别为40.40%和84.85%；10-30%组（n=539）分别为62.21%和73.96%。小目标组样本量很小，应作为探索性结果。鸟框条件依赖额外真实位置标注，同时改变目标尺度和背景范围，因此结果支持继续研究前景定位，但不能单独证明背景的因果作用。

训练目录为 runs/full_letterbox_seed42、seed43、seed44 和 runs/crop_letterbox_3060_seed43、seed44（seed42 使用 runs/milestone_letterbox_3060）。可分享的分析结果同时保存在 runs/local_geometry_comparison 与 results/local_geometry_comparison，后者包含 summary.json、validation_results.csv、validation_predictions.csv、paired_differences.csv、bird_size_analysis.csv、performance_comparison.png 和 experiment_report.txt。该子实验完成时官方测试集尚未评估；最终测试见文末。

## 冻结主干与全参数微调对照

运行 Run_Freeze_FineTune_Comparison.bat，或在项目目录运行：

    powershell -ExecutionPolicy Bypass -File run_freeze_finetune_experiment.ps1

所有条件使用相同的鸟框 letterbox 输入、固定 split-seed=42、训练 seed=42、43、44、batch size 32 和 10 轮预算。比较三种训练策略：全程冻结主干、从第1轮起微调全部参数、前2轮冻结后微调全部参数。

2026-11-04 验证结果（3个训练种子的均值±样本标准差）：

- 全程冻结主干：Top-1 61.97% ± 0.99%，Top-5 87.86% ± 0.71%，Macro-F1 61.37% ± 1.20%。
- 从第1轮微调：Top-1 74.14% ± 0.46%，Top-5 93.08% ± 0.83%，Macro-F1 73.93% ± 0.59%。
- 冻结2轮后微调：Top-1 74.00% ± 0.36%，Top-5 93.44% ± 0.42%，Macro-F1 73.91% ± 0.33%。

从第1轮微调相对全程冻结的逐种子 Top-1 差值为 +11.42、+13.83、+11.25 个百分点。两轮预热与从第1轮微调的平均差值仅 -0.14 个百分点，且种子方向不一致（+0.50、-0.83、-0.08），没有证据支持两轮预热能稳定提升准确率。结论是：微调主干必要；两轮预热可作为历史基线保留，但最终训练不必依赖它。

结果保存在 runs/freeze_finetune_comparison 和 results/freeze_finetune_comparison，包含 summary.json、per_seed_results.csv、validation_predictions.csv、paired_differences.csv、performance_comparison.png、learning_curves.png 和 experiment_report.txt。该子实验完成时官方测试集尚未评估；最终测试见文末。

## 鸟框扩展范围消融

运行 Run_Margin_Ablation.bat，或在项目目录运行：

    powershell -ExecutionPolicy Bypass -File run_margin_ablation.ps1

固定 ResNet-18、真实鸟框、letterbox、从第1轮全参数微调、训练4794张、验证1200张、split-seed=42、训练 seed=42、43、44、batch size32和10轮。只改变鸟框每侧扩展比例：0%、5%、10%、20%。10%条件复用 runs/finetune_all_seed42、seed43、seed44；新增模型分别保存在 runs/margin000_seed*、runs/margin005_seed*、runs/margin020_seed*。

2026-11-04 验证结果（3个训练种子的均值±样本标准差）：

- margin 0%：Top-1 76.06% ± 1.54%，Top-5 93.58% ± 0.14%，Macro-F1 75.97% ± 1.44%。
- margin 5%：Top-1 75.19% ± 0.54%，Top-5 93.14% ± 0.94%，Macro-F1 75.03% ± 0.59%。
- margin 10%：Top-1 74.14% ± 0.46%，Top-5 93.08% ± 0.83%，Macro-F1 73.93% ± 0.59%。
- margin 20%：Top-1 72.78% ± 0.63%，Top-5 92.08% ± 0.63%，Macro-F1 72.73% ± 0.54%。

相对10%的逐种子Top-1差值：0%为+1.25、+3.17、+1.33个百分点；5%为+0.67、+0.92、+1.58个百分点；20%为-1.58、-1.17、-1.33个百分点。0%和5%三个种子都优于10%，20%三个种子都弱于10%。0%验证均值最高，但种子间波动较大；5%是更稳定的次优范围。

验证集平均输入构成：0%、5%、10%、20%下，鸟框分别占网络输入约100.00%、83.35%、71.49%、56.49%；裁剪区域分别约占原图34.78%、41.47%、47.83%、59.26%。这说明扩展比例越大，背景与目标尺度变化越明显。

结果保存在 runs/margin_ablation 和 results/margin_ablation，包含 summary.json、per_seed_results.csv、validation_predictions.csv、paired_differences.csv、area_summary.csv、bird_size_analysis.csv、performance_comparison.png、context_composition.png、bird_size_comparison.png、margin_inputs.png 和 experiment_report.txt。该子实验完成时官方测试集尚未评估；最终测试见文末。

## 背景干预与错误分析

运行 Run_Background_Analysis.bat，或在项目目录运行：

    powershell -ExecutionPolicy Bypass -File run_background_analysis.ps1

分析使用本机训练的三个鸟框 letterbox checkpoint，仅在固定验证集的1200张图片上评估。模型输入视野为真实鸟框每侧扩展10%；所有条件统一使用 letterbox。

- baseline：原始鸟框输入。
- background_blur：保留鸟体，将扩展框内、紧贴鸟框外部的背景做半径12像素高斯模糊。
- background_occluded：保留鸟体，将扩展框内的背景替换为RGB(124,116,104)。
- background_only：将紧贴鸟框内部的鸟体遮为RGB(124,116,104)，保留周围背景。

2026-11-04 三训练种子验证结果（均值±样本标准差）：

- baseline：Top-1 74.00% ± 0.36%，Top-5 93.44% ± 0.42%，Macro-F1 73.91% ± 0.33%。
- background_blur：Top-1 73.19% ± 0.51%，Top-5 92.94% ± 0.39%，Macro-F1 73.02% ± 0.57%。
- background_occluded：Top-1 72.83% ± 0.46%，Top-5 92.78% ± 0.10%，Macro-F1 72.64% ± 0.55%。
- background_only：Top-1 1.61% ± 0.34%，Top-5 6.92% ± 0.74%，Macro-F1 0.70% ± 0.16%。

背景模糊和三种子平均只下降0.81个百分点，背景遮挡下降1.17个百分点；遮住鸟体后准确率下降72.39个百分点。该结果支持正常预测主要依赖鸟体区域，但 background-only 是分布变化下的辅助诊断，不能单独证明全部普通输入中的因果依赖。按鸟框面积分组，背景模糊和遮挡的影响均较小，而遮住鸟体在所有大小组都会造成大幅下降。

结果保存在 runs/background_analysis 与 results/background_analysis。除上述三项外，还包含 seed42 的 Grad-CAM、每个背景条件的 Top-20 预测变化、Top-10 混淆类别对、最小/最大鸟框以及 improve/regress 案例比较。该子实验完成时官方测试集尚未评估；最终测试见文末。

## 最终冻结模型与官方测试集

最终模型在打开测试集前冻结：

- ResNet-18 ImageNet 预训练；
- CUB 真实鸟框，不额外扩展（margin 0%）；
- letterbox，最长边224、保持比例并居中填充；
- 从第1轮开始全参数微调；
- 训练 seed 43；验证集最佳 epoch 10；
- 固定 split-seed 42。

模型选择只使用验证集：margin 0% 三种子均值最高，seed 43 是 margin 0% 中验证 Top-1 最高的 checkpoint。

冻结目录：runs/final_candidate_margin000_seed43。best.pt SHA-256 为 0BE3EB6E5FB540ED7B90A83BDF54177AF25A35BEBF96C1B77B40642F61FB6982。

官方测试集唯一一次评估结果（5,794张）：

- Top-1：77.87%
- Top-5：94.15%
- Macro-F1：77.85%
- Cross entropy：0.8551

同一 checkpoint 的验证集 Top-1 为77.83%，测试集比验证集高0.04个百分点。测试后没有重新选择 checkpoint、seed、margin 或超参数。

测试结果保存在 runs/final_test_margin000_seed43 和 results/final_test_analysis；后者包含 test_metrics.json、test_confusion_matrix.csv、test_predictions.csv、per_class_accuracy.csv、summary.json、validation_vs_test.png、per_class_accuracy.png、top_confusion_pairs.png 和 final_test_report.txt。

## 报告与答辩版图

统一视觉风格的实验图保存在 results/report_visuals：

- 00_research_overview.png：六面板研究总览；
- 01_original_input_conditions.png；
- 02_controlled_whole_vs_bird.png；
- 03_training_strategy.png；
- 04_margin_ablation.png；
- 05_background_interventions.png；
- 06_background_by_bird_size.png；
- 07_top10_confusions.png。

最终模型的 Grad-CAM、混淆类别和模型卡保存在 results/final_visuals：gradcam_examples.png、top10_confusion_pairs.png、FINAL_MODEL_CARD.txt。

## Final Report 与提交包

Final Report 6页初稿位于 final_report/main.pdf，LaTeX 源文件为 final_report/main.tex。报告包含最终测试集结果、背景干预、模型比较、Grad-CAM、混淆分析、限制和课程AI使用披露。

可直接检查的提交包：

- submission/FINAL_REPORT.pdf
- submission/MILESTONE.pdf
- submission/SUPPLEMENTARY_MATERIAL.zip

补充材料 ZIP 约3.5MB，包含源码、报告源文件、主要结果表和统一风格图；不包含 CUB 数据、checkpoint、虚拟环境或下载缓存。
