# 历史结果档案

归档日期：2026-10-08。原始正文保留；文内“Active / In Progress / Pending”是历史状态，
不代表本分支当前任务。下表转录已有文档，不表示本次重新评估；checkpoint 原文件本地未核实。

## 旧剪枝：数据划分未确认

| 方案 | 点数 | 历史准确率 (%) | 历史 GFLOPs |
|---|---:|---:|---:|
| 0，未剪枝 | 25 | 96.29 | 3.46 |
| 7，手部剪枝 | 19 | 95.12 | 2.57 |
| 10，手部与下半身剪枝 | 14 | 94.23 | 1.86 |
| 16，继续剪枝 | 10 | 94.12 | 1.30 |

来源：[完整剪枝表](legacy_joint_pruning/acceleration_baseline_note.md)、
[当时总结](legacy_joint_pruning/2026_5_12_legacy_joint_pruning_summary.md)。
94.23% 比同表未剪枝 96.29% 低 2.06 个百分点。
原记录缺少 split、准确率对应人数及 checkpoint 证据；不得推断为 XSub 或 XView。
GFLOPs 为历史单人口径整理值，包含补测和缩放估计，不能等同于当前模型实测。
此处的 25 点历史标签也不能与当前标准 feeder 的 24 点输出混用。

## 早期识别：NTU60 XSub，旧 24 点双人配置

| 实验 | 观察比例 | Top-1 (%) | 来源 |
|---|---|---:|---|
| 全序列 baseline | full observation | 91.73 | [H1](early_recognition/H1_prefix-baseline/analysis.md) |
| 固定前缀 r=0.1 | 0.1 | 27.66 | [H1](early_recognition/H1_prefix-baseline/analysis.md) |
| 固定前缀 r=0.3 | 0.3 | 67.04 | [H1](early_recognition/H1_prefix-baseline/analysis.md) |

以下各列为不同观察比例下 Top-1 (%)：

| 实验 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| prefix_multi | 31.88 | 68.92 | 85.52 | 90.17 | 91.44 | 91.42 |
| semantic intent | 31.69 | 67.95 | 85.35 | 90.08 | 91.11 | 91.41 |
| trajectory intent | 31.32 | 68.37 | 85.42 | 90.08 | 91.25 | 91.54 |
| early_observable_v2 | 31.53 | 68.33 | 85.46 | 90.27 | 91.39 | 91.55 |
| adaptive gated intent | 31.30 | 68.48 | 85.53 | 未记录 | 未记录 | 未记录 |

来源：[H1 完整分析](early_recognition/H1_prefix-baseline/analysis.md)、
[H2 完整分析](early_recognition/H2_single-module-ablation/analysis.md)。
各方法的 checkpoint、Top-5 和评估路径留在原始分析文档中。
H3 尚无已填入的结果，不能作为已完成实验。

## 文件导航

- [H0 旧协议](early_recognition/H0_protocol-freeze/protocol.md)
- [H1 基线](early_recognition/H1_prefix-baseline/analysis.md)
- [H2 单模块消融](early_recognition/H2_single-module-ablation/analysis.md)
- [H3 联合模型计划](early_recognition/H3_joint-model-and-robustness/analysis.md)
- [旧标签映射说明](early_recognition/2026_5_12_ntu60_coarse_mapping_summary.md)

## 迁移对应

- experiments/H0～H3 目录 → archive/early_recognition/ 下同名目录。
- experiments/acceleration_baseline_note.md → archive/legacy_joint_pruning/ 下同名文件。
- docs/2026_5_12_legacy_joint_pruning_summary.md → archive/legacy_joint_pruning/。
- docs/2026_5_12_ntu60_coarse_mapping_summary.md → archive/early_recognition/。

训练代码、训练配置、数据映射、paper/figures、实际 work_dir 均未移动。
此前对话提及的 batch_capacity_20260915 目录在本次盘点时不在本地，
因此本轮没有重建其数值或伪造原始测速文件；找回原文件后再登记为辅助测速。
