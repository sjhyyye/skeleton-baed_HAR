# baseline_14p_1person_seed1

状态：已收到用户训练日志，至少一条运行已输出完成汇总；checkpoint 与独立复测待核实。
2026-10-09 22:58:08 的汇总报告 best Top-1=88.93067265%、best epoch=497、
参数量=3,386,707；epoch 497 对应 Top-5=97.71%。
[原始日志片段](evidence/2026-10-09_pasted_training_excerpt.txt)已按原字节保存。
日志存在交错 epoch 与重复结果，疑似重复运行或合并输出；若共用 work_dir，
权重及评估文件可能覆盖。不能把两条同 seed 轨迹算作两次独立重复实验。
本地未获得服务器实际 config.yaml 和 checkpoint，精度暂作为“日志报告值”登记。

## 实验定义

- NTU60 XSub，60 类，joint，输入 (B,3,64,14,1)。
- 1-based 关节点及顺序：2,3,4,5,6,8,9,10,12,15,16,19,20,21。
- feeder partition=False，实际取出 14 点；单人直接保留第一人。
- block_type=skateformer；未启用 RCA、前缀采样、intent 和一致性损失。
- 四类分区均为 [8,7]，MLP ratio=4，num_heads=32。
- AdamW，batch=32，500 epochs，seed=1，LR=2.5e-4，25 epochs warmup，cosine，LSCE。
- 沿用标准时间采样：训练 p_interval=[0.5,1]，测试 [0.95]，uniform=True。
- 与官方 24 点双人配置相比，人数、排列、分区和 batch/LR 均有变化；这是本分支模块对照基线，不是单变量剪枝复现。

## 文件

- [可执行配置](../../../SkateFormer/config/train/acceleration/baseline_14p_1person.yaml)
- [登记时配置快照](config.snapshot.yaml)
- [结果与证据元数据](run.yaml)

实际训练输出预期在仓库内：
`SkateFormer/work_dir/acceleration/baseline_14p_1person_seed1/`。
路径仅表示预期位置，不表示 checkpoint 已存在。

## 运行方式（历史 batch=32）

2026-10-10 起原配置入口已更新到 batch=128。下面命令改为使用本记录的冻结快照，以保留旧协议；实际新实验请使用 current 中带 bs128 的记录。

在服务器 SkateFormer 目录执行：

```bash
conda activate skateformer
python -u main.py --config ../experiments/current/baseline_14p_1person_seed1/config.snapshot.yaml
```

默认输出到终端并保留 work_dir/log.txt。若使用 --print-log False，
应另行保存评估证据以便登记结果；这里未修改程序的日志行为。

## 训练后补录

1. 确认实际保存的 config.yaml 与此快照一致，包括命令行覆盖。
2. 填运行 commit、GPU/环境、开始结束时间和实际完成 epoch。
3. 填 best epoch、Top-1/Top-5、checkpoint 路径与 SHA256、评估日志位置。
4. 用同一配置和 checkpoint 测计算成本，记录设备、精度、batch、计时方法及重复测量。
5. 更新 run.yaml 和当前对照表；仍缺证据的字段保留 null。
