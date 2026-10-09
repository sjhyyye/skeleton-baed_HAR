# 实验记录入口

本分支记录改进模块相对 **NTU60 XSub、14 点、单人、原始 SkateFormer block** 的效果。
早期动作识别和旧剪枝探索归入历史档案，不作为本轮结果。

## 从这里查

- [当前对照表](RESULTS.md)：本轮 baseline 和改进模块的精度、延迟、计算量。
- [Baseline 记录](current/baseline_14p_1person_seed1/README.md)：配置快照、运行方式、待补证据。
- [历史结果总表](archive/README.md)：旧剪枝和早期识别的数值及来源。
- [新实验记录模板](templates/run.md)：以后每次运行复制一份，使用独立 run_id。

## 目录约定

```text
experiments/
  README.md
  RESULTS.md
  current/<run_id>/     # 配置快照、结果元数据、分析；小文件纳入 Git
  archive/
    legacy_joint_pruning/
    early_recognition/
  templates/run.md
SkateFormer/work_dir/   # 原始训练输出、config.yaml、log.txt、checkpoint
SkateFormer/logs/       # 可选的 stdout 重定向；不作为唯一结果来源
```

训练输出继续使用已有路径，不移动正在运行的目录，也不改训练配置。
work_dir/logs 被 Git 忽略；服务器数据不会自动进入此处。每次实验结束后，
把实际运行的 config.yaml、选定 checkpoint 的名称/哈希、评估日志摘要和环境信息登记到记录中。
本地准备的配置快照不等于服务器实际运行配置。

## 登记规则

1. 相同数据划分、关节点名单及顺序、人物选择、64 帧输入、训练预算下比较模块。
2. RCA 默认 FFN 比例与 baseline 不同；必须写清实际比例，缩小 FFN 应另列消融。
3. Top-1 使用百分数；精度变化使用百分点；加速比 = baseline 延迟 / 变体延迟。
4. 精度和测速要对应同一模型配置；速度测试固定设备、精度、batch、warmup、迭代数和计时方法。
5. 没有证据的结果填“待补”或 null，不能填 0，也不能从旧表搬数值。
6. 正式完成状态需要 checkpoint 和评估证据；只有启动命令不能证明训练已完成。
7. 重跑用新 run_id 和新 work_dir，保留旧结果；训练结束前不覆盖实际配置快照。
