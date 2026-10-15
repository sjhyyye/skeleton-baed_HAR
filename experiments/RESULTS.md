# 当前模块改进对照表

协议：NTU60 XSub，joint 输入，14 点、第一人、64 帧。对照为原始 SkateFormer block。
当前训练协议为 batch=128、base LR=1e-3、500 epochs、seed=1；主测速仍为 batch=1、FP32，同设备同计时口径比较。

## 当前 batch=128 对照（结果更新：2026-10-15）

| 实验 | 运行记录 | 模块 | Best Top-1 (%) | 差值 (百分点) | Best epoch | Top-5 @ best (%) | 参数量 |
|---|---|---|---:|---:|---:|---|---:|
| B0 | [baseline_14p_1person_bs128_seed1](current/baseline_14p_1person_bs128_seed1/README.md) | 原始 SkateFormer，FFN=4 | 88.9307 | — | 500 | 97.74 | 3,386,707 |
| M1 | [rca_14p_1person_ffn4_bs128_seed1](current/rca_14p_1person_ffn4_bs128_seed1/README.md) | 全 stage RCA，FFN=4 | 88.2695 | -0.6611 | 489 | 未提供 | 3,376,819 |
| M1-attndrop01 | [rca_14p_1person_ffn4_attndrop01_bs128_seed1](current/rca_14p_1person_ffn4_attndrop01_bs128_seed1/README.md) | 全 stage RCA，FFN=4，attn_drop=0.1 | 88.6092 | -0.3215 | 483 | 97.90 | 3,376,819 |
| M1-attndrop01-anchors32 | [rca_14p_1person_ffn4_attndrop01_anchors32_bs128_seed1](current/rca_14p_1person_ffn4_attndrop01_anchors32_bs128_seed1/README.md) | 全 stage RCA，FFN=4，attn_drop=0.1，anchors=32 | 88.6153 | -0.3154 | 492 | 97.88 | 3,376,819 |
| M1-localt3 | [rca_14p_1person_ffn4_attndrop01_localt3_bs128_seed1](current/rca_14p_1person_ffn4_attndrop01_localt3_bs128_seed1/README.md) | RCA，dropout=0.1，anchors=16，局部时间卷积3 | 88.8033 | -0.1274 | 494 | 97.73 | 3,380,859 |

证据等级：用户提供的训练末期日志与完成汇总；尚未核验服务器实际配置、checkpoint 或独立复测。
RCA 最佳 epoch=489 的 Top-5 未贴出，不能用 epoch 495 或 500 的 Top-5 代替。

单 seed 初步观察：
- LocalT3 达到88.8033%，较无卷积的RCA dropout0.1提高0.1941个百分点，较B0低0.1274个百分点；未达成同时提升速度与精度的目标。
- anchors=32 达到 88.6153%，较 anchors=16 仅高 0.0061 个百分点，基本持平；仍未超过 B0。暂优先保留 anchors=16 作为后续结构消融起点。
- dropout=0.1 的 RCA 达到 88.6092%，比原 RCA 提高 0.3397 个百分点，仍比 B0 低 0.3215 个百分点。推理结构不变，但未重新测量该 checkpoint 的延迟。
- RCA 精度低 0.6611 个百分点，参数少 9,888（约 0.292%）；当前没有精度提升证据。
- 时间戳显示 B0 epoch 499/500 训练部分各约 73 秒，RCA epoch 496–500 为 59–60 秒。
  这是不同时间段的粗略观察，不能归因于模块本身，也不是推理延迟。
- 本机随机权重结构测速已完成，见下表；最佳 checkpoint 的推理耗时仍待核验。
- 先核验两组最佳 checkpoint 并补同设备 batch=1 FP32 推理对照，再决定是否进行 stage 替换或 FFN 消融。

## 本机推理结构测速（2026-10-12）

RTX 4070 Laptop，FP32、TF32 off、batch=1、GPU-resident 输入 (1,3,64,14,1)，随机权重。
5 轮 × 100 次同步墙钟计时；不含数据加载/传输。FLOPs 是 profiler 支持算子的部分计数。

| 模型 | GFLOPs/sample（部分计数） | Mean ms | P95 ms | samples/s |
|---|---:|---:|---:|---:|
| B0 | 1.9946 | 17.028 | 20.469 | 58.73 |
| M1 | 1.8072 | 11.694 | 14.919 | 85.51 |

RCA 本次平均延迟降低 31.32%，加速比 1.456x；计入 FLOPs 下降 9.39%。
速度与训练精度来自不同证据：不能表述为已完成最佳权重的 accuracy-latency 验证。
[完整协议、脚本与原始结果](current/inference_20261012/README.md)。

## 旧 batch=32 记录

下表保留原结果，不与新的 batch=128 结果混为同一训练协议。

| Run ID | 模块 | 状态 | Top-1 (%) | 相对 baseline (百分点) | 延迟 (ms/sample) | 加速比 | GFLOPs/sample | 参数量 |
|---|---|---|---|---|---|---|---|---|
| [baseline_14p_1person_seed1](current/baseline_14p_1person_seed1/README.md) | 原始 SkateFormer | 日志报告完成；混写及权重待核实 | 88.9307 | — | 待补 | — | 待补 | 3,386,707 |

2026-10-10 已登记用户日志中的 B0 结果：best epoch=497，Top-5=97.71%。日志疑似混写，尚未核验实际 checkpoint 或独立复测；不作为完整验证结果。batch=32 的 RCA 没有最终结果记录。
本表是摘要；每行详细证据放在对应 run 目录。后续有实际运行再追加变体，避免把计划误当结果。

历史 94.23% 的 split 未知，91.73% 属于旧 24 点双人 XSub；均不能填进本表。
历史记录见 [归档总表](archive/README.md)。

## 时间锚点成本消融（2026-10-13）

同机同次测试，dropout=0.1、FP32、B=1、随机权重，每组 10×100 次：
16 anchors 平均 9.820 ms，32 anchors 平均 10.050 ms
（+2.34%，轮间有抖动）；部分 GFLOPs 1.8072 → 1.8226。
参数量相同。后续用户日志报告 32 anchors Top-1=88.6153%，较 16 anchors 的 88.6092% 基本持平；checkpoint 尚未独立核验。
[协议与原始记录](current/anchors_20261013/README.md)。

## 局部时间卷积分支消融（2026-10-13）

新配置：RCA attn_drop=0.1、anchors=16，增加共享 value 上的3帧深度时间卷积，初始可学习缩放0.1。
2026-10-15 用户报告最佳准确率88.8033%，epoch494，仍比B0低0.1274个百分点；旧配置默认关闭此分支。

同次本机 RTX4070 Laptop、FP32、B1 随机权重，6轮×100次：
B0 14.401ms、RCA 9.631ms、新 RCA LocalT3 10.326ms。
新分支比 RCA 慢7.21%，仍比 B0 快28.30%；新参数量3,380,859。
部分 GFLOPs 分别1.994590、1.807211、1.810523。
[协议、验证与原始记录](current/localt3_20261013/README.md)。
