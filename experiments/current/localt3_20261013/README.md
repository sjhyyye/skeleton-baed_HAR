# RCA local temporal kernel=3 消融与测速

共享 value 上增加时间深度卷积分支（3×1，无 bias），在 out_proj 前相加；每块一个可学习标量，初始化 0.1。
新增模型参数 rca_local_temporal_kernel 默认 0；旧配置不增加权重。新配置设为 3。
attn_drop=0.1、anchors=16、FFN=4、batch128、LR0.001、seed1，独立 work_dir；尚未训练，无精度结论。

## 验证

[验证脚本](validate.py) / [验证结果](validation.txt)：八个 block 分支生效，前向反向有限且新增参数有非零梯度；加载共享权重并将新分支缩放置零，输出与原 RCA 完全一致。配置只改变开关和 work_dir。

## 测速

RTX4070 Laptop，FP32，TF32 off，CPU threads1，eval/no_grad，GPU 输入 (1,3,64,14,1)。
随机权重；预热80次，各轮额外预热10次。6轮，每轮每模型100次，遍历三模型全部6种顺序。
同步墙钟计时包含 Python 调度，不含数据加载、H2D 或骨骼提取。GFLOPs 为 CPU profiler 支持算子的部分计数。
CUPTI 不可用不影响 CUDA 同步墙钟计时；没有 GPU kernel profiler 结论。

| 模型 | 参数量 | 部分 GFLOPs | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| B0 | 3386707 | 1.994590 | 14.4006 | 13.8134 | 18.1879 |
| RCA | 3376819 | 1.807211 | 9.6308 | 9.3150 | 12.2631 |
| LocalT3 | 3380859 | 1.810523 | 10.3256 | 9.8852 | 13.5194 |

LocalT3 比 RCA 增加约7.21%延迟（0.695ms），比 B0 减少28.30%延迟；参数比 RCA 多4,040。
这是本次结构测速，存在轮间波动；不能视为训练后 checkpoint 的 accuracy-latency 联合验证。

[测速脚本](benchmark.py) / [原始结果](results.json) / [源码哈希](source_hashes.json)。
目录保存三组配置快照和两个模型源码快照。

训练（在 SkateFormer 目录）：

    python -u main.py --config config/train/acceleration/rca_14p_1person_ffn4_attndrop01_localt3.yaml
