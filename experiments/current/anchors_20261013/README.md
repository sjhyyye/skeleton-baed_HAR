# RCA 时间锚点 16 vs 32 — 2026-10-13

本机 NVIDIA GeForce RTX 4070 Laptop GPU；PyTorch 1.12.1，CUDA runtime 11.3。
仅改变 rca_temporal_anchors，全 stage 共用 16 或 32。attn_drop=0.1、FFN=4、rank=2 均相同。
随机权重（固定 seed=1），非零随机 GPU-resident 输入 (1,3,64,14,1)；
FP32，TF32 off，eval/no_grad，CPU 线程 1，cuDNN benchmark=True。

预热 80 次；每轮额外预热 10 次；10 轮交替 A16/A32 顺序，每轮每模型 100 次。
每次 forward 前后同步 CUDA，用 perf_counter 统计墙钟延迟，包含 Python 调度。
不包含数据加载、主机到设备拷贝、骨架提取；不是训练耗时。
GPU 时钟和背景负载没有锁定。

| 锚点 | GFLOPs/样本（部分算子统计） | Mean ms | Median ms | P95 ms | samples/s | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| A16 | 1.807211 | 9.8198 | 9.6082 | 11.6236 | 101.83 | 3,376,819 |
| A32 | 1.822636 | 10.0498 | 9.7183 | 11.9910 | 99.50 | 3,376,819 |

32 锚点本次平均多 0.2299 ms（+2.34%），计入 FLOPs 增加约 0.85%，参数量不变。
轮均耗时范围：16 锚点 9.61–10.31 ms，
32 锚点 9.63–11.45 ms。
区间重叠，且有抖动；不把 2.34% 视为跨设备/运行固定开销。
本轮 16 锚点耗时与前次 11.69ms 不同；使用本次配对结果判断锚点影响，不跨运行拼接速度比。
没有测训练精度，不能据此声称 32 锚点更准。

FLOPs 使用 CPU torch.profiler(with_flops=True) 支持算子的部分统计，
乘加分开计数；CUPTI 不可用，因此不声称有 GPU kernel profile，独立 GPU 墙钟计时不依赖 CUPTI。

[脚本](benchmark.py) · [逐次原始耗时和环境](results.json)
· [16 锚点参数](A16.model_args.yaml) · [32 锚点参数](A32.model_args.yaml)

复现（在仓库根目录；新输出文件名避免覆盖）：

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/sjh/miniconda3/envs/skateformer/lib \
/home/sjh/miniconda3/envs/skateformer/bin/python \
experiments/current/anchors_20261013/benchmark.py \
--output experiments/current/anchors_20261013/results_repeat.json
```
