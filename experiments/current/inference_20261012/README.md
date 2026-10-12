# 本机 B0 / M1 推理测试 — 2026-10-12

## 测试条件

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU；此处不是此前训练日志中的 RTX 4080。
- Python / PyTorch / CUDA runtime：3.9.16 / 1.12.1 / 11.3。
- 输入：(1,3,64,14,1)，非零随机骨架；index_t 为 [-1,1] 上的 64 帧。
- FP32，TF32 关闭，eval + no_grad，CPU 线程 1，cuDNN benchmark 开启。
- 两个模型均随机初始化；本地没有对应训练 checkpoint。参数量与训练日志匹配，但未做权重精度验证。
- 输入预先驻留 GPU；仅测分类器，不含骨架提取、CPU 数据预处理及 H2D 传输。
- 每模型预热 80 次，每轮再预热 10 次；5 轮，每轮 100 次。
- 每次 forward 前后 CUDA synchronize，以 perf_counter 记录墙钟时间，包括 Python 调度。
- 顺序 B0/M1、M1/B0 交替。报告吞吐为 batch=1 顺序推理的 1000 / mean_ms，不是大 batch 最大吞吐。
- 机器有显示/后台负载，温度/时钟未锁定；结果适用于本次本机测试条件。

## 结果

| 模型 | 参数量 | 计入的 GFLOPs/sample | Mean ms | Median ms | P95 ms | samples/s |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3,386,707 | 1.994590 | 17.0278 | 16.6895 | 20.4687 | 58.73 |
| M1 | 3,376,819 | 1.807211 | 11.6941 | 11.3470 | 14.9192 | 85.51 |

RCA / M1：平均延迟下降 31.32%，
加速比 1.456x，
计入的 FLOPs 下降 9.39%。
不能把该随机权重模型的耗时当作已核验最佳 checkpoint 的测速。

## FLOPs 口径与限制

CPU torch.profiler(with_flops=True) 运行相同模型形状，汇总工具支持的算子计数，
乘法和加法分别计数（不是 MACs）。这是部分算子统计，不是完整精确 FLOPs。
results.json 保留计入 FLOPs 的算子及未计数算子列表。
Profiler 报告 CUPTI 初始化不可用；这里没有依赖 CUDA profiler activity：
FLOPs 来自 CPU profile，GPU 耗时来自独立同步墙钟测量。不声称有 GPU kernel 时间线。

## 复现

在仓库根目录的 WSL 环境执行；使用新输出文件名避免覆盖原始结果：

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/sjh/miniconda3/envs/skateformer/lib \
/home/sjh/miniconda3/envs/skateformer/bin/python \
experiments/current/inference_20261012/benchmark.py \
--output experiments/current/inference_20261012/results_repeat.json
```

以后有本地 checkpoint 可加 --b0-weights PATH --m1-weights PATH；使用严格加载。
[脚本](benchmark.py)、[原始结果](results.json)、[B0 配置快照](B0.config.yaml)、[M1 配置快照](M1.config.yaml)。

机器状态（name, driver, temperature, power, utilization, memory）：
```text
before: NVIDIA GeForce RTX 4070 Laptop GPU, 591.86, 48, 11.57 W, 39 %, 1682 MiB
after: NVIDIA GeForce RTX 4070 Laptop GPU, 591.86, 54, 29.41 W, 17 %, 3211 MiB
```
