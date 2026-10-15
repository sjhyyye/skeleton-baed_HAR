# RCA 实现优化：缓存与批量矩阵乘法

2026-10-15，保留 LocalT3 结构及全部参数，不需要重新训练。
缓存只包含固定时间距离矩阵，非持久 buffer；长度、anchors、kernel、设备或默认 dtype 改变时重建。
可学习的 temporal_local_strength 仍在每次前向中计算。
空间与时间注意力改为合并行维度后的 bmm，避免按通道广播注意力矩阵。保留必要的池化布局转换。

## 同次测速

RTX4070 Laptop、FP32、TF32 off、B1 T64 V14 M1、随机权重。
三个 RCA 变体加载完全相同的 state_dict。CPU threads1，eval/no_grad，预热80次，8轮×100次，轮间预热10次，轮换并反转测试顺序。
同步墙钟包含 Python 调度；不含数据加载或H2D。

| 模型 | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|
| B0 | 13.1165 | 12.9709 | 14.3690 |
| Before | 9.1416 | 8.9969 | 10.2449 |
| Cache | 8.3851 | 8.2434 | 9.4677 |
| Optimized | 8.2030 | 8.0422 | 9.3411 |

仅缓存相比优化前降低8.28%耗时；组合降低10.27%（0.939ms）。
组合相比仅缓存进一步降低2.17%；该小增益有轮间波动。
优化后比同次B0降低37.46%耗时。
每个RCA参数量均为3,380,859；没有改训练配置。

## 验证与限制

CPU/CUDA 共8组 block 前向及输入/参数梯度一致性通过，覆盖不同长度、anchors、开关和local分支。
严格加载旧 state_dict；完整模型随机输入输出在容差内一致，误差记录在results.json。
缓存不进入checkpoint，长度变化可重建。
额外验证 inference_mode 推理后恢复训练、dtype转换后重建缓存通过。
final.py 相比测速时 optimized.py 仅在缓存首次创建处加 inference_mode(False)，保证缓存可供后续反向使用；稳态命中路径不变。
尚未获得服务器训练checkpoint，不能声称完整测试集准确率已经复测。
CPU profiler记录copy/clone/linspace次数作为实现证据；CUPTI不可用，未做GPU kernel profiler归因。

[原始测速与验证](results.json) · [复现实验脚本](benchmark.py) · [最终实现快照](final.py)
