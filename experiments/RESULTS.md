# 当前模块改进对照表

协议：NTU60 XSub，joint 输入，14 点、第一人、64 帧。对照为原始 SkateFormer block。
训练 batch=32、500 epochs、seed=1；主测速 batch=1、FP32，同设备同计时口径比较。

| Run ID | 模块 | 状态 | Top-1 (%) | 相对 baseline (百分点) | 延迟 (ms/sample) | 加速比 | GFLOPs/sample | 参数量 |
|---|---|---|---|---|---|---|---|---|
| [baseline_14p_1person_seed1](current/baseline_14p_1person_seed1/README.md) | 原始 SkateFormer | 日志报告完成；混写及权重待核实 | 88.9307 | — | 待补 | — | 待补 | 3,386,707 |

2026-10-10 已登记用户日志中的 B0 结果：best epoch=497，Top-5=97.71%。日志疑似混写，尚未核验实际 checkpoint 或独立复测；不作为完整验证结果。RCA 已有代码入口，但尚未登记配套运行证据，不填结果行。
本表是摘要；每行详细证据放在对应 run 目录。后续有实际运行再追加变体，避免把计划误当结果。

历史 94.23% 的 split 未知，91.73% 属于旧 24 点双人 XSub；均不能填进本表。
历史记录见 [归档总表](archive/README.md)。
