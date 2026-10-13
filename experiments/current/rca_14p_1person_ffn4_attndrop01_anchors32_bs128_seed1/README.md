# RCA attn_drop=0.1、temporal_anchors=32

2026-10-13 登记用户提供的完成日志。NTU60 XSub、14 点单人、batch=128、LR=0.001、seed=1、FFN=4。

- Best Top-1：88.61527264%，best epoch：492。
- 最佳权重复测 Top-5：97.88%，参数量：3,376,819。
- 相对 anchors16：+0.006065 个百分点；相对 B0：-0.315400 个百分点。
- epoch500 训练准确率 96.53%，测试 Top-1 88.40%；末尾 Eval epoch: 1 是最佳权重复测标签。
- 本 seed 结果与 anchors16 基本持平，尚无可靠增益证据；未独立核验 checkpoint 或服务器配置。
- 此前随机权重结构测速 anchors16/32 为 9.820/10.050 ms（+2.34%，轮间波动重叠），不是本次训练权重的延迟。

[用户日志摘录](evidence/2026-10-13_user_log.txt) · [元数据](run.yaml) · [配置快照](config.snapshot.yaml) · [结构测速](../anchors_20261013/README.md)
