# RCA + 3帧局部时间卷积

2026-10-15 用户日志报告完成。Best Top-1=88.80329957%，epoch494，Top-5=97.73%，参数3,380,859。
相对 B0 -0.127373 个百分点；相对 RCA dropout0.1 anchors16 +0.194092 个百分点。
epoch500训练准确率96.61%；最终 Eval epoch: 1 为最佳权重复测标签。
本 seed 有改善，但未超过 B0；不能据此断定稳定收益或局部时间建模是唯一瓶颈。
服务器实际配置和 checkpoint 尚未独立核验。
此前随机权重结构测速10.326ms，比同次B0低28.30%，不是本 checkpoint 的实测延迟。

[原始用户日志](evidence/2026-10-15_user_log.txt) · [元数据](run.yaml) · [配置快照](config.snapshot.yaml) · [结构测速](../localt3_20261013/README.md)
