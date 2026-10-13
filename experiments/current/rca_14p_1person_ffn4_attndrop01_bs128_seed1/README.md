# M1 attention dropout=0.1

用户日志报告完成，2026-10-13 登记。单变量对照为 M1 的 attention dropout=0.5。
NTU60 XSub、14 点单人、batch=128、LR=0.001、500 epochs、seed=1、FFN=4。

- Best Top-1：88.609207%。
- Best epoch：483。
- 最佳权重的训练结束复测 Top-5：97.90%。
- 参数量：3,376,819。
- 相对 B0：-0.321465 个百分点；相对原 M1：+0.339662 个百分点。
- 第 500 epoch 训练准确率 96.44%，测试 Top-1 88.51%；最终汇总使用最佳权重而非最后权重。
- 结束处 Eval epoch: 1 是代码 eval(epoch=0) 的显示标签，最佳训练 epoch 仍为 483。
- dropout 在 eval 关闭，推理结构不变；本变体没有重新测速，不单独登记 11.69ms 为实测值。
- 降低 dropout 后本 seed 的精度提高，但尚未超过 B0；没有多 seed 统计结论。
- 日志包含训练结束后的最佳权重复测；尚未独立核验 checkpoint、SHA256 或服务器实际配置。

[原始日志](evidence/2026-10-13_user_log.txt) · [元数据](run.yaml) · [配置快照](config.snapshot.yaml)
