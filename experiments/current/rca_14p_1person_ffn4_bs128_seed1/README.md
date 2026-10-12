# rca_14p_1person_ffn4_bs128_seed1

状态：用户日志报告训练完成；实际 checkpoint 和独立复测待核实。配置登记于 2026-10-10，结果登记于 2026-10-12。

- 对照协议：NTU60 XSub、14 点、第一人、64 帧、seed=1、500 epochs。
- train/test batch=128；base LR=1e-3、min LR=1e-5、warmup LR=1e-7。
- warmup=25 epochs；AdamW、cosine、LSCE 与旧配置一致。
- 模块：skate_rca，FFN ratio=4。
- 配置：[当前入口](../../../SkateFormer/config/train/acceleration/rca_14p_1person_ffn4.yaml)，[登记快照](config.snapshot.yaml)。
- 输出目录：SkateFormer/work_dir/acceleration/rca_14p_1person_ffn4_bs128_seed1/。
- 本次从头训练；B0 与 M1 使用相同 batch-128 协议。
- batch=32 的旧记录独立保留。本次 batch=128 B0 恰好报告相同 Top-1，但 best epoch 为 500。
- GPU、实际服务器配置、checkpoint 及推理速度指标仍待补。

用户提供了训练末期时间戳，可作粗略耗时观察；不能替代同设备推理 benchmark。

## 已报告结果

- Best Top-1：88.269546%。
- Best epoch：489。
- 参数量：3,376,819。
- 最佳 epoch 的 Top-5：未提供；不能使用最后 epoch 的值代替。
- [原文汇总片段](evidence/2026-10-11_user_summary_excerpt.txt)；[结果元数据](run.yaml)。
