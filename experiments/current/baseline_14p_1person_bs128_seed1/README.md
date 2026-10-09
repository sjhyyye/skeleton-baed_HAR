# baseline_14p_1person_bs128_seed1

状态：配置已准备，尚未登记训练结果。登记日期：2026-10-10。

- 对照协议：NTU60 XSub、14 点、第一人、64 帧、seed=1、500 epochs。
- train/test batch=128；base LR=1e-3、min LR=1e-5、warmup LR=1e-7。
- warmup=25 epochs；AdamW、cosine、LSCE 与旧配置一致。
- 模块：skateformer，FFN ratio=4。
- 配置：[当前入口](../../../SkateFormer/config/train/acceleration/baseline_14p_1person.yaml)，[登记快照](config.snapshot.yaml)。
- 输出目录：SkateFormer/work_dir/acceleration/baseline_14p_1person_bs128_seed1/。
- 本次从头训练；B0 与 M1 使用相同 batch-128 协议。
- 旧 B0 的 88.9307% 属于 batch=32，不是这一行的结果。
- GPU、实际配置、checkpoint、Top-1/Top-5 和速度指标均待补。

用户实测 batch=128 训练最快，但尚未提供测速数值；不登记具体吞吐提升。
