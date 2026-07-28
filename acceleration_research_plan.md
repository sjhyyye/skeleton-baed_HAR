# 基于算子等价改写的 SkateFormer 推理加速研究计划

## 1. 研究题目
面向剪枝后 `SkateFormer` 的算子级推理加速研究

## 2. 当前定位
这个分支不再以“大幅重写 block 结构”为主线，而是明确聚焦在：

1. 不依赖完整重训练
2. 尽量保持函数形式不变
3. 通过算子等价改写直接提升推理速度

也就是说，本分支的核心问题不是“设计一个全新的 mixer block”，而是：

**如何把现有 `SkateFormer` 的实现改写成更适合推理的算子图。**

当前目标场景已经明确固定为：

- `B=1`
- `C=192`
- `T=64`
- `V=14`

这是本分支的第一性约束。

## 3. 当前最值得做的研究问题
当前最值得研究的不是新的学习算法，而是以下几类算子级优化是否能在不训练的前提下立刻生效：

1. `cat + proj` 是否可以改写成多个 branch-wise projection 的求和
2. `Linear` 是否可以改写成 channel-first 的 `1x1 Conv2d`
3. `LayerNorm + Linear` 是否可以整体改写成更少 layout 变换的实现
4. graph branch 中按 head 切分并循环的实现是否可以 fused

这条线的价值在于：

- 立即可测
- 不依赖 7 天训练
- 可以用数值一致性直接验证正确性

## 4. 核心研究问题与假设

### 4.1 核心研究问题
1. 当前实现里，哪些地方是“数学形式没必要这样写，但实现上很慢”？
2. 哪些 rewrite 是严格等价的？
3. 哪些 rewrite 在 `B=1, V=14` 下最有效？
4. 多个 rewrite 叠加后，收益能否保持？

### 4.2 工作假设
1. `H1`：`cat + proj` 可以做严格等价改写，并去掉显式 `cat`。
2. `H2`：大量 channel-last `Linear` 可以改写为 channel-first `1x1 Conv2d`。
3. `H3`：graph branch 的循环实现可以合并为更大的 batched tensor op。
4. `H4`：operator-level rewrite 的收益在 `B=1, V=14` 场景下比大 batch 训练场景更明显。

## 5. 研究范围与 benchmark 冻结

### 5.1 范围
当前阶段只做：

- 推理 benchmark
- 数值一致性验证
- 算子级改写

当前阶段不做：

- 依赖重训练的新 block
- 大规模精度实验
- 结构级重新设计主结论

### 5.2 benchmark 约定
固定以下条件作为第一 benchmark：

- 输入形状：`B=1, C=192, T=64, V=14`
- 设备：当前 GPU
- 指标：
  - latency
  - throughput
  - 参数量
  - 数值误差

数值误差至少报告：

- `max_abs_diff`
- `mean_abs_diff`

## 6. 方法设计

### 6.1 `cat + proj` 改写
原始形式是：

`output = W [y1; y2; ...; yk] + b`

这可以严格改写为：

`output = W1 y1 + W2 y2 + ... + Wk yk + b`

因此首要任务是：

- 去掉显式 `cat`
- 把大投影拆成每个 branch 的独立投影再求和

这样做的优点：

- 数学上等价
- 不需要训练
- 直接命中推理热路径

### 6.2 `Linear -> 1x1 Conv2d`
对于按位置独立做通道混合的 `Linear`，可改写为：

- channel-first 下的 `1x1 Conv2d`

这一步的重点不是参数变少，而是：

- 减少 `permute`
- 减少 `contiguous`
- 保持 `[B, C, T, V]` 数据流更稳定

### 6.3 graph branch fuse
当前 graph 分支的一个问题是：

- 先 chunk
- 再按 head 循环
- 再 `einsum`
- 再 `cat`

这更像实现问题，不是模型本身的问题。  
因此这里应尝试：

- 更大的 batched `einsum`
- 更少的 Python 循环

### 6.4 组合策略
不是一开始就把所有 rewrite 一起上，而是：

1. 单独验证 `cat + proj`
2. 单独验证 `Linear -> 1x1 Conv`
3. 单独验证 graph fuse
4. 再做 cumulative benchmark

## 7. 验证策略

### 7.1 正确性验证
每个 rewrite 都必须做：

1. 权重映射
2. 随机输入 forward
3. 输出差异统计

只有在误差足够小的情况下，才能算“等价 rewrite”。

### 7.2 性能验证
每个 rewrite 都必须在同一 benchmark 下测：

- rewrite 前 latency
- rewrite 后 latency
- speedup

### 7.3 组合验证
最后再报告：

- 单项收益
- 累积收益
- 是否存在互相抵消

## 8. 基线、对比与消融

### 8.1 必须保留的基线
- 原始 block 实现

### 8.2 核心对比组
- baseline
- baseline + `cat + proj` rewrite
- baseline + `Linear -> 1x1 Conv`
- baseline + graph fuse
- baseline + all safe rewrites

### 8.3 消融重点
- 单项 rewrite 是否独立有效
- 哪一项贡献最大
- 多项 rewrite 是否可叠加
- 数值误差是否可接受

## 9. 当前阶段的可执行任务

### 9.1 第一优先级
1. 固定 benchmark 命令
2. 重测当前 baseline
3. 完成 `cat + proj` 等价改写

### 9.2 第二优先级
1. 完成 `Linear -> 1x1 Conv2d`
2. 完成 graph branch fuse
3. 做单项 benchmark

### 9.3 第三优先级
1. 做 cumulative benchmark
2. 总结哪些 rewrite 值得保留
3. 决定后续是否还需要训练型结构改动

## 10. 预期贡献的收敛表述
当前分支更稳妥的贡献应该写成：

1. 面向剪枝后小 `V` 的 `SkateFormer` 推理场景，系统分析现有实现中的算子级低效模式。
2. 提出若干不依赖重训练的等价改写方法。
3. 用真实 latency 和数值一致性共同验证 operator-level acceleration 的有效性。

## 11. 风险与应对

### 风险 1：理论等价但实际不更快
应对方式：直接删掉，不保留无效 rewrite。

### 风险 2：改写后误差过大
应对方式：将其归类为结构改动，而不是本分支的算子等价改写。

### 风险 3：单项收益成立，但叠加后无明显增益
应对方式：同时报告单项收益和组合收益，不强行讲故事。

## 12. 一句话总结当前计划
当前分支的研究计划已经明确收敛为：围绕 `B=1, C=192, T=64, V=14` 的剪枝后 `SkateFormer` 推理场景，优先研究不依赖重训练的算子等价改写，用数值一致性和真实 latency 直接验证加速效果。
