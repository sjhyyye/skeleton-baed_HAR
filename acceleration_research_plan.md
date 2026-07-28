# 基于 ACmix 思路的 SkateFormer 计算加速研究计划

## 1. 研究题目
基于 `ACmix` 风格共享投影的 `SkateFormer` 骨架动作识别模型计算加速研究

## 2. 当前定位
这个分支已经明确切换研究目标，不再围绕“早期动作识别”展开，而是回到“如何让现有 `SkateFormer` 跑得更快、算得更省”的主问题。

当前定位不是泛泛地做轻量化，也不是直接复用旧的关节点剪枝表，而是做一条更具体的架构线：

1. 以 `ACmix` 的设计思想作为主参考。
2. 先分析当前 `SkateFormerBlock` 的真实计算瓶颈，而不是先假设瓶颈。
3. 在 skeleton 时空 token 场景下，重点重构 `partition -> reverse -> cat -> proj` 这套 block 组织方式，而不是只盯着单个算子。
4. 用真实延迟、吞吐、`GFLOPs` 和精度共同评价，而不是只看单一指标。

当前更明确的论文切入点是：

- 不把“共享前面的重投影”本身当主创新，因为这更接近已有 ACmix 思路。
- 把主方法收敛为 `partition-free / branch-collapsed / cat-proj-free` 的 `SkateFormer` block redesign。
- 强调该问题在剪枝后的小关节点推理场景更明显，特别是 `B=1, V=14`。

## 3. 参考论文给出的关键启发
参考论文 `On the Integration of Self-Attention and Convolution (arXiv:2111.14556)` 的关键启发不是“注意力和卷积可以简单并排放在一起”，而是：

1. 卷积和自注意力的大头计算都可以被理解为前面的 `1x1` 特征投影。
2. 真正昂贵的往往不是后面的局部聚合本身，而是前面的高维通道映射。
3. 因此，更合理的设计是让注意力路径和卷积路径共享前面的特征生成，再用轻量聚合分别完成不同归纳偏置。
4. 这给当前 `SkateFormer` 一个很直接的方向：不要继续堆更多分支，而要想办法减少分支前后的重通道变换与冗余拼接。

## 4. 核心研究问题与假设

### 4.1 核心研究问题
1. 当前 `SkateFormerBlock` 中，真实运行时的主要开销到底来自哪里？
2. `ACmix` 的共享投影思想，迁移到 skeleton 的 `(T, V)` token 结构后，最适合放在哪一层？
3. 是整网替换更有效，还是只替换部分 stage 更有效？
4. `GFLOPs` 的下降是否真的能换来 wall-clock latency 的下降？

### 4.2 工作假设
1. `H1`：当前块中的大头成本仍然主要来自通道映射和多分支输出融合，而不只是某一个局部聚合算子。
2. `H2`：借鉴 `ACmix` 的共享投影后，可以用更少的中间通道预算同时支撑局部卷积式建模和注意力式建模。
3. `H3`：部分 stage 替换比整网替换更容易先拿到稳定的 Pareto 改进。
4. `H4`：如果一个方案只降低 `GFLOPs`、却不能降低真实延迟，那它不能作为主结果。
5. `H5`：对剪枝后的小 `V` 推理场景，真正值得写论文的不是单独优化某个算子，而是取消显式 partition、多分支展开和 `cat + proj` 融合这一整套 block 模式。

## 5. 研究范围与协议冻结

### 5.1 数据集与任务范围
当前阶段先锁定如下范围：

- 主数据集：`NTU60`
- 首轮协议：`XSub`
- 第二验证协议：`XView`
- 任务定义：标准 full-sequence skeleton action recognition
- 暂缓内容：`NTU120`、量化、设备特化部署

### 5.2 计算 benchmark 约定
首轮必须冻结两条路径：

1. 精度路径  
   使用当前 `SkateFormer` 的标准训练/测试配置做 `Top-1` 对比。

2. 速度路径  
   使用 `SkateFormer/tools/benchmark_inference.py` 做统一 latency / throughput / `GFLOPs` 报告。

默认输入约定先固定为：

- `T = 64`
- `V = 25`
- `M = 2`
- 至少报告 `batch_size = 1` 与一个较大 batch 的速度结果

### 5.3 当前阶段必须避免的漂移
当前阶段不是先大改模型，而是先避免以下混乱：

- 用不同 checkpoint 比较速度
- 用不同输入尺寸比较 `GFLOPs`
- 用不同 batch size 选择性展示结果
- 把旧的 early-recognition 指标混入新的加速主线

## 6. 方法设计

### 6.1 对当前 `SkateFormerBlock` 的代码级理解
现有块并不是“纯注意力块”。它已经包含：

- 一次 `mapping`
- 一条 graph-conv 路径
- 一条 temporal-conv 路径
- 四条 partitioned attention 路径
- 一次 `proj`
- 一次 `MLP`

所以新工作的重点不是再加一个“混合块”概念，而是：

1. 找出哪些通道变换和分支拼接最贵。
2. 判断哪些分支值得保留、哪些分支可以合并或缩窄。
3. 用共享中间表示取代过宽的 branch-specific 预算。

### 6.2 ACmix 风格共享投影块
首轮新块设计建议遵循以下原则：

1. 只做一次主特征生成，尽量避免为不同分支重复构造高维中间特征。
2. 在共享特征之上，分出两类轻量聚合：
   - 卷积式或局部图时序聚合
   - 注意力式或动态加权聚合
3. 输出阶段尽量避免“大拼接 + 大投影”的重融合方式。
4. 若可能，用可学习混合权重代替部分固定宽度分配。

但当前版本更推荐把这套思想落成一个更明确的 block redesign，而不是停留在“共享投影”四个字：

- **partition-free**：不再显式 `view / permute / reverse` 去构造四种 partition token。
- **branch-collapsed**：不再保留 `gconv + tconv + 4 attention` 这种 6 路展开形式，而是压成更少的必要分支。
- **cat-proj-free**：不再默认采用所有分支完整输出后再 `cat + proj` 的融合方式。

换句话说，真正要改的是 block 的内部拓扑，而不是只加一个新算子。

### 6.3 分阶段替换策略
不建议一开始整网替换，优先按以下顺序验证：

1. 只替换第一阶段
2. 只替换中后阶段
3. 替换所有 stage

这样更容易回答“收益来自哪里”，也更容易定位失败原因。

### 6.4 当前最值得写进论文的 block-level redesign
当前版本建议把方法草图收敛为下面这类结构：

`input -> shared pre-mix -> local mixer -> relation mixer -> lightweight fusion -> residual -> slim FFN`

它和原始 `SkateFormerBlock` 的关键区别不是某个层更快，而是：

1. 去掉显式 `partition -> attention -> reverse`
2. 去掉 6 路完整 branch materialization
3. 去掉 `cat(y) -> proj`
4. 用少分支、轻融合替代“先展开、后重融合”

这样写出来的方法更像一个新的 skeleton mixer，而不是若干小优化拼在一起。

### 6.5 训练期精度恢复
如果新块出现“速度有提升，但精度掉得偏多”的情况，优先考虑：

- 以原始 `SkateFormer` 作为 teacher 的蒸馏
- 轻量的 logits KD
- 中间特征对齐

但要明确：蒸馏是精度恢复手段，不是本分支的主创新点。

### 6.6 与结构剪枝的关系
旧仓库里已经有一批 joint pruning 结果。这些结果当前只作为后续组合实验的候选，不是第一阶段的主线。

更合理的顺序是：

1. 先把架构级共享投影故事做清楚。
2. 再测试“新块 + 剪枝”是否进一步改善 Pareto。

不过当前论文动机已经进一步收敛为：剪枝后的小 `V` 场景放大了原始 block 组织方式的问题，因此 block redesign 和剪枝是强相关的，不再是完全割裂的两条线。

## 7. 损失函数与训练策略

### 7.1 首轮损失
首轮优先保持训练目标简单：

`L = L_cls`

如果需要做精度恢复，再扩展为：

`L = L_cls + lambda_kd * L_kd`

其中：

- `L_cls`：动作分类损失
- `L_kd`：teacher-student 蒸馏损失

### 7.2 训练流程
建议分三步：

1. 跑通原始 `SkateFormer` baseline  
   得到新的精度和速度统一参考表。

2. 训练单个 `ACmix` 风格原型  
   先只做部分 stage 替换。

3. 在确认有真实加速趋势后  
   再做整网替换、蒸馏恢复或与剪枝结合。

## 8. 基线、对比与消融

### 8.1 必须保留的基线
- 原始 `SkateFormer`
- 简单宽度缩减版本
- 简单 head 数缩减版本
- 历史 joint pruning 候选中的代表方案

### 8.2 核心对比组
- `SkateFormer + redesign block@stage1`
- `SkateFormer + redesign block@late_stages`
- `SkateFormer + redesign block@all_stages`
- `SkateFormer + redesign block + KD`
- `SkateFormer + redesign block + pruning`（后续）

### 8.3 消融重点
重点分析以下因素：

- 是否显式 partition
- 分支是否从 6 路压缩到 2 路或 3 路
- 是否保留 `cat + proj`
- 共享投影宽度
- 卷积式分支与关系建模分支的预算分配
- stage-wise replacement 的差异
- `GFLOPs` 和 latency 是否一致
- 是否需要 KD 才能维持精度

## 9. 评价指标与分析维度

### 9.1 主要指标
- `Top-1 Accuracy`
- latency (`ms/iter`)
- throughput (`samples/s`)
- `GFLOPs`
- 参数量

### 9.2 辅助分析
- 不同 stage 替换位置的收益差异
- 小 batch 与大 batch 下的速度差异
- 理论复杂度下降与真实延迟下降的偏差
- 不同关节点数量下的收益是否一致

## 10. 当前阶段的可执行任务

### 10.1 第一优先级
1. 固定 benchmark 命令  
   明确精度路径与速度路径各自的 canonical 命令。

2. 跑出新的原始 baseline  
   统一记录 `Top-1`、latency、throughput、`GFLOPs`、参数量。

3. 做块级 profile  
   判断“单个算子慢”还是“block 组织方式不合理”才是主要矛盾。

### 10.2 第二优先级
1. 实现第一版 partition-free / branch-collapsed block
2. 先做部分 stage 替换
3. 记录第一轮 Pareto 变化
4. 若速度提升明显但精度下降，加入 KD 恢复

### 10.3 第三优先级
1. 扩展到 `XView`
2. 结合旧 pruning 方案
3. 补充 Pareto 图、模块结构图和失败案例

## 11. 预期创新点的收敛表述
当前版本更稳妥的创新点应该收敛为以下三条：

1. 针对剪枝后小关节点数的 skeleton 时空 token，指出原始 `SkateFormer` 的显式 partition、多分支展开和 `cat + proj` 融合在推理时存在结构性低效。
2. 提出一种 `partition-free / branch-collapsed / cat-proj-free` 的 skeleton mixer block，在保持必要关系建模能力的同时改写 block 内部拓扑。
3. 系统比较“部分替换”和“整网替换”的 accuracy-cost Pareto，并强调 wall-clock latency 与 `GFLOPs` 的一致性验证。

## 12. 可能风险与应对

### 风险 1：理论上更省，但实际不更快
应对方式：把 wall-clock latency 作为主指标之一，不能只报 `GFLOPs`。

### 风险 2：新块速度变快但精度下降过多
应对方式：优先尝试 KD 恢复，再决定是否保留该路线。

### 风险 3：简单缩宽/减头就能达到相同收益
应对方式：必须保留简单 baseline，不能拿复杂改动和弱 baseline 比。

### 风险 4：整网替换过早导致结论混乱
应对方式：先做 stage-wise replacement，分清收益来源。

### 风险 5：旧 pruning 结果干扰主线
应对方式：先把它们明确降级为“后续组合实验候选”，不与第一阶段架构结果混合叙述。

### 风险 6：只优化了 10% 左右的局部开销，整体收益不足以支撑论文
应对方式：不要把目标写成“优化 partition/reverse/cat”本身，而要把它提升为 block-level redesign，争取同时带动 attention/fusion 结构一起变化。

## 13. 当前版本的阶段安排

### 阶段 A0：benchmark 冻结
- 锁定 `NTU60 XSub`
- 锁定速度脚本与输入约定
- 锁定结果汇报模板

### 阶段 A1：baseline 建立
- 重测原始 `SkateFormer`
- 输出第一版精度-速度基线表
- 完成块级 profile

### 阶段 A2：原型验证
- 实现第一版 partition-free / branch-collapsed block
- 先做局部 stage 替换
- 输出第一版 Pareto 对比

### 阶段 A3：系统消融
- 比较不同替换范围
- 比较不同通道预算
- 比较是否需要 KD

### 阶段 A4：扩展与写作
- 扩展到 `XView`
- 视情况叠加 pruning
- 固化图表与实验结论

## 14. 一句话总结当前计划
当前这条研究线已经被明确重置为“以 `ACmix` 作为启发，但把真正的论文方法收敛为 `partition-free / branch-collapsed / cat-proj-free` 的 `SkateFormer` block redesign；先冻结 benchmark，再做块级 profile，再做部分 stage 替换验证，最后用真实 Pareto 结果决定是否继续扩展”的加速计算计划。
