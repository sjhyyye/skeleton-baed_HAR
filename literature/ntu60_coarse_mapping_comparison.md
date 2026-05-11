# NTU60 Semantic Coarse vs Trajectory Coarse 对照文档

## 1. 文档目的
本文档将 `NTU60` 的两套 coarse 映射统一整理到一个地方，区分它们的设计目标、分类原则和适用场景，避免后续在实验和论文中混用术语。

两套映射分别是：

- `semantic coarse`：按动作语义、行为功能、作用对象进行归类
- `trajectory coarse`：按主导关节运动模式、身体部位参与方式、轨迹形态进行归类

## 2. 使用建议

### 2.1 作为主实验标签的建议
主实验建议优先使用 `semantic coarse`。

原因：

- 更容易写成论文中的高层先验或辅助监督
- 更容易解释“为什么在早期阶段就有帮助”
- 更接近 `coarse intent / action family` 的叙事

### 2.2 作为稳健性标签的建议
`trajectory coarse` 更适合作为稳健性对照。

原因：

- 更贴近骨架模型直接看到的关节变化轨迹
- 能检验收益是否只是来自语义先验
- 能帮助区分“高层语义辅助”和“低层运动模式辅助”两种来源

## 3. 术语约束

- 不要把 `trajectory coarse` 写成 `intent`
- `semantic coarse` 更准确的写法可以是 `coarse action family` 或 `coarse semantic family`
- 如果论文里仍使用 `intent`，应明确说明这里的 intent 是一种粗粒度行为功能标签，而不严格等同于心理学意义上的真实主观意图

## 4. Semantic Coarse 定义

### 4.1 Semantic 10 类

| ID | Name | 说明 |
|---|---|---|
| `C10-01` | `eat_drink` | 吃喝摄入类 |
| `C10-02` | `personal_grooming` | 个人清洁整理 |
| `C10-03` | `dressing_accessories` | 穿戴与配饰整理 |
| `C10-04` | `object_manipulation` | 物体拿取、放置、处理 |
| `C10-05` | `posture_locomotion` | 姿态变化与移动 |
| `C10-06` | `expressive_gestures` | 手势表达与符号动作 |
| `C10-07` | `reading_writing_device_use` | 阅读、书写、设备使用 |
| `C10-08` | `health_body_state` | 身体不适、异常状态 |
| `C10-09` | `interpersonal_interaction` | 非攻击性人际互动 |
| `C10-10` | `aggressive_interaction` | 攻击性人际互动 |

### 4.2 Semantic 8 类

| ID | Name | 来源 |
|---|---|---|
| `C8-01` | `self_care_and_dressing` | 合并 `C10-01`、`C10-02`、`C10-03` |
| `C8-02` | `object_manipulation` | 对应 `C10-04` |
| `C8-03` | `posture_locomotion` | 对应 `C10-05` |
| `C8-04` | `expressive_gestures` | 对应 `C10-06` |
| `C8-05` | `reading_writing_device_use` | 对应 `C10-07` |
| `C8-06` | `health_body_state` | 对应 `C10-08` |
| `C8-07` | `interpersonal_interaction` | 对应 `C10-09` |
| `C8-08` | `aggressive_interaction` | 对应 `C10-10` |

## 5. Trajectory Coarse 定义

### 5.1 Trajectory 10 类

| ID | Name | 说明 |
|---|---|---|
| `M10-01` | `hand_to_head_face` | 手到头/脸/口区域的主导运动 |
| `M10-02` | `torso_self_touch_reach` | 手到躯干的自我触碰或够取 |
| `M10-03` | `dressing_accessory_motion` | 穿戴整理相关的复合身体运动 |
| `M10-04` | `fine_manual_front_object` | 身前精细手部操作 |
| `M10-05` | `object_directed_limb_action` | 面向外部目标的肢体操作/打击 |
| `M10-06` | `posture_transition_locomotion` | 姿态变化与跳跃/移动 |
| `M10-07` | `symbolic_expressive_gesture` | 上肢主导的表达性动作 |
| `M10-08` | `unstable_whole_body_state` | 失衡、跌倒等全身异常轨迹 |
| `M10-09` | `interpersonal_non_aggressive` | 非攻击性人际接触或关系运动 |
| `M10-10` | `interpersonal_aggressive` | 攻击性人际动作 |

### 5.2 Trajectory 8 类

| ID | Name | 来源 |
|---|---|---|
| `M8-01` | `self_upper_body_actions` | 合并 `M10-01`、`M10-02` |
| `M8-02` | `dressing_accessory_motion` | 对应 `M10-03` |
| `M8-03` | `fine_manual_front_object` | 对应 `M10-04` |
| `M8-04` | `object_directed_limb_action` | 对应 `M10-05` |
| `M8-05` | `posture_transition_locomotion` | 对应 `M10-06` |
| `M8-06` | `symbolic_expressive_gesture` | 对应 `M10-07` |
| `M8-07` | `unstable_or_abnormal_state` | 对应 `M10-08` |
| `M8-08` | `interpersonal_actions` | 合并 `M10-09`、`M10-10` |

## 6. 两套表的关键差异

| 维度 | Semantic Coarse | Trajectory Coarse |
|---|---|---|
| 分类依据 | 行为功能、对象、语义关系 | 关节轨迹、主导部位、运动模式 |
| 解释性 | 更强 | 中等 |
| 与 early 叙事兼容性 | 更高 | 中等 |
| 与骨架输入直接对应性 | 中等 | 更高 |
| 适合用途 | 主实验、论文主叙事 | 稳健性实验、辅助分析 |

## 7. 建议实验顺序

1. 主结果先跑 `semantic coarse`
2. 消融中加入 `trajectory coarse`
3. 如果两套 coarse 标签都带来一致收益，再强调方法具有标签方案稳健性
4. 如果只有 `semantic coarse` 有用，结论偏向“高层语义辅助有效”
5. 如果两套都有效，结论可扩展为“粗粒度辅助本身有效，不局限于单一语义定义”

## 8. 配置文件位置

训练时直接加载以下机器可读配置：

- `data/label_mappings/ntu60/semantic_coarse_v1.csv`
- `data/label_mappings/ntu60/semantic_coarse_v1.json`
- `data/label_mappings/ntu60/trajectory_coarse_v1.csv`
- `data/label_mappings/ntu60/trajectory_coarse_v1.json`
- `data/label_mappings/ntu60/semantic_vs_trajectory_comparison.csv`

研究草案原始文件保留在：

- `literature/ntu60_coarse_mapping_initial.csv`
- `literature/ntu60_motion_mapping_initial.csv`
