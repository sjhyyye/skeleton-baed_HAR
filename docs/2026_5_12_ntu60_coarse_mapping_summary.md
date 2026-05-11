# 2026_5_12 NTU60 高层动作分类总结

## 结论

- 目前已经为 `NTU60` 的 `60` 个动作类整理出两套可直接用于实验的高层标签方案：`semantic coarse` 和 `trajectory coarse`。
- `semantic coarse` 更适合作为主实验标签。它按动作功能、交互对象和行为语义归类，更容易支撑论文中的 `coarse intent / coarse action family` 叙事。
- `trajectory coarse` 更适合作为稳健性对照标签。它按骨架主导部位和变化轨迹归类，更接近骨架模型实际看到的运动模式。
- 两套方案都同时提供 `10 类` 和 `8 类` 版本，且 `8 类` 由 `10 类` 合并得到，层级关系明确，适合做主实验与稳健性实验的对照。
- 从类别分布上看，`semantic coarse` 中最集中的类别是 `expressive_gestures = 10`、`health_body_state = 9`、`dressing_accessories = 8`、`interpersonal_interaction = 8`；`trajectory coarse` 中最集中的类别是 `symbolic_expressive_gesture = 10`，其次是 `hand_to_head_face = 8`、`fine_manual_front_object = 8`、`dressing_accessory_motion = 8`、`interpersonal_non_aggressive = 8`。
- 当前最合理的用法是：先用 `semantic coarse` 跑 `intent-only` 主实验，再用 `trajectory coarse` 验证收益是否对标签定义稳健。

## 简短操作过程

1. 基于本地 `NTU60` 官方类名，先对 `60` 个动作类逐类整理高层标签。
2. 第一套标签按语义功能、作用对象和人际关系定义，形成 `semantic coarse`。
3. 第二套标签按主导骨骼部位和运动轨迹模式定义，形成 `trajectory coarse`。
4. 每套标签先构造 `10 类` 版本，再向上合并为 `8 类` 版本，保证层级一致。
5. 将两套映射分别导出为研究草案、对照说明文档和代码可直接加载的 `csv/json` 配置。

## 完整统计表格

### Semantic Coarse 10 类

| 类别 ID | 类别名 | 数量 | 成员动作 ID |
|---|---|---:|---|
| `C10-01` | `eat_drink` | 2 | `1, 2` |
| `C10-02` | `personal_grooming` | 3 | `3, 4, 37` |
| `C10-03` | `dressing_accessories` | 8 | `14, 15, 16, 17, 18, 19, 20, 21` |
| `C10-04` | `object_manipulation` | 6 | `5, 6, 7, 13, 24, 25` |
| `C10-05` | `posture_locomotion` | 4 | `8, 9, 26, 27` |
| `C10-06` | `expressive_gestures` | 10 | `10, 22, 23, 31, 34, 35, 36, 38, 39, 40` |
| `C10-07` | `reading_writing_device_use` | 7 | `11, 12, 28, 29, 30, 32, 33` |
| `C10-08` | `health_body_state` | 9 | `41, 42, 43, 44, 45, 46, 47, 48, 49` |
| `C10-09` | `interpersonal_interaction` | 8 | `53, 54, 55, 56, 57, 58, 59, 60` |
| `C10-10` | `aggressive_interaction` | 3 | `50, 51, 52` |

### Semantic Coarse 8 类

| 类别 ID | 类别名 | 数量 | 来源 |
|---|---|---:|---|
| `C8-01` | `self_care_and_dressing` | 13 | `C10-01 + C10-02 + C10-03` |
| `C8-02` | `object_manipulation` | 6 | `C10-04` |
| `C8-03` | `posture_locomotion` | 4 | `C10-05` |
| `C8-04` | `expressive_gestures` | 10 | `C10-06` |
| `C8-05` | `reading_writing_device_use` | 7 | `C10-07` |
| `C8-06` | `health_body_state` | 9 | `C10-08` |
| `C8-07` | `interpersonal_interaction` | 8 | `C10-09` |
| `C8-08` | `aggressive_interaction` | 3 | `C10-10` |

### Trajectory Coarse 10 类

| 类别 ID | 类别名 | 数量 | 成员动作 ID |
|---|---|---:|---|
| `M10-01` | `hand_to_head_face` | 8 | `1, 2, 3, 4, 37, 41, 44, 49` |
| `M10-02` | `torso_self_touch_reach` | 5 | `25, 45, 46, 47, 48` |
| `M10-03` | `dressing_accessory_motion` | 8 | `14, 15, 16, 17, 18, 19, 20, 21` |
| `M10-04` | `fine_manual_front_object` | 8 | `11, 12, 13, 28, 29, 30, 32, 33` |
| `M10-05` | `object_directed_limb_action` | 4 | `5, 6, 7, 24` |
| `M10-06` | `posture_transition_locomotion` | 4 | `8, 9, 26, 27` |
| `M10-07` | `symbolic_expressive_gesture` | 10 | `10, 22, 23, 31, 34, 35, 36, 38, 39, 40` |
| `M10-08` | `unstable_whole_body_state` | 2 | `42, 43` |
| `M10-09` | `interpersonal_non_aggressive` | 8 | `53, 54, 55, 56, 57, 58, 59, 60` |
| `M10-10` | `interpersonal_aggressive` | 3 | `50, 51, 52` |

### Trajectory Coarse 8 类

| 类别 ID | 类别名 | 数量 | 来源 |
|---|---|---:|---|
| `M8-01` | `self_upper_body_actions` | 13 | `M10-01 + M10-02` |
| `M8-02` | `dressing_accessory_motion` | 8 | `M10-03` |
| `M8-03` | `fine_manual_front_object` | 8 | `M10-04` |
| `M8-04` | `object_directed_limb_action` | 4 | `M10-05` |
| `M8-05` | `posture_transition_locomotion` | 4 | `M10-06` |
| `M8-06` | `symbolic_expressive_gesture` | 10 | `M10-07` |
| `M8-07` | `unstable_or_abnormal_state` | 2 | `M10-08` |
| `M8-08` | `interpersonal_actions` | 11 | `M10-09 + M10-10` |

## 相关文件位置

- 对照说明文档： [ntu60_coarse_mapping_comparison.md](/data00/home/sjh/skeleton-based-har/literature/ntu60_coarse_mapping_comparison.md:1)
- Semantic 草案表： [ntu60_coarse_mapping_initial.csv](/data00/home/sjh/skeleton-based-har/literature/ntu60_coarse_mapping_initial.csv:1)
- Trajectory 草案表： [ntu60_motion_mapping_initial.csv](/data00/home/sjh/skeleton-based-har/literature/ntu60_motion_mapping_initial.csv:1)
- Semantic 训练配置： [semantic_coarse_v1.csv](/data00/home/sjh/skeleton-based-har/data/label_mappings/ntu60/semantic_coarse_v1.csv:1)
- Semantic JSON 配置： [semantic_coarse_v1.json](/data00/home/sjh/skeleton-based-har/data/label_mappings/ntu60/semantic_coarse_v1.json:1)
- Trajectory 训练配置： [trajectory_coarse_v1.csv](/data00/home/sjh/skeleton-based-har/data/label_mappings/ntu60/trajectory_coarse_v1.csv:1)
- Trajectory JSON 配置： [trajectory_coarse_v1.json](/data00/home/sjh/skeleton-based-har/data/label_mappings/ntu60/trajectory_coarse_v1.json:1)
- 两套映射对照表： [semantic_vs_trajectory_comparison.csv](/data00/home/sjh/skeleton-based-har/data/label_mappings/ntu60/semantic_vs_trajectory_comparison.csv:1)
