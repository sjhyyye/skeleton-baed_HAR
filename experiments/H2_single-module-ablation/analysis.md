# H2 Analysis

## Status

In Progress

## Confirmatory Results

### Intent-Only (`SkateFormer-prefix + semantic coarse intent`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/intent_semantic_multi_24p/runs-499-624748.pt`
- Best epoch: `499`
- Training setting: `NTU60 XSub`, multi-ratio prefix training on `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Intent mapping: `data/label_mappings/ntu60/semantic_coarse_v1.json` with key `ntu60_to_coarse10_index`
- Evaluation setting: `NTU60 XSub`, prefix-only test path, ratios `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r01/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r03/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r05/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r07/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r09/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_semantic_multi_r10/`

| Ratio | Mean Test Loss | Intent Acc | Top-1 | Top-5 | Delta vs `prefix_multi` Top-1 |
|---|---:|---:|---:|---:|---:|
| `0.1` | `3.6219` | `52.41%` | `31.69%` | `61.29%` | `-0.19` |
| `0.3` | `2.1805` | `79.62%` | `67.95%` | `90.94%` | `-0.97` |
| `0.5` | `1.5294` | `91.91%` | `85.35%` | `97.22%` | `-0.17` |
| `0.7` | `1.3613` | `94.79%` | `90.08%` | `98.29%` | `-0.09` |
| `0.9` | `1.3200` | `95.48%` | `91.11%` | `98.53%` | `-0.33` |
| `1.0` | `1.3133` | `95.62%` | `91.41%` | `98.62%` | `-0.01` |
| **Mean** | **`1.8877`** | **`84.97%`** | **`76.27%`** | **`90.81%`** | **`-0.29`** |

### Intent-Only (`SkateFormer-prefix + trajectory coarse intent`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/intent_trajectory_multi_24p/runs-500-626000.pt`
- Best epoch: `500`
- Training setting: `NTU60 XSub`, multi-ratio prefix training on `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Intent mapping: `data/label_mappings/ntu60/trajectory_coarse_v1.json` with key `ntu60_to_motion10_index`
- Evaluation setting: `NTU60 XSub`, prefix-only test path, ratios `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r01/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r03/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r05/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r07/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r09/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_multi_r10/`

| Ratio | Mean Test Loss | Intent Acc | Top-1 | Top-5 | Delta vs `prefix_multi` Top-1 | Delta vs semantic intent-only Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `0.1` | `3.6151` | `51.99%` | `31.32%` | `61.76%` | `-0.56` | `-0.37` |
| `0.3` | `2.1522` | `81.15%` | `68.37%` | `91.30%` | `-0.55` | `+0.42` |
| `0.5` | `1.5172` | `92.88%` | `85.42%` | `97.43%` | `-0.10` | `+0.07` |
| `0.7` | `1.3522` | `95.43%` | `90.08%` | `98.33%` | `-0.09` | `+0.00` |
| `0.9` | `1.3122` | `96.14%` | `91.25%` | `98.56%` | `-0.19` | `+0.14` |
| `1.0` | `1.3037` | `96.30%` | `91.54%` | `98.51%` | `+0.12` | `+0.13` |
| **Mean** | **`1.8754`** | **`85.65%`** | **`76.33%`** | **`90.98%`** | **`-0.23`** | **`+0.06`** |

### Intent-Only (`SkateFormer-prefix + early-observable grouping v2`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/intent_earlyobs_v2_multi_24p/runs-499-624748.pt`
- Best epoch: `499`
- Training setting: `NTU60 XSub`, multi-ratio prefix training on `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Intent mapping: `data/label_mappings/ntu60/early_observable_v2.json` with key `ntu60_to_earlyobs10_index`
- Evaluation setting: `NTU60 XSub`, prefix-only test path, ratios `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r01/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r03/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r05/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r07/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r09/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_earlyobs_v2_multi_r10/`

| Ratio | Mean Test Loss | Intent Acc | Top-1 | Top-5 | Delta vs `prefix_multi` Top-1 | Delta vs trajectory intent-only Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `0.1` | `3.6101` | `54.10%` | `31.53%` | `61.42%` | `-0.35` | `+0.21` |
| `0.3` | `2.1392` | `82.85%` | `68.33%` | `91.24%` | `-0.59` | `-0.04` |
| `0.5` | `1.5047` | `93.59%` | `85.46%` | `97.42%` | `-0.06` | `+0.04` |
| `0.7` | `1.3468` | `95.82%` | `90.27%` | `98.34%` | `+0.10` | `+0.19` |
| `0.9` | `1.3065` | `96.40%` | `91.39%` | `98.50%` | `-0.05` | `+0.14` |
| `1.0` | `1.3031` | `96.40%` | `91.55%` | `98.53%` | `+0.13` | `+0.01` |
| **Mean** | **`1.8684`** | **`86.53%`** | **`76.42%`** | **`90.91%`** | **`-0.14`** | **`+0.09`** |

### Intent-Only Comparison Summary

| Ratio | `prefix_multi` Top-1 | semantic intent-only Top-1 | trajectory intent-only Top-1 | `early_observable_v2` intent-only Top-1 | EarlyObs v2 vs `prefix_multi` | EarlyObs v2 vs trajectory |
|---|---:|---:|---:|---:|---:|---:|
| `0.1` | `31.88%` | `31.69%` | `31.32%` | `31.53%` | `-0.35` | `+0.21` |
| `0.3` | `68.92%` | `67.95%` | `68.37%` | `68.33%` | `-0.59` | `-0.04` |
| `0.5` | `85.52%` | `85.35%` | `85.42%` | `85.46%` | `-0.06` | `+0.04` |
| `0.7` | `90.17%` | `90.08%` | `90.08%` | `90.27%` | `+0.10` | `+0.19` |
| `0.9` | `91.44%` | `91.11%` | `91.25%` | `91.39%` | `-0.05` | `+0.14` |
| `1.0` | `91.42%` | `91.41%` | `91.54%` | `91.55%` | `+0.13` | `+0.01` |
| **Mean** | **`76.56%`** | **`76.27%`** | **`76.33%`** | **`76.42%`** | **`-0.14`** | **`+0.09`** |

### Intent-Improved (`trajectory coarse intent + ratio-adaptive lambda + intent-to-action bias/gate`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/intent_trajectory_adaptive_gated_multi_24p/runs-499-624748.pt`
- Best epoch: `499`
- Training setting: `NTU60 XSub`, multi-ratio prefix training on `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Intent mapping: `data/label_mappings/ntu60/trajectory_coarse_v1.json` with key `ntu60_to_motion10_index`
- Intent weighting: ratio-adaptive `lambda_intent = {0.1: 1.0, 0.3: 0.8, 0.5: 0.5, 0.7: 0.3, 0.9: 0.1, 1.0: 0.0}`
- Intent conditioning: `bias_gate` from intent probabilities into action logits
- Evaluation setting: `NTU60 XSub`, prefix-only test path, targeted ratios `0.1 / 0.3 / 0.5`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_adaptive_gated_multi_r01/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_adaptive_gated_multi_r03/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_intent_trajectory_adaptive_gated_multi_r05/`

| Ratio | Mean Test Loss | Intent Acc | Intent Lambda | Top-1 | Top-5 | Delta vs `prefix_multi` Top-1 | Delta vs trajectory intent-only Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.1` | `4.3965` | `52.17%` | `1.0000` | `31.30%` | `61.50%` | `-0.58` | `-0.02` |
| `0.3` | `2.4131` | `81.53%` | `0.8000` | `68.48%` | `91.34%` | `-0.44` | `+0.11` |
| `0.5` | `1.5039` | `92.76%` | `0.5000` | `85.53%` | `97.47%` | `+0.01` | `+0.11` |
| **Mean** | **`2.7712`** | **`75.49%`** | **`0.7667`** | **`61.77%`** | **`83.44%`** | **`-0.34`** | **`+0.07`** |

## Exploratory Findings

- The repository now has three complete fixed `intent-only` rows under the same six-ratio protocol used by `H1`: semantic coarse, trajectory coarse, and `early_observable_v2`.
- The fixed-taxonomy story is now clearer. Trajectory mapping is a better implementation than the original semantic mapping, and `early_observable_v2` is the best fixed intent taxonomy tried so far on average: mean Top-1 rises from `76.27%` (semantic) to `76.33%` (trajectory) to `76.42%` (`early_observable_v2`).
- `early_observable_v2` also gives the strongest auxiliary intent prediction among the fixed rows, with mean intent accuracy `86.53%` and peak intent accuracy `96.40%` at both `0.9` and `1.0`.
- The gain from `early_observable_v2` over trajectory is real but small: `+0.21` at `0.1`, `-0.04` at `0.3`, `+0.04` at `0.5`, `+0.19` at `0.7`, `+0.14` at `0.9`, and `+0.01` at `1.0`.
- The key low-ratio result still remains negative against the main baseline. At `0.1`, `early_observable_v2` reaches `31.53%`, below `prefix_multi` at `31.88%`. At `0.3`, it reaches `68.33%`, still below `68.92%`.
- From `0.5` upward, the gaps to `prefix_multi` become negligible: `-0.06`, `+0.10`, `-0.05`, and `+0.13`. This shows that better grouping helps make intent supervision less wasteful, but not that it solves the genuinely early ambiguity.
- The current fixed intent comparison therefore supports a tighter claim: better grouping helps a little, and trajectory-style grouping is directionally right, but intent taxonomy refinement alone is still insufficient as the main path.
- A first `intent-improved` variant has now been tested on the stronger trajectory mapping with ratio-adaptive intent weights and direct intent-to-action bias/gating. It produces small gains at `0.3` and `0.5` relative to fixed trajectory intent-only (`+0.11` / `+0.11`), but is slightly worse at `0.1` (`-0.02`).
- The key low-ratio conclusion therefore does not change. Even after adaptive weighting and classifier conditioning, the model still trails `prefix_multi` at `0.1` (`31.30%` vs `31.88%`) and `0.3` (`68.48%` vs `68.92%`). It only reaches a negligible edge at `0.5` (`85.53%` vs `85.52%`).
- The current H2 evidence therefore supports a stronger judgment than before: `trajectory_coarse_v1` was a better intent baseline than semantic grouping, `early_observable_v2` is the best fixed taxonomy so far, and the adaptive gated refinement is a mild mid-ratio improvement on top of the trajectory row, but the entire intent line still should not be promoted as the main path because it does not beat `prefix_multi` where the benchmark is hardest.

## Planned Refinement Variants

### Intent-Improved

- Keep the current fixed semantic `intent-only` row as the semantic reference, the fixed trajectory row as the original motion-aware reference, and the fixed `early_observable_v2` row as the strongest taxonomy-only reference.
- The first trajectory-based `intent-improved` variant is now complete at `0.1 / 0.3 / 0.5`, and it does not solve the low-ratio problem.
- Do not spend another iteration on intent taxonomy or lightweight gating unless teacher-guided baselines also fail; the next comparison pressure should move to `KD-only` and `consistency-only`.

### Consistency-Improved

- First run plain `consistency-only` and `KD-only`.
- If plain consistency is too close to ordinary distillation, add an uncertainty-aware consistency variant as a refined follow-up rather than claiming novelty too early.
- The key question is whether reliability-aware teacher guidance beats both plain `KD-only` and fixed `intent-only` in the low-ratio regime.

### Optional Semantic Enhancement

- If the lighter semantic and teacher-guided variants stabilize, add `confusion-aware language prototype distillation` as an optional training-time enhancement.
- This line should remain secondary until `NTU60 XSub` has a clean semantic-versus-teacher story.

## Open Issues

- Whether any further intent refinement can recover a real low-ratio gain once semantic, trajectory, `early_observable_v2`, and adaptive-gated variants have all failed to beat `prefix_multi` at `0.1 / 0.3`
- Whether consistency adds value beyond a simple teacher baseline
- Whether uncertainty-aware consistency provides a real gain beyond plain `KD-only`
- Whether the remaining gap at `0.1 / 0.3` is best attacked by teacher guidance rather than another intent taxonomy

## Decision

Keep `H2` open. The intent line has now been evaluated in four forms: semantic fixed intent-only, trajectory fixed intent-only, `early_observable_v2` fixed intent-only, and trajectory adaptive-gated intent-improved. The conclusion is now stable: trajectory mapping is a better implementation of semantic mapping, `early_observable_v2` is the best fixed taxonomy tried so far, but none of these intent-driven variants beats `prefix_multi` on the decisive `0.1 / 0.3` comparisons. Do not promote the joint model on the basis of intent supervision alone. Treat `early_observable_v2` as the strongest taxonomy reference, keep the adaptive-gated row as the best current mechanism tweak, and move comparison pressure to `consistency-only` and `KD-only`.
