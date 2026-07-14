# H1 Analysis

## Status

Substantially Complete

## Confirmatory Results

### Full Upper Bound (`SkateFormer-full`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/2026_5_13_60actions_24p/runs-495-619740.pt`
- Best epoch: `495`
- Evaluation setting: `NTU60 XSub`, full-observation train/test path
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/2026_5_13_60actions_24p/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/2026_5_13_60actions_24p/runs-495-619740_right.txt`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/2026_5_13_60actions_24p/runs-495-619740_wrong.txt`

| Train Ratio | Test Ratio | Best Top-1 | Best Top-5 |
|---|---:|---:|---:|
| `1.0` | `1.0` | `91.73%` | `98.02%` |

### Single-ratio Baseline (`SkateFormer-prefix`, `r=0.3`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r03_24p/runs-496-620992.pt`
- Best epoch: `496`
- Evaluation setting: `NTU60 XSub`, prefix-only train/test path, fixed ratio `0.3`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r03_24p/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r03_24p/runs-496-620992_right.txt`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r03_24p/runs-496-620992_wrong.txt`

| Train Ratio | Test Ratio | Best Top-1 | Best Top-5 |
|---|---:|---:|---:|
| `0.3` | `0.3` | `67.04%` | `88.50%` |

### Single-ratio Baseline (`SkateFormer-prefix`, `r=0.1`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r01_24p_rerun_20260609/runs-455-569660.pt`
- Best epoch: `455`
- Evaluation setting: `NTU60 XSub`, prefix-only train/test path, fixed ratio `0.1`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r01_24p_rerun_20260609/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r01_24p_rerun_20260609/runs-455-569660_right.txt`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_single_r01_24p_rerun_20260609/runs-455-569660_wrong.txt`

| Train Ratio | Test Ratio | Best Top-1 | Best Top-5 |
|---|---:|---:|---:|
| `0.1` | `0.1` | `27.66%` | `55.23%` |

### Multi-ratio Baseline (`SkateFormer-prefix + multi-ratio`)

- Checkpoint: `SkateFormer/work_dir/ntu/cs/SkateFormer_j/prefix_multi_24p/runs-477-597204.pt`
- Evaluation setting: `NTU60 XSub`, prefix-only test path, ratios `0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0`
- Evaluation artifacts:
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r01/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r03/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r05/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r07/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r09/`
  - `SkateFormer/work_dir/ntu/cs/SkateFormer_j/eval_prefix_multi_r10/`

| Ratio | Mean Test Loss | Top-1 | Top-5 |
|---|---:|---:|---:|
| `0.1` | `2.8388` | `31.88%` | `61.92%` |
| `0.3` | `1.6690` | `68.92%` | `91.06%` |
| `0.5` | `1.1779` | `85.52%` | `97.13%` |
| `0.7` | `1.0427` | `90.17%` | `98.26%` |
| `0.9` | `1.0105` | `91.44%` | `98.39%` |
| `1.0` | `1.0097` | `91.42%` | `98.34%` |
| **Mean** | - | **`76.56%`** | **`90.85%`** |

## Exploratory Findings

- The repository now has the clean `SkateFormer-full` upper bound for `NTU60 XSub`, so the H1 baseline family is no longer missing its full-observation reference.
- The repository now has current-protocol single-ratio prefix baselines at both `r=0.3` and `r=0.1`, so H1 is no longer missing the hardest-regime single-ratio reference.
- The multi-ratio checkpoint is now fully evaluated across the canonical six-ratio set, so the repository finally has one complete H1 result row for `NTU60 XSub`.
- At matched `r=0.1` evaluation, the single-ratio rerun reaches only `27.66%`, while the multi-ratio checkpoint reaches `31.88%`, so multi-ratio training is ahead by `+4.22` Top-1 points and `+6.69` Top-5 points in the hardest regime.
- At matched `r=0.3` evaluation, the multi-ratio checkpoint (`68.92%`) outperforms the single-ratio `r=0.3` checkpoint (`67.04%`) by `+1.88` Top-1 points, which is the first direct evidence in the repository that multi-ratio training is helping rather than simply changing the test path.
- The new `r=0.1` single-ratio result therefore removes the main ambiguity in H1: the low-ratio advantage of `prefix_multi` is real and not an artifact of comparing against an incomplete baseline family.
- Accuracy rises sharply from `0.1 -> 0.3` (`31.88% -> 68.92%`), which confirms that the genuinely early regime remains the hardest part of the benchmark.
- Performance is already strong by `0.5` (`85.52%`) and largely saturates by `0.9` / `1.0` (`91.44%` / `91.42%`), suggesting that most remaining ambiguity is concentrated in low-observation prefixes rather than near-complete sequences.
- The `1.0` result from `prefix_multi` (`91.42%`) is only `-0.31` Top-1 points below the clean `SkateFormer-full` upper bound (`91.73%`), so the prefix-based multi-ratio path does not appear to give up meaningful full-observation performance.
- Relative to the full upper bound, the remaining headroom is concentrated in the genuinely early regime: `-59.85` points at `0.1`, `-22.81` at `0.3`, and only `-0.31` at `1.0`.

### Low-ratio Failure Analysis (`r=0.1`)

- The weakest coarse semantic families at `0.1` are `health_body_state` (`18.09%` mean class accuracy), `expressive_gestures` (`18.82%`), and `reading_writing_device_use` (`28.70%`), while `interpersonal_interaction` is the strongest family (`61.30%`).
- This split suggests that the hardest early-recognition cases are actions whose defining evidence appears in small-amplitude local motion or delayed intent revelation, whereas two-person relational structure remains visible much earlier.
- The worst individual classes at `0.1` are `rub two hands together` (`8.33%`), `sneeze/cough` (`9.42%`), `touch head (headache)` (`10.51%`), `touch chest (stomachache/heart pain)` (`10.87%`), `check time (from watch)` (`11.23%`), `taking a selfie` (`11.59%`), and `make a phone call/answer phone` (`11.64%`).
- These weak classes recover sharply once more motion is observed, which argues that the low-ratio failure is mainly an early-ambiguity problem rather than a full-sequence modeling failure. For example, `rub two hands together` rises from `8.33% -> 71.74% -> 93.12%` across `0.1 -> 0.3 -> full`, `touch chest` rises from `10.87% -> 73.55% -> 95.65%`, and `taking a selfie` rises from `11.59% -> 63.77% -> 94.57%`.
- Several low-ratio confusions are semantically plausible under truncated observation. `make a phone call/answer phone` is often predicted as `playing with phone/tablet`, `drink water`, or `reach into pocket`; `taking a selfie` is often predicted as `drink water`, `tear up paper`, or `make a phone call/answer phone`; `check time (from watch)` is often predicted as `pointing to something with finger`, `salute`, or `standing up`.
- The health-state line also shows strong early confusion with generic upper-body discomfort or unstable-motion cues. `nausea or vomiting condition` is frequently predicted as `staggering` and `shake head`, while `sneeze/cough` and `touch head` are repeatedly confused with `shake head`.
- In contrast, the stronger `interpersonal_interaction` family at `0.1` implies that pairwise spatial layout and relational motion provide earlier discriminative evidence than many single-person fine-manipulation or subtle self-touch actions.

## Open Issues

- The current `r=0.1` analysis is class-level rather than sample-level because the saved evaluation artifacts expose per-class counts but not a dedicated wrong/right prediction dump for this multi-ratio evaluation path.
- The `r=0.1` single-ratio rerun is now complete, but its large gap versus `prefix_multi` still needs interpretation: whether this is an inherent advantage of multi-ratio supervision or partly a consequence of optimization instability under single-ratio ultra-low observation.

## Decision

Treat `H1` as substantially complete for baseline purposes. The repository now contains the clean `SkateFormer-full` upper bound, single-ratio prefix baselines at `r=0.1` and `r=0.3`, and the full six-ratio multi-ratio table. The main H1 conclusion is now stable: `prefix_multi` is stronger than matched single-ratio training at both `r=0.1` and `r=0.3`, with the largest benefit appearing in the hardest early regime, while preserving near-full-observation accuracy.
