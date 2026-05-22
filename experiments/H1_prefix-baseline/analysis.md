# H1 Analysis

## Status

In Progress

## Confirmatory Results

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

- The multi-ratio checkpoint is now fully evaluated across the canonical six-ratio set, so the repository finally has one complete H1 result row for `NTU60 XSub`.
- Accuracy rises sharply from `0.1 -> 0.3` (`31.88% -> 68.92%`), which confirms that the genuinely early regime remains the hardest part of the benchmark.
- Performance is already strong by `0.5` (`85.52%`) and largely saturates by `0.9` / `1.0` (`91.44%` / `91.42%`), suggesting that most remaining ambiguity is concentrated in low-observation prefixes rather than near-complete sequences.
- The `0.9` and `1.0` results are effectively tied, so the multi-ratio model does not appear to lose obvious full-observation performance while improving the shared prefix path.

## Open Issues

- `SkateFormer-full` as a clean upper-bound reference is still missing from the H1 table.
- A current-protocol single-ratio prefix baseline is still missing, so the repository cannot yet quantify how much of the gain comes specifically from multi-ratio training.
- The low-ratio failure pattern at `0.1` is still severe enough that class-wise error analysis should be added before moving to auxiliary-supervision claims.

## Decision

Keep `H1` open. The multi-ratio baseline row is now recorded, but `H1` should only be locked after the repository also contains the matching `SkateFormer-full` and single-ratio prefix references under the same protocol.
