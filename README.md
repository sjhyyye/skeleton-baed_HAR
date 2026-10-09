# Skeleton-Based HAR Acceleration

This repository is now organized around one primary research line: accelerating `SkateFormer`-style skeleton action recognition, with the current branch explicitly using `ACmix`-style shared projection as the main design reference.

The active question is no longer early action recognition from partial prefixes. The active question is whether the current `SkateFormerBlock` can be restructured for a better accuracy-latency-FLOPs Pareto frontier, while keeping the benchmark centered on standard full-sequence skeleton classification.

## Active Baseline

The baseline is the repository's original SkateFormer block with 14 selected joints,
the first person only, and NTU60 XSub. This is an adapted baseline, not the untouched
official 24-joint/two-person configuration. RCA and early-recognition objectives are disabled.
The user log reports Top-1 88.9307% at epoch 497; checkpoint verification is pending. Historical 94.23% must not be used as its accuracy.
See [baseline record](experiments/current/baseline_14p_1person_seed1/README.md).
This branch primarily records matched module improvements over that baseline.

## Current Focus

- Task: compute acceleration for full-sequence skeleton action recognition
- Backbone: `SkateFormer`
- Reference direction: `ACmix`-style shared projection plus lightweight dual aggregation
- Primary benchmark: `NTU60`
- Split order: `XSub` first, `XView` second
- Canonical inference input: `(B, C, T, V, M) = (1, 3, 64, 14, 1)`
- Primary metrics: `Top-1`, latency, throughput, `GFLOPs`, parameter count
- Main question: can we reduce real inference cost without paying an unacceptable accuracy penalty?

## Repository Map

- `SkateFormer/`
  Backbone code plus local training, evaluation, and benchmarking utilities.
- `SkateFormer/tools/benchmark_inference.py`
  Canonical local latency and FLOPs measurement entry point.
- [Experiment index](experiments/README.md) and [current results](experiments/RESULTS.md)
  Module improvements against the 14-joint, one-person NTU60 XSub baseline.
- [Historical results](experiments/archive/README.md)
  Archived pruning and early-recognition evidence, separated from current results.
- `research-plan.md`
  Main acceleration plan in English.
- `acceleration_research_plan.md`
  Detailed execution-oriented acceleration plan in Chinese.
- `research-state.yaml`
  Central project tracking state.
- `findings.md`
  Condensed current understanding and next decisions.
- `literature/`
  Survey notes for acceleration, hybrid attention-convolution design, and profiling.

## Archive Note

The previous early-recognition line remains in the repository as archived context. Its experiment folders, label-mapping files, and notes are no longer the active research plan for this branch.
