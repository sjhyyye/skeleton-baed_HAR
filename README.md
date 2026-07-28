# Skeleton-Based HAR Acceleration

This repository is now organized around one primary research line: accelerating `SkateFormer`-style skeleton action recognition, with the current branch explicitly using `ACmix`-style shared projection as the main design reference.

The active question is no longer early action recognition from partial prefixes. The active question is whether the current `SkateFormerBlock` can be restructured for a better accuracy-latency-FLOPs Pareto frontier, while keeping the benchmark centered on standard full-sequence skeleton classification.

## Current Focus

- Task: compute acceleration for full-sequence skeleton action recognition
- Backbone: `SkateFormer`
- Reference direction: `ACmix`-style shared projection plus lightweight dual aggregation
- Primary benchmark: `NTU60`
- Split order: `XSub` first, `XView` second
- Primary metrics: `Top-1`, latency, throughput, `GFLOPs`, parameter count
- Main question: can we reduce real inference cost without paying an unacceptable accuracy penalty?

## Repository Map

- `SkateFormer/`
  Backbone code plus local training, evaluation, and benchmarking utilities.
- `SkateFormer/tools/benchmark_inference.py`
  Canonical local latency and FLOPs measurement entry point.
- `experiments/acceleration_baseline_note.md`
  Acceleration-oriented baseline note, including the older pruning table and current benchmark conventions.
- `experiments/H0_protocol-freeze/`, `H1_prefix-baseline/`, `H2_single-module-ablation/`, `H3_joint-model-and-robustness/`
  Archived material from the earlier early-recognition direction. Keep for record only; do not treat as the current benchmark definition.
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
