# Skeleton-Based HAR Acceleration

This repository is now organized around one primary research line: operator-level inference acceleration for pruned `SkateFormer`.

The active question is no longer early action recognition from partial prefixes, and it is also no longer primarily about training-backed block redesign. The current question is whether the existing implementation can be rewritten into faster inference operators without requiring full retraining.

## Current Focus

- Task: operator-level inference acceleration for pruned skeleton action recognition
- Backbone: `SkateFormer`
- Primary benchmark shape: `B=1, C=192, T=64, V=14`
- Primary metrics: latency, throughput, parameter count, numerical difference
- Main question: can we reduce real inference cost through exact or near-exact rewrites of the existing implementation?

## Repository Map

- `SkateFormer/`
  Backbone code plus local training, evaluation, and benchmarking utilities.
- `SkateFormer/tools/benchmark_inference.py`
  Canonical local latency and FLOPs measurement entry point.
- `SkateFormer/tools/benchmark_block_redesign.py`
  Existing local block benchmark utility; use only if structural comparison is still needed later.
- `experiments/acceleration_baseline_note.md`
  Acceleration-oriented baseline note, including the older pruning table and current benchmark conventions.
- `experiments/H0_protocol-freeze/`, `H1_prefix-baseline/`, `H2_single-module-ablation/`, `H3_joint-model-and-robustness/`
  Archived material from the earlier early-recognition direction. Keep for record only; do not treat as the current benchmark definition.
- `research-plan.md`
  Main operator-acceleration plan in English.
- `acceleration_research_plan.md`
  Detailed execution-oriented operator-acceleration plan in Chinese.
- `research-state.yaml`
  Central project tracking state.
- `findings.md`
  Condensed current understanding and next decisions.
- `literature/`
  Survey notes for acceleration, hybrid attention-convolution design, and profiling.

## Archive Note

The previous early-recognition line and the later block-redesign line remain in the repository as archived context. They are not the active research plan for this branch.
