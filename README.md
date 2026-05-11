# Early Skeleton Action Recognition

This repository is organized around one primary research line: early skeleton action recognition with `SkateFormer` on `NTU60`, using coarse-grained auxiliary supervision and long-short sequence consistency as the main modeling ideas.

The current project phase is protocol freeze and baseline construction, not large-scale result reporting. The first priority is to lock a clean prefix-based evaluation path, establish a stable `SkateFormer-prefix` baseline, and then test `intent-only`, `consistency-only`, and `KD-only` separately before training a joint model.

## Current Focus

- Task: early skeleton action recognition from partial prefixes
- Backbone: `SkateFormer`
- Primary benchmark: `NTU60`
- Split order: `XSub` first, `XView` second
- Observation ratios: `0.1`, `0.3`, `0.5`, `0.7`, `0.9`, `1.0`
- Main question: can coarse semantic supervision and training-time full-sequence guidance improve low-ratio recognition without adding test-time complexity?

## Repository Map

- `SkateFormer/`
  Upstream backbone code plus local dataset/config changes used for the current project.
- `data/label_mappings/ntu60/`
  Machine-readable coarse supervision mappings for NTU60.
- `experiments/H0_protocol-freeze/`
  Canonical early-recognition protocol freeze notes.
- `experiments/H1_prefix-baseline/`
  Prefix and multi-ratio baseline stage.
- `experiments/H2_single-module-ablation/`
  `intent-only`, `consistency-only`, and `KD-only` validation stage.
- `experiments/H3_joint-model-and-robustness/`
  Full-model and robustness stage.
- `experiments/legacy_engineering_optimization.md`
  Short historical note for the older pruning/compression/deployment thread.
- `research-plan.md`
  Main project plan.
- `research-state.yaml`
  Central project tracking state.
- `findings.md`
  Condensed current understanding.
- `literature/`
  Survey notes and label-mapping drafts.

## Historical Note

An older engineering-oriented line on joint pruning, compression, and deployment remains in the repository as background only. It is not the active benchmark definition, not the current success criterion, and should not drive the main paper narrative unless the project explicitly pivots back.
