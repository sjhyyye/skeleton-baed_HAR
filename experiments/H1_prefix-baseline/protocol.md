# H1 Protocol: Prefix Baseline

## Objective

Establish the first stable early-recognition baselines under the frozen `H0` protocol.

## Hypothesis

A clean `SkateFormer-prefix` baseline, plus a fair `multi-ratio` variant, is required before any auxiliary supervision claim becomes credible.

## Baseline Family

- `SkateFormer-full` as the `1.0` upper bound
- `SkateFormer-prefix` under a fixed ratio
- `SkateFormer-prefix + multi-ratio` under the same protocol

## Variables

- Observation ratio at train and test time
- Single-ratio versus multi-ratio training
- Uniform sampling versus resize-based temporal handling
- Prefix-only input without any full-sequence teacher or coarse auxiliary loss

## Required Metrics

- Per-ratio `Top-1 Accuracy`
- Mean accuracy across the fixed ratio list
- Stability across repeated evaluations on `NTU60 XSub`
- Full-observation reference accuracy for context

## Experiment Matrix

1. Run `SkateFormer-full` as the upper bound reference.
2. Run `SkateFormer-prefix` under the canonical ratio protocol.
3. Add `multi-ratio` training without changing the evaluation path.
4. Compare single-ratio and multi-ratio behavior at `0.1` and `0.3`.
5. Record the first canonical result table for `NTU60 XSub`.

## Acceptance Criteria

- The prefix baseline runs end to end.
- Per-ratio metrics are stable enough to support ablations.
- The baseline path is simple enough that later method variants can reuse it without hidden protocol changes.

## Deliverables

- Prefix baseline result table
- Multi-ratio baseline result table
- Notes on low-ratio failure modes and unstable classes

## Risks

- Multi-ratio training may obscure whether the base prefix path itself is sound.
- Full-ratio performance may look acceptable while low-ratio behavior remains poor.
- Baseline instability would invalidate later auxiliary-supervision comparisons.
