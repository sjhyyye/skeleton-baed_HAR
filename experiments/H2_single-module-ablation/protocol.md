# H2 Protocol: Single-Module Ablation

## Objective

Test each proposed modeling idea separately before training the full joint model.

## Hypothesis

At least one of `intent-only`, `consistency-only`, or `KD-only` should improve low-ratio early recognition beyond the plain prefix baseline.

## Scope

This phase isolates gain sources. It does not yet claim a final method and should avoid bundling multiple changes together prematurely.

## Inputs

- Locked `H0` protocol
- Stable `H1` prefix baseline
- NTU60 coarse label mappings under `data/label_mappings/ntu60`
- Current `SkateFormer` backbone and feeder path

## Ablation Tracks

### Track A: Intent-Only

- Add coarse semantic supervision without long-short consistency
- Start with the primary semantic mapping
- Later test whether the gain survives a second mapping

### Track B: Consistency-Only

- Add a full-sequence training-time branch or equivalent consistency target
- Keep test-time inference prefix-only
- Prefer simple logits or KL consistency before heavier variants

### Track C: KD-Only

- Compare against a straightforward teacher-guided baseline
- Use this to avoid renaming simple distillation as a novel consistency method

## Required Metrics

- Per-ratio `Top-1 Accuracy`
- Low-ratio gains at `0.1` and `0.3`
- Mean accuracy across all fixed ratios
- Optional auxiliary intent accuracy when relevant

## Experiment Matrix

1. Run `intent-only` against the locked prefix baseline.
2. Run `consistency-only` under the same protocol.
3. Run `KD-only` as the honest teacher baseline.
4. Compare gains source by source, especially at low ratios.
5. Reject redundant modules before moving to the full model.

## Acceptance Criteria

- Gain sources are separated cleanly.
- Terminology is honest if `consistency` reduces to ordinary distillation.
- At least one isolated module shows a meaningful low-ratio benefit, or the project is simplified.

## Deliverables

- Single-module ablation table
- Notes on whether intent or teacher guidance contributes more
- Decision on which modules survive into the joint model

## Risks

- Auxiliary gains may vanish once the baseline is stabilized.
- `KD-only` may absorb nearly all of the observed benefit.
- Mapping-dependent gains may weaken the intent narrative.
