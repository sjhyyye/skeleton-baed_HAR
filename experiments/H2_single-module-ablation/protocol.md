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

### Track A2: Intent-Improved

- Keep the original `intent-only` run as a fixed reference row
- Add a ratio-adaptive coarse-to-fine semantic variant instead of overwriting the original intent setting
- Use this track only if plain `intent-only` is too weak to justify keeping semantic supervision in the main model

### Track B: Consistency-Only

- Add a full-sequence training-time branch or equivalent consistency target
- Keep test-time inference prefix-only
- Prefer simple logits or KL consistency before heavier variants

### Track B2: Uncertainty-Aware Consistency

- Add reliability-aware gating or weighting on the full-to-prefix consistency term
- Use this track only if plain `consistency-only` is too close to ordinary distillation
- Keep the naming honest if the implementation still reduces to `KD-only`

### Track C: KD-Only

- Compare against a straightforward teacher-guided baseline
- Use this to avoid renaming simple distillation as a novel consistency method

### Track D: Optional Language Prototype Distillation

- Add training-time language prototypes only after the lighter semantic and consistency variants are understood
- Keep test-time inference skeleton-only
- Treat this as an optional enhancement line rather than a required main-track ablation

## Required Metrics

- Per-ratio `Top-1 Accuracy`
- Low-ratio gains at `0.1` and `0.3`
- Mean accuracy across all fixed ratios
- Optional auxiliary intent accuracy when relevant

## Experiment Matrix

1. Run `intent-only` against the locked prefix baseline.
2. If `intent-only` is weak, run `intent-improved` as an additional semantic refinement rather than replacing the original result row.
3. Run `consistency-only` under the same protocol.
4. Run `KD-only` as the honest teacher baseline.
5. If plain consistency collapses to ordinary distillation, run `uncertainty-aware consistency`.
6. Compare gains source by source, especially at low ratios.
7. Reject redundant modules before moving to the full model.

## Acceptance Criteria

- Gain sources are separated cleanly.
- Terminology is honest if `consistency` reduces to ordinary distillation.
- Improved semantic variants must be compared against plain `intent-only`, not substituted for it.
- At least one isolated module shows a meaningful low-ratio benefit, or the project is simplified.

## Deliverables

- Single-module ablation table
- Notes on whether intent or teacher guidance contributes more
- Notes on whether `intent-improved` or `uncertainty-aware consistency` changes the conclusion of the plain single-module runs
- Decision on which modules survive into the joint model

## Risks

- Auxiliary gains may vanish once the baseline is stabilized.
- `KD-only` may absorb nearly all of the observed benefit.
- Mapping-dependent gains may weaken the intent narrative.
