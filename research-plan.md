# Autoresearch Plan

## Project

**Title:** Early Skeleton Action Recognition with Coarse Intent Supervision and Long-Short Sequence Consistency

**Primary Question:** How can a skeleton-based action-recognition model classify actions reliably from partial motion prefixes, especially at low observation ratios, without relying on frame-level stage annotation or complex sub-action parsing?

## Executive Summary

The project has pivoted away from the earlier acceleration-and-deployment plan for rehabilitation exoskeleton control. That direction is now deprecated as the main line of work. The current primary research thread is early skeleton action recognition.

The new focus is to build a clean early-recognition protocol on `NTU60`, establish a strong `SkateFormer-prefix` baseline, and then test whether a small set of lightweight ideas improve low-observation-ratio performance:

- fixed coarse intent supervision
- full-prefix consistency learning
- ratio-adaptive coarse-to-fine semantic supervision
- uncertainty-aware full-to-prefix consistency
- optional confusion-aware language prototype distillation

The intended contribution is not a heavy pipeline. The design goal is a simple and reproducible early-recognition framework that uses only action labels plus manually organized coarse intent labels, while using the full sequence only during training as a stabilizing reference.

## Locked Scope

- **Primary task:** Early skeleton action recognition
- **Primary backbone:** `SkateFormer`
- **Primary dataset:** `NTU60`
- **Primary protocols:** `XSub`, `XView`
- **Observation ratios:** `0.1`, `0.3`, `0.5`, `0.7`, `0.9`, `1.0`
- **Execution environment:** Run `SkateFormer` training and evaluation inside the `conda` environment `skateformer`
- **Primary comparison target:** Prefix-only baseline under the same early protocol
- **Main auxiliary signal:** Coarse intent labels derived from original action labels
- **Main training aid:** Full-sequence branch used only during training
- **Deferred extension:** `NTU120`
- **Explicitly deprecated main line:** Real-time acceleration / Jetson deployment / exoskeleton control optimization

## Hard Success Criteria

- **Protocol clarity:** One fixed early-recognition data protocol with reproducible prefix construction
- **Baseline clarity:** A stable `SkateFormer-prefix` baseline on `NTU60 XSub`
- **Method validity:** At least one of `intent-only`, `consistency-only`, or `full model` improves low observation ratios over the prefix baseline
- **Ablation completeness:** Clear separation of gains from `multi-ratio`, `intent`, `consistency`, and `KD-only`
- **Narrative stability:** The final claim must remain valid under at least two coarse-intent mapping schemes
- **Refinement honesty:** Any improved intent or consistency variant must be compared against the plain `intent-only` and `KD-only` references rather than replacing them silently

## Research Hypotheses

### H0: Protocol Lock

If the early-recognition protocol is frozen before large-scale experiments, later comparisons across baselines and methods will remain fair and interpretable.

**Reason this matters:** Early recognition is easy to confound if prefix cropping, interpolation, ratio scheduling, and evaluation settings drift over time.

### H1: Coarse Intent Helps Early Recognition

Coarse intent labels provide more stable high-level supervision than fine action labels in the low-observation regime and therefore improve early skeleton recognition.

**Reason this matters:** Many action pairs are visually ambiguous in the prefix stage but already separable at a coarser semantic level.

### H2: Full-Prefix Consistency Stabilizes Prefix Predictions

Using a full-sequence branch as a training-time teacher or consistency target reduces premature bias and improves prefix-stage prediction stability.

**Reason this matters:** Prefix inputs often underdetermine the final class, so a full-sequence reference may regularize the decision boundary.

### H3: Combined Intent And Consistency Matter Most At Low Ratios

The joint model will produce its clearest gains at `0.1` and `0.3`, while benefits will shrink as the observation ratio approaches full observation.

**Reason this matters:** If gains appear only at high ratios, the method is not solving the core early-recognition problem.

### H4: The Claim Survives Intent-Mapping Variation

If the method remains useful under multiple coarse-intent grouping schemes, the contribution is more likely to reflect a real modeling effect rather than a fragile manual taxonomy.

**Reason this matters:** Coarse intent labels are partly human-designed and must be stress-tested for robustness.

## Near-Term Refinement Candidates

- **Intent-improved:** Replace one fixed coarse label target with ratio-adaptive coarse-to-fine supervision so that very low ratios emphasize stable semantics while higher ratios shift weight toward fine action discrimination.
- **Consistency-improved:** Replace plain full-to-prefix KL with uncertainty-aware consistency so that the full branch teaches most strongly when its own prediction is reliable.
- **Optional advanced variant:** Add confusion-aware language prototype distillation only if the lighter semantic and consistency variants have already produced a stable `NTU60` story.

## Phase Plan

### Phase 0: Protocol Freeze

**Goal:** Lock the early-recognition setting before committing to major training runs.

**Outputs**
- Written definition of valid-frame counting and prefix cropping
- Written interpolation policy and fixed sequence length
- Fixed observation-ratio list
- Reproducible baseline command path for `NTU60 XSub`

**Exit Criteria**
- Prefix construction is unambiguous
- Random crop and prefix crop are not mixed anywhere in the evaluation path
- One baseline training path is identified as canonical

### Phase 1: Prefix Baseline

**Goal:** Build the reference early-recognition baseline.

**Outputs**
- `SkateFormer-full` upper bound
- `SkateFormer-prefix` baseline
- `SkateFormer-prefix + multi-ratio` baseline
- First per-ratio result table on `NTU60 XSub`

**Exit Criteria**
- The baseline runs end to end
- Per-ratio metrics are stable enough to support ablations

### Phase 2: Single-Module Validation

**Goal:** Measure each idea separately before combining them.

**Outputs**
- `intent-only`
- `intent-improved` if plain intent supervision is too weak
- `consistency-only`
- `uncertainty-aware consistency` if plain consistency collapses to ordinary KD
- `KD-only` or equivalent teacher-guided baseline
- Optional `language-prototype distillation`
- Comparison table against prefix and multi-ratio baselines

**Exit Criteria**
- Gain sources are separated cleanly
- Terminology between `consistency` and `KD` is not redundant or ambiguous

### Phase 3: Full Model

**Goal:** Test the joint method.

**Outputs**
- Joint `intent + consistency` model, using either the plain modules or the stronger refined variants
- Main result table across observation ratios
- Error analysis by action type and confusion pattern

**Exit Criteria**
- The joint model outperforms the strongest fair baseline on the key low-ratio regime, or the method is revised

### Phase 4: Robustness And Extension

**Goal:** Stress-test the claim and prepare the paper narrative.

**Outputs**
- Results under at least two coarse-intent mappings
- `NTU60 XView` validation
- Optional `NTU120` extension
- Final figures, tables, and writing backbone

**Exit Criteria**
- The paper story does not depend on one arbitrary mapping or one split

## Inner-Loop Rules

- Every experiment must report per-ratio `Top-1 Accuracy`
- Low-ratio performance is the primary decision criterion, not full-ratio accuracy alone
- The full-sequence branch must not add test-time inference cost
- Prefix construction must be identical across all compared methods
- If `consistency` is implemented as simple distillation, name it honestly and avoid inflated novelty claims
- Claims about intent supervision must be checked against more than one label grouping
- Plain `intent-only` remains a required reference even if `intent-improved` is later added
- Optional language-based supervision is training-time only unless a stronger justification emerges

## Outer-Loop Triggers

Run an outer-loop synthesis when any of the following happens:

- The prefix baseline is stable on `NTU60 XSub`
- A single-module ablation shows a clear and repeatable gain
- The full model fails to beat a simpler baseline
- Intent-mapping sensitivity becomes the dominant uncertainty
- `NTU60 XSub` and `XView` begin to tell different stories

## Immediate Next Actions

1. Freeze the early-recognition protocol on `NTU60`.
2. Build at least two coarse-intent mapping schemes.
3. Implement and run the canonical `SkateFormer-prefix` baseline on `XSub`.
4. Add `multi-ratio` training under the same protocol.
5. Add `intent-only`, `consistency-only`, and `KD-only` before training the full joint model.
6. If plain `intent-only` remains weak, test `intent-improved` as a ratio-adaptive semantic refinement instead of overwriting the original baseline.
7. If plain `consistency-only` reduces to ordinary distillation, test an uncertainty-aware variant before claiming a distinct consistency contribution.

## Kill Criteria And Decision Points

- If the prefix baseline cannot be reproduced cleanly, stop method design and fix the protocol first.
- If coarse intent helps only under one fragile mapping, reduce the claim and reposition the contribution.
- If `KD-only` already captures nearly all gains, simplify the method and drop redundant modules.
- If the full model does not beat the best simpler baseline at low ratios, do not force the combined story.
- If `NTU60` does not support a stable early-recognition narrative, do not expand to `NTU120` prematurely.
