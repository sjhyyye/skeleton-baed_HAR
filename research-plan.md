# Autoresearch Plan

## Project

**Title:** Operator-Level Inference Acceleration for Pruned SkateFormer

**Primary Question:** How can the current pruned `SkateFormer` implementation be accelerated at inference time through mathematically equivalent operator rewrites, without requiring full retraining?

## Executive Summary

This branch is no longer centered on block redesign as the primary research line. The active goal is operator-level acceleration: keep the functional form as unchanged as possible, rewrite expensive implementation patterns into more inference-friendly operators, and validate the gain with direct latency benchmarks plus numerical-equivalence checks.

The branch is explicitly scoped to a pruned low-`V` inference regime, with the current reference shape:

- `B=1`
- `C=192`
- `T=64`
- `V=14`

The intended contribution is not a new learning algorithm. It is a principled inference-time acceleration study for `SkateFormer`, built around exact or near-exact rewrites of the existing block computation graph.

## Locked Scope

- **Primary task:** Inference acceleration for pruned `SkateFormer`
- **Primary setting:** `B=1, C=192, T=64, V=14`
- **Primary goal:** Reduce real latency without full retraining
- **Primary method family:** operator-level equivalent rewrites
- **Primary benchmark:** latency, throughput, parameter count, optional memory footprint
- **Primary validation:** numerical equivalence or output closeness under fixed weights
- **Execution environment:** local `skateformer` Python environment with GPU benchmarking
- **Explicitly de-emphasized line:** large block redesign that requires full end-to-end retraining

## Hard Success Criteria

- **Inference honesty:** Every claimed speedup must be measured on real wall-clock latency.
- **Functional honesty:** Every operator rewrite must be checked by output-difference tests against the original implementation.
- **No hidden retraining dependency:** A result should remain meaningful even before full model retraining.
- **Operator isolation:** Each gain source must be attributable to one rewrite rather than broad architecture drift.
- **Practical relevance:** The benchmark must reflect the actual pruned target regime rather than generic large-batch training settings.

## Research Hypotheses

### H0: The Main Opportunity Is Implementation, Not New Modeling

For the current branch goal, the most actionable speedups come from rewriting operators and tensor layouts rather than inventing a new block that needs full retraining.

**Reason this matters:** The current training budget is limited, so implementation-level gains have higher immediate value.

### H1: `cat + proj` Can Be Rewritten Without Changing The Function

The current concat-projection fusion can be algebraically rewritten into a sum of branch-specific projections, removing explicit concatenation overhead while preserving the same output.

**Reason this matters:** This directly targets an existing cost center without changing model semantics.

### H2: Channel-Last `Linear` Paths Can Be Replaced By Channel-First `1x1 Conv`

The current `Linear`-based channel mixing inside the block can be rewritten as `1x1 Conv2d` under channel-first layout, reducing layout conversion overhead while preserving the same transformation.

**Reason this matters:** Many current `permute/contiguous` steps exist only to feed `Linear` and `LayerNorm` in channel-last form.

### H3: GCN Head Loops Can Be Fused

The current graph-convolution branch uses explicit Python-level chunking and looping that should be replaceable by a fused batched tensor operation.

**Reason this matters:** Removing Python control overhead and repeated small ops is especially relevant for `B=1`.

### H4: Low-`V`, Low-Batch Inference Needs Its Own Benchmark Story

Operator choices that look minor in training or large-batch settings can become important under the actual target regime of `V=14` and `B=1`.

**Reason this matters:** Benchmark conclusions from large `V` or large batch settings may not transfer.

## Near-Term Candidate Rewrites

- **Fusion rewrite:** Replace `cat + proj` with branch-wise projection summation.
- **Projection rewrite:** Replace channel-last `Linear` with channel-first `1x1 Conv2d`.
- **Normalization rewrite:** Introduce channel-first exact `LayerNorm` to reduce layout thrashing.
- **GCN rewrite:** Fuse per-head graph operations into a larger batched operation.
- **FFN rewrite:** Replace `Linear` FFN implementation with equivalent channel-first pointwise conv form for inference benchmarking.

## Phase Plan

### Phase O0: Benchmark Freeze

**Goal:** Freeze one reproducible inference benchmark for the real target shape.

**Outputs**
- Canonical benchmark shape and device setting
- Canonical timing script path
- Reporting template for latency and numerical difference

**Exit Criteria**
- Every operator rewrite is measured under the same target shape
- The branch no longer mixes block-redesign evaluation with operator-speed evaluation

### Phase O1: Baseline Operator Audit

**Goal:** Identify where operator-level speedups are most available.

**Outputs**
- Baseline latency table for the current block and key subpaths
- List of candidate exact rewrites
- Numerical validation protocol

**Exit Criteria**
- The first rewrite target is selected based on measured overhead

### Phase O2: Exact Rewrite Prototypes

**Goal:** Implement and test exact or near-exact operator rewrites.

**Outputs**
- `cat + proj` rewrite
- `Linear -> 1x1 Conv` rewrite
- Fused graph branch rewrite
- Output difference tables and latency tables

**Exit Criteria**
- At least one rewrite gives measurable speedup with negligible output difference

### Phase O3: Composition Study

**Goal:** Combine compatible rewrites into one faster inference block implementation.

**Outputs**
- Cumulative speedup table
- Numerical closeness table
- Optional memory comparison

**Exit Criteria**
- Combined implementation remains stable and meaningfully faster than baseline

### Phase O4: Escalation Decision

**Goal:** Decide whether the operator work alone is enough or whether retraining-backed structural work is still needed later.

**Outputs**
- Final operator benchmark summary
- Decision note on whether to continue into structural redesign

**Exit Criteria**
- Clear recommendation exists for the next branch or next experimental stage

## Inner-Loop Rules

- Every rewrite must preserve input/output shape exactly.
- Every rewrite must be tested with direct output comparison against the baseline block.
- Latency claims must use the actual target shape first, not only large synthetic benchmarks.
- If a rewrite changes semantics materially, it no longer counts as operator-level acceleration and must be separated from this branch.

## Immediate Next Actions

1. Freeze the operator benchmark for `B=1, C=192, T=64, V=14`.
2. Rewrite `cat + proj` into branch-wise projection summation and test exactness.
3. Rewrite `Linear` paths into channel-first `1x1 Conv2d` equivalents.
4. Fuse the graph branch head loop.
5. Measure cumulative speedup from stacking equivalent rewrites.

## Kill Criteria And Decision Points

- If a rewrite does not improve latency under the real target shape, drop it.
- If a rewrite breaks numerical equivalence beyond acceptable tolerance, treat it as structural, not operator-level.
- If operator-level rewrites plateau too early, postpone further changes until a later training-backed branch.
