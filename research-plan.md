# Autoresearch Plan

## Project

**Title:** ACmix-Inspired Compute Acceleration for SkateFormer-Based Skeleton Action Recognition

**Primary Question:** How can the current `SkateFormer` backbone be restructured to reduce real inference cost, especially the dominant channel-mixing and multi-branch aggregation overhead, while preserving strong full-sequence recognition accuracy?

## Executive Summary

The branch has pivoted back to computation acceleration as the primary line of work. The earlier early-recognition plan is now archived for this branch and should not define the benchmark, success criteria, or method design.

The new focus is to treat `SkateFormer` as the baseline system, profile where the current block spends computation, and then design an `ACmix`-inspired replacement that shares feature generation across attention-like and convolution-like aggregation paths. The goal is not to paste an image-model block blindly into a skeleton model. The goal is to use the `ACmix` design principle to build a skeleton-specific acceleration path that is measurable on real latency, not only on paper FLOPs.

The first active plan is to:

- freeze one canonical compute benchmark on `NTU60`
- profile the current `SkateFormerBlock`
- redesign the `SkateFormerBlock` around a partition-free and branch-collapsed mixer
- compare stage-wise replacement variants
- validate a real Pareto improvement under matched accuracy evaluation

The current paper-facing hypothesis is now more specific than a generic `ACmix` transplant. For the pruned small-joint regime, especially `B=1` and `V=14`, the most defensible contribution is to remove explicit `partition -> reverse -> cat -> proj` style structure as a block-design principle, not merely to micro-optimize one tensor op.

## Locked Scope

- **Primary task:** Full-sequence skeleton action recognition with explicit compute acceleration
- **Primary backbone:** `SkateFormer`
- **Primary dataset:** `NTU60`
- **Primary protocols:** `XSub`, `XView`
- **Primary input convention:** `T=64`, `V=25`, `M=2` unless a pruning variant explicitly changes it
- **Execution environment:** Run `SkateFormer` training and evaluation inside the `conda` environment `skateformer`
- **Primary comparison target:** Unmodified `SkateFormer` under matched data and benchmark settings
- **Primary reference paper:** `On the Integration of Self-Attention and Convolution (ACmix)`
- **Primary outputs:** model variants, latency tables, FLOPs tables, parameter tables, accuracy tables
- **Deferred extension:** `NTU120`, device-specific deployment work, quantization
- **Explicitly archived line:** Early skeleton action recognition from partial prefixes

## Hard Success Criteria

- **Benchmark clarity:** One fixed accuracy path and one fixed inference-benchmark path
- **Profiling clarity:** The dominant cost inside the current block is measured before redesign claims are made
- **Method validity:** At least one acceleration variant improves the accuracy-latency or accuracy-FLOPs Pareto frontier over baseline
- **Ablation completeness:** Gains from block redesign, stage-wise replacement, width reduction, and optional pruning are separated cleanly
- **Latency honesty:** Real wall-clock speedup must be reported alongside FLOPs
- **Narrative discipline:** If a variant only reduces FLOPs but not latency, the claim must be narrowed accordingly

## Research Hypotheses

### H0: Benchmark Freeze

If the accuracy protocol and inference benchmark are frozen before major redesign, later claims about acceleration will remain fair and interpretable.

**Reason this matters:** Acceleration results are easy to distort when batch size, device, input shape, warmup, or checkpoint quality drift across runs.

### H1: ACmix-Style Shared Projection Can Improve The Pareto Frontier

The current block can be redesigned around a more aggressive shared-projection budget so that attention-like and convolution-like aggregation reuse the same intermediate features and reduce total compute at comparable accuracy.

**Reason this matters:** The current model already mixes graph, temporal, and attention branches, so the main opportunity is not adding more branches, but simplifying how expensive channel mixing is produced and consumed.

### H2: Partial Replacement Will Beat Full Replacement Early

Replacing only the most expensive or least cost-effective stages will produce a better early Pareto frontier than replacing every block at once.

**Reason this matters:** Different stages may have different sensitivity to attention-range modeling versus local aggregation.

### H3: Distillation Will Be More Useful Than Architectural Over-Expansion

If the accelerated student loses noticeable accuracy, a teacher-guided recovery path will be more efficient than adding back heavy modules.

**Reason this matters:** The branch objective is compute reduction, not designing a larger hybrid than the original model.

### H4: FLOPs Gains Must Translate To Real Latency Gains

Some theoretically cheaper variants will not speed up real inference because partition, reshape, memory movement, or kernel-launch overhead dominates.

**Reason this matters:** The final claim should be about usable acceleration, not only symbolic arithmetic savings.

### H5: Block Organization, Not A Single Operator, Is The Core Problem

For the pruned low-`V` inference regime, the main `SkateFormer` inefficiency is better described as a block-organization problem than as one isolated kernel bottleneck. A partition-free and branch-collapsed redesign should therefore be more paper-worthy than only tuning attention count or MLP width.

**Reason this matters:** Changing `attention` count or `mlp_ratio` is useful engineering, but the more defensible research claim is that explicit multi-partition branch materialization and concat-projection fusion become structurally inefficient after pruning.

## Near-Term Refinement Candidates

- **Block redesign:** Replace the current explicit multi-partition branch materialization with a partition-free and branch-collapsed mixer.
- **Fusion redesign:** Replace `cat + proj` with additive, gated, or low-rank fusion so the block no longer depends on large explicit branch concatenation.
- **Shared projection:** Keep the `ACmix` lesson as a supporting design principle, but do not present shared projection alone as the main novelty claim.
- **Stage-wise replacement:** Test early-only, late-only, and all-stage replacement schedules.
- **Width refinement:** Reduce head count, branch width, or MLP expansion only after the new block is stable.
- **Teacher recovery:** Add baseline-to-student distillation only if the accelerated block shows a promising compute gain but an avoidable accuracy drop.
- **Structural extension:** Revisit joint-pruning combinations only after the block-level story is stable.

## Phase Plan

### Phase A0: Benchmark And Profiling Freeze

**Goal:** Lock the canonical compute benchmark before redesigning the model.

**Outputs**
- One canonical training/evaluation path for baseline accuracy
- One canonical latency/FLOPs benchmarking command path
- Fixed benchmark input shape and reporting template
- Initial module-level profiling notes for the current block

**Exit Criteria**
- Speed and accuracy numbers are reproducible under one fixed setup
- The benchmark no longer mixes archived early-recognition language with active acceleration language
- The most expensive block components are explicitly identified

### Phase A1: Baseline Cost Audit

**Goal:** Establish the baseline Pareto reference.

**Outputs**
- Baseline `Top-1`, latency, throughput, `GFLOPs`, parameter count
- Stage-level or block-level profiling breakdown
- Width/head/MLP sensitivity notes if cheap to collect
- A paper-facing diagnosis of whether the core issue is operator cost or block organization

**Exit Criteria**
- The baseline is measured end to end
- The first optimization target is chosen based on profiling rather than intuition alone

### Phase A2: ACmix-Style Prototype

**Goal:** Validate a first accelerated block design.

**Outputs**
- One `SkateFormer` variant with a partition-free / branch-collapsed mixed block
- Matched benchmark table against baseline
- Stability notes on training, memory, and implementation complexity

**Exit Criteria**
- The prototype shows either a promising Pareto gain or a clear failure mode worth revising
- The redesigned block does not silently increase hidden overhead elsewhere

### Phase A3: Systematic Ablation

**Goal:** Separate where the gains really come from.

**Outputs**
- Stage-wise replacement table
- Branch-width or head-count ablation
- Optional distillation recovery table
- Accuracy-latency Pareto plot

**Exit Criteria**
- The best accelerated variant is identified with a clean rationale
- The claim is no longer dependent on one arbitrary architecture tweak

### Phase A4: Robustness And Extension

**Goal:** Check whether the best variant survives outside the first narrow setup.

**Outputs**
- `NTU60 XView` validation
- Optional `NTU120` or alternate joint-count validation
- Optional pruning-plus-architecture combination study
- Final figures, tables, and writing backbone

**Exit Criteria**
- The acceleration story does not depend on one device, one split, or one misleading metric

## Inner-Loop Rules

- Every reported variant must include `Top-1`, latency, throughput, `GFLOPs`, and parameter count if measurable
- Real latency is the primary decision criterion; FLOPs alone are insufficient
- Accuracy comparisons must use the same data path and training budget unless clearly marked exploratory
- A speedup claim is invalid if it depends on a weaker checkpoint or a changed input shape
- If a method needs distillation to recover accuracy, that dependency must be stated explicitly
- Archived early-recognition files must not be cited as active evidence for this branch

## Outer-Loop Triggers

Run an outer-loop synthesis when any of the following happens:

- The baseline profile is frozen
- A prototype block shows a repeatable latency gain
- FLOPs and latency tell conflicting stories
- Partial replacement beats full replacement decisively
- `NTU60 XSub` and `XView` begin to tell different Pareto stories

## Immediate Next Actions

1. Freeze the baseline accuracy config and the benchmark command for `SkateFormer`.
2. Profile the current `SkateFormerBlock` under the pruned inference regime and separate operator cost from block-organization overhead.
3. Design one skeleton-specific block that removes explicit `partition -> reverse -> cat -> proj` as the primary computation pattern.
4. Test the redesigned block first in a partial stage replacement rather than a full-model swap.
5. Use simple width or MLP reductions only as engineering baselines, not as the main paper claim.
6. Only if the prototype has a promising cost reduction, add teacher-guided recovery for accuracy.

## Kill Criteria And Decision Points

- If the baseline benchmark is not stable, stop architecture iteration and fix measurement first.
- If the redesigned block reduces FLOPs but not latency, narrow the claim or redesign the implementation.
- If the accelerated block loses too much accuracy for a modest speed gain, do not force the method story.
- If a simple width reduction beats the architectural change, prefer the simpler baseline.
- If `NTU60` does not show a stable Pareto improvement, do not expand to `NTU120` or deployment claims prematurely.
