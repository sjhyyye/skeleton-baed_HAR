# Research Findings

## Research Question

Can `SkateFormer` be accelerated in a way that improves the accuracy-latency-FLOPs Pareto frontier, rather than only lowering symbolic complexity on paper?

## Current Understanding

The branch has pivoted back to acceleration as the main organizing problem. The earlier early-recognition plan is now archived for this branch and should not drive benchmark choice, evaluation criteria, or method design.

The important local code observation is that the current `SkateFormerBlock` is already a hybrid block. It has one shared `mapping`, then splits computation into graph convolution, temporal convolution, and four partitioned attention branches, followed by `proj` and an `MLP`. That means the new branch should not assume a naive "pure conv plus pure attention" baseline. The real question is which parts of this existing hybrid are still unnecessarily expensive and whether an `ACmix`-style redesign can simplify them.

This makes the acceleration plan more concrete:

- freeze one benchmark path first
- profile the current block rather than guessing
- redesign the block organization rather than only one operator
- verify real latency, not only FLOPs

The current paper-facing direction is now narrower than a generic ACmix adaptation. For the pruned low-`V` regime, the more defensible claim is that `SkateFormer` becomes structurally inefficient because it explicitly materializes multiple partition branches and then fuses them with `cat + proj`. This suggests a partition-free, branch-collapsed, and cat-proj-free redesign as the core method story.

## Current Status

- The active direction is now `SkateFormer` acceleration, not early recognition.
- The new central benchmark is standard full-sequence `NTU60`, with `XSub` first and `XView` second.
- The first phase is benchmark freeze and cost profiling, not model ablation.
- The earlier early-recognition experiment folders remain archived only.

## What Is Already Clear

- The repository already contains a usable inference benchmark entry point in `SkateFormer/tools/benchmark_inference.py`.
- The current block structure suggests that `mapping`, multi-branch aggregation, `proj`, and `MLP` are the first places to audit.
- The repository also contains older pruning evidence, which is useful as acceleration context but should not be treated as the current baseline table.
- An architecture claim will only be credible if it beats simple baselines such as width reduction, head reduction, or old pruning settings.
- For the paper narrative, operator-level speedups are weaker than a redesign that removes explicit partition-reverse and concat-projection structure.

## Open Questions

- Which part of the current block actually dominates wall-clock runtime under the canonical benchmark?
- How closely should the `ACmix` idea be adapted versus rewritten for temporal-joint skeleton tokens?
- Does partial stage replacement outperform a full-model swap?
- Will FLOPs reductions survive contact with real latency measurement?
- Can the `partition -> reverse -> cat -> proj` pattern be replaced by a cleaner skeletal mixer without losing the useful inductive bias?

## Optimization Trajectory

The project should now move in a strict order:

1. Freeze one baseline benchmark for both accuracy and speed.
2. Measure the unmodified `SkateFormer` cleanly under that benchmark.
3. Profile the current block and decide whether block organization, not a single operator, is the main paper-worthy problem.
4. Implement one partition-free and branch-collapsed prototype block and test it in a limited stage replacement.
5. Only after a real cost gain appears, decide whether distillation, pruning, or further simplification is worth adding.

This trajectory is intentionally conservative. The current bottleneck is not paper inspiration; it is matched measurement and disciplined architecture iteration.
