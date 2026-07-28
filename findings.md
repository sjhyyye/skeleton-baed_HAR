# Research Findings

## Research Question

How much of the current pruned `SkateFormer` inference latency can be removed by exact or near-exact operator rewrites, without any full retraining?

## Current Understanding

This branch is now focused on operator-level acceleration, not block redesign as the primary goal. The branch assumes that immediate, training-free gains are more valuable than a larger structural story that cannot be validated quickly under the current training budget.

The central implementation question is no longer "what new block should replace `SkateFormerBlock`?" It is "which parts of the current block can be rewritten into more inference-friendly operators while preserving the same computation or a numerically negligible deviation?"

## Current Status

- The active branch is now `fast_op`.
- The active target shape is `B=1, C=192, T=64, V=14`.
- The branch goal is inference acceleration through operator rewrites.
- Training-backed structural redesign is explicitly de-emphasized for this branch.

## What Is Already Clear

- The current branch should prioritize changes that can be benchmarked immediately.
- `cat + proj`, channel-last `Linear`, and graph-branch looping are the most obvious rewrite candidates.
- The correct validation protocol is not training accuracy first; it is output-difference plus latency.

## Open Questions

- How much speedup is available from `cat + proj` rewriting alone?
- Does a `Linear -> 1x1 Conv` rewrite reduce latency once layout changes are removed?
- How much does graph-branch loop fusion matter under `B=1`?
- Do the gains compose cleanly when multiple exact rewrites are stacked?

## Optimization Trajectory

The branch should move in this order:

1. Freeze the operator benchmark under the real target shape.
2. Benchmark the current implementation cleanly.
3. Implement one exact rewrite at a time.
4. Verify numerical equivalence for each rewrite.
5. Measure cumulative speedup after composing the safe rewrites.

This trajectory is intentionally narrow. The branch should earn every claim through immediate inference benchmarks rather than through speculative architectural ambition.
