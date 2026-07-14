# Research Findings

## Research Question

Can a skeleton-based action-recognition model classify actions reliably from partial motion prefixes, especially at low observation ratios, without using frame-level stage labels or heavy sub-action parsing?

## Current Understanding

The project has now pivoted fully to early skeleton action recognition. The older acceleration and deployment agenda is no longer the main organizing problem. It remains historical context only and should not drive current benchmark choice, evaluation criteria, or method design.

The central challenge is no longer latency under deployment constraints. It is ambiguity under partial observation. In the early stage of an action, the model often lacks enough evidence to discriminate fine-grained labels confidently. This makes two directions especially relevant:

- coarse semantic supervision that may emerge earlier than full class identity
- training-time guidance from the full sequence to stabilize prefix predictions

The current project is therefore best framed as a lightweight early-recognition study built around a `SkateFormer` backbone, coarse intent auxiliary supervision, and full-prefix consistency learning.

## Current Status

- The early-recognition direction is now the primary project line.
- The new central benchmark is `NTU60` under an early-recognition protocol.
- `XSub` is the first split to lock; `XView` is the next validation split.
- `NTU120` is postponed until the `NTU60` story is stable.
- The `H1` baseline family is now substantially complete on `NTU60 XSub`.
- The current phase remains single-module ablation, but the intent line is now materially clarified.

## What Is Already Clear

- The project now has a fixed prefix-construction path and a usable early-recognition baseline family on `NTU60 XSub`.
- `prefix_multi` is stronger than matched single-ratio training at both `0.1` and `0.3`.
- The largest baseline gain appears in the hardest early regime: `31.88%` vs `27.66%` Top-1 at `r=0.1`, and `68.92%` vs `67.04%` at `r=0.3`.
- The prefix-based multi-ratio path preserves near-full-observation performance: `91.42%` at `1.0` versus the clean full upper bound `91.73%`.
- Fixed `intent-only` has now been evaluated under both semantic and trajectory coarse mappings.
- The trajectory mapping is the better intent implementation, but it still does not beat `prefix_multi` at `0.1` or `0.3`.
- A minimal `intent-improved` variant with ratio-adaptive intent weights and intent-to-action bias/gating reaches `31.30% / 68.48% / 85.53%` at `0.1 / 0.3 / 0.5`. This is a small gain over fixed trajectory intent-only at `0.3 / 0.5`, but it still misses `prefix_multi` at the decisive low ratios.
- Full-sequence guidance must be compared against a simpler `KD-only` or equivalent teacher baseline.
- The main claim should target low observation ratios first, especially `0.1` and `0.3`.

## Open Questions

- Which coarse-intent grouping is stable enough to support a paper claim?
- Does consistency learning add value beyond straightforward teacher guidance?
- Are the gains concentrated in specific action families such as interaction or object manipulation?
- Does the story hold on `XView` after it is established on `XSub`?

## Optimization Trajectory

The project should now move in a strict order:

1. Keep the completed `H1` baseline family fixed as the reference (`r=0.1`, `r=0.3`, and `prefix_multi`).
2. Treat the semantic row, the trajectory row, and the adaptive-gated row as the complete current intent reference set rather than continuing to tweak intent first.
3. Test `KD-only` and then `consistency-only` under the same locked protocol.
4. Train the joint model only after the simpler teacher-guided comparisons are understood.
5. Move to `XView` only after the `NTU60 XSub` gain sources are separated cleanly.

This trajectory is intentionally conservative. The current bottleneck is not idea generation; it is experimental discipline.
