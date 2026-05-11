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
- The current phase is protocol freeze, not large-scale result reporting.

## What Is Already Clear

- The project needs a single canonical prefix-construction protocol before method comparisons are meaningful.
- A strong `SkateFormer-prefix` baseline is required before any method claims are credible.
- Coarse intent supervision must be evaluated against more than one mapping scheme.
- Full-sequence guidance must be compared against a simpler `KD-only` or equivalent teacher baseline.
- The main claim should target low observation ratios first, especially `0.1` and `0.3`.

## Open Questions

- What exact prefix-cropping and interpolation pipeline should be treated as canonical?
- Which coarse-intent grouping is stable enough to support a paper claim?
- Does consistency learning add value beyond straightforward teacher guidance?
- Are the gains concentrated in specific action families such as interaction or object manipulation?
- Does the story hold on `XView` after it is established on `XSub`?

## Optimization Trajectory

The project should now move in a strict order:

1. Freeze the early-recognition protocol.
2. Build the `SkateFormer-prefix` baseline.
3. Add `multi-ratio` training under the same protocol.
4. Test `intent-only`, `consistency-only`, and `KD-only`.
5. Train the joint model only after the simpler comparisons are understood.

This trajectory is intentionally conservative. The current bottleneck is not idea generation; it is experimental discipline.
