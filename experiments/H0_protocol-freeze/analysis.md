# H0 Analysis

## Status

Active

## Confirmatory Results

- The repository-level project definition now treats early skeleton action recognition as the primary line.
- `NTU60` with `XSub` first and `XView` next is the locked benchmark order.
- Two coarse label-mapping schemes already exist and are ready for later robustness experiments.

## Exploratory Findings

- The feeder already supports ratio-based crop-and-resize logic, but the canonical early protocol is not yet frozen in one explicit config path.
- Current repository narrative was previously inconsistent because the experiment folder still reflected the older engineering line.

## Open Issues

- The first official `SkateFormer-prefix` baseline run has not yet been logged.
- The exact canonical prefix protocol still needs to be written as code-facing guidance, not only project-level planning.
- Multi-ratio training should remain secondary until the single baseline is stable.

## Decision

Promote `H0` only after the prefix construction path and baseline run command are frozen in practice.
