# H1 Analysis

## Status

Pending

## Confirmatory Results

<!-- Record `SkateFormer-full`, `SkateFormer-prefix`, and `multi-ratio` baseline results here. -->

## Exploratory Findings

<!-- Record ratio-specific confusion trends, unstable classes, and training behavior here. -->

## Open Issues

- The canonical prefix baseline has not yet been logged in the repository.
- It is not yet clear whether the first baseline should start from a single ratio or directly from a multi-ratio schedule.
- Low-ratio failure patterns need to be characterized before adding auxiliary supervision.

## Decision

Lock `H1` only after the baseline table for `NTU60 XSub` is stable enough to anchor all later comparisons.
