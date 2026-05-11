# H0 Protocol: Early-Recognition Freeze

## Objective

Freeze the canonical early skeleton action-recognition protocol before running major experiments.

## Core Question

What exact prefix-construction and evaluation pipeline defines the project benchmark?

## Hypothesis

If prefix construction, valid-frame counting, interpolation, and observation-ratio handling are frozen now, later baseline and method comparisons will remain fair and interpretable.

## Scope To Lock

- Primary dataset: `NTU60`
- Primary split: `XSub`
- Next validation split: `XView`
- Observation ratios: `0.1`, `0.3`, `0.5`, `0.7`, `0.9`, `1.0`
- Backbone family: `SkateFormer`
- Evaluation target: per-ratio `Top-1 Accuracy`

## Required Decisions

- How valid frames are counted for skeleton sequences with missing trailing frames
- Whether prefix inputs are center-cropped, front-cropped, or randomly sampled
- Which interpolation and fixed-length policy is canonical
- Whether multi-ratio training is deferred until after the single-ratio prefix baseline
- What exact command/config path defines the baseline run

## Inputs

- Existing `SkateFormer` feeder crop/resize logic
- Current NTU preprocessing pipeline under `SkateFormer/data/ntu`
- Current `SkateFormer` training configs for NTU60
- Current coarse label-mapping files for later auxiliary supervision

## Metrics To Lock

- Primary metric: per-ratio `Top-1 Accuracy`
- Secondary metric: average accuracy across the fixed ratio set
- Sanity metric: full-observation (`1.0`) upper bound under the same data path

## Experiment Matrix

1. Write the canonical prefix protocol in one place.
2. Verify that evaluation does not mix random crop with prefix crop.
3. Confirm the fixed window length and interpolation behavior.
4. Define the first official `NTU60 XSub` baseline command path.
5. Defer method additions until the protocol and baseline path are stable.

## Acceptance Criteria

- Prefix construction is unambiguous.
- The observation-ratio list is fixed.
- One canonical `SkateFormer-prefix` baseline path is identified.
- The repository no longer mixes the old deployment benchmark with the new early-recognition benchmark.

## Deliverables

- Locked early-recognition protocol note
- Canonical baseline config/command reference
- Short list of remaining implementation gaps before the first baseline run

## Risks

- Random crop and prefix crop may still be conflated in current feeder settings.
- The current NTU preprocessing path may not yet encode the desired early protocol cleanly.
- A baseline may appear to improve simply because the protocol drifted.
