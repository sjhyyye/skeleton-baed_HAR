# H3 Protocol: Joint Model And Robustness

## Objective

Train the full early-recognition model and stress-test the final claim.

## Hypothesis

The joint `intent + consistency` model should matter most at low observation ratios and should remain useful under more than one coarse label mapping.

## Preconditions

- `H0` protocol is frozen.
- `H1` prefix baseline is stable.
- `H2` has identified which modules are genuinely useful.

## Main Questions

- Does the joint model beat the strongest fair simpler baseline at `0.1` and `0.3`?
- Does the story survive more than one coarse mapping?
- Does the claim transfer from `NTU60 XSub` to `XView`?
- Are improvements concentrated in specific action families or confusion pairs?

## Required Metrics

- Per-ratio `Top-1 Accuracy`
- Mean accuracy across the fixed ratios
- Low-ratio gain over the strongest simpler baseline
- Mapping sensitivity and split sensitivity

## Experiment Matrix

1. Train the final joint model on `NTU60 XSub`.
2. Compare it against the strongest simpler baseline from `H2`.
3. Re-run with at least one alternative coarse mapping.
4. Validate on `NTU60 XView`.
5. Perform error analysis by action family and confusion pattern.
6. Defer `NTU120` until the `NTU60` story is stable.

## Acceptance Criteria

- The full model beats the strongest simpler baseline in the low-ratio regime, or the method is simplified.
- The claim does not depend entirely on one handcrafted mapping.
- `XSub` and `XView` tell a coherent enough story for paper writing.

## Deliverables

- Main result table across ratios
- Mapping-robustness table
- `XView` validation table
- Final keep-or-simplify decision for the method

## Risks

- The full model may fail to beat a simpler `KD-only` or `intent-only` baseline.
- Gains may disappear outside one mapping or one split.
- Improvements may appear mainly at high ratios, weakening the early-recognition claim.
