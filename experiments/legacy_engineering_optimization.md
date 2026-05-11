# Legacy Engineering Optimization Note

This note preserves the older engineering-oriented line as historical context only.

## What It Was

Before the project pivoted to early skeleton action recognition, the main question was how far a `SkateFormer`-style recognizer could be simplified and accelerated for a rehabilitation or deployment-oriented setting.

That older line emphasized:

- joint pruning
- compression
- inference benchmarking
- eventual deployment considerations

## What Remains Useful

- It provides some implementation context for why parts of the repository mention reduced joint sets or benchmarking scripts.
- It may still be useful later if the project returns to an engineering or deployment subsection.

## What It Does Not Control Now

- It does not define the main benchmark.
- It does not define the paper contribution.
- It does not define the current success criteria.

## Working Rule

Treat the engineering optimization thread as a small side record unless the project explicitly pivots back.

## Historical Joint-Pruning Table

The following table preserves the older pruning exploration results.

Note on computation:
- The corrected single-person baseline fixes `25 joints = 3.46 GFLOPS`.
- The earlier `6.95 GFLOPS` figure corresponds to a two-person input setting and should not be used for this pruning table.
- Missing GFLOPS values below were backfilled by running the current local `SkateFormer/tools/benchmark_inference.py` on matching joint counts and then rescaling all results to the same baseline convention so the table remains internally consistent.
- These numbers should therefore be treated as consistent comparative estimates under the old engineering line, not as the active project metric.

### Hand Pruning

| Scheme | Removed joints | Accuracy | Remaining joints | GFLOPS |
|---|---|---:|---:|---:|
| 0 | None | 96.29% | 25 | 3.46 |
| 1 | 23, 25 | 96.31% | 23 | 3.16 |
| 2 | 8, 12 | 95.96% | 23 | 3.16 |
| 3 | 7, 11 | 96.28% | 23 | 3.16 |
| 4 | 8, 12, 23, 25 | 95.43% | 21 | 2.86 |
| 5 | 7, 11, 23, 25 | 95.91% | 21 | 2.86 |
| 6 | 7, 11, 8, 12, 23, 25 | 94.86% | 19 | 2.57 |
| 7 | 7, 11, 22, 23, 24, 25 | 95.12% | 19 | 2.57 |
| 8 | 8, 12, 22, 23, 24, 25 | 94.78% | 19 | 2.57 |

Current historical choice:
- Scheme `7` removes more hand-related joints while staying closest to the retained accuracy target, so it was the preferred old engineering candidate.

### Lower-Body Pruning

Starting from the hand-pruned setting, the remaining joints were:
`1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21`

| Scheme | Removed joints | Accuracy | Remaining joints | GFLOPS |
|---|---|---:|---:|---:|
| 9 | 1, 13, 17, 15, 19 | 93.97% | 14 | 1.86 |
| 10 | 1, 13, 17, 14, 18 | 94.23% | 14 | 1.86 |
| 11 | 1, 15, 19, 14, 18 | 93.86% | 14 | 1.86 |
| 12 | 13, 17, 14, 18 | 93.81% | 15 | 2.00 |

### Upper-Body Pruning

After removing hand and lower-body joints, the remaining joints were:
`2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 19, 20, 21`

| Scheme | Removed joints | Accuracy | Remaining joints | GFLOPS |
|---|---|---:|---:|---:|
| 13 | 5, 9 | 94.04% | 12 | 1.58 |
| 14 | 6, 10 | 93.81% | 12 | 1.58 |
| 15 | 5, 6, 9, 10 | 93.73% | 10 | 1.30 |
| 16 | 5, 9, 2, 3 | 94.12% | 10 | 1.30 |
| 17 | 5, 9, 3, 4 | 93.67% | 10 | 1.30 |
| 18 | 5, 9, 2, 4 | 93.92% | 10 | 1.30 |
| 19 | 5, 9, 2, 3, 4 | 93.70% | 9 | 1.17 |
