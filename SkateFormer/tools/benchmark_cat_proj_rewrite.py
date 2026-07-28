import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.SkateFormer import SkateFormerBlock
from model.operators import SkateFormerBlockCatProjFreeExact


def benchmark(module, x, iters, warmup):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)

        if x.is_cuda:
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(iters):
                _ = module(x)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = module(x)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0

    return total_ms / max(iters, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=14)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--partition-size", type=int, nargs=2, default=(8, 7))
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.window_size,
        args.num_points,
        device=device,
    )
    partition = tuple(args.partition_size)

    baseline = SkateFormerBlock(
        in_channels=args.channels,
        num_points=args.num_points,
        kernel_size=args.kernel_size,
        num_heads=args.num_heads,
        type_1_size=partition,
        type_2_size=partition,
        type_3_size=partition,
        type_4_size=partition,
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    rewrite = SkateFormerBlockCatProjFreeExact(
        in_channels=args.channels,
        num_points=args.num_points,
        kernel_size=args.kernel_size,
        num_heads=args.num_heads,
        type_1_size=partition,
        type_2_size=partition,
        type_3_size=partition,
        type_4_size=partition,
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    rewrite.load_state_dict(baseline.state_dict(), strict=True)

    with torch.no_grad():
        baseline_out = baseline(x)
        rewrite_out = rewrite(x)

    diff = (baseline_out - rewrite_out).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    allclose = torch.allclose(baseline_out, rewrite_out, atol=1e-6, rtol=1e-5)

    baseline_ms = benchmark(baseline, x, args.iters, args.warmup)
    rewrite_ms = benchmark(rewrite, x, args.iters, args.warmup)
    speedup = baseline_ms / rewrite_ms if rewrite_ms > 0 else float("inf")

    print("===== Cat+Proj Rewrite Benchmark =====")
    print(f"device: {device}")
    print(
        "shape: "
        f"(B, C, T, V)=({args.batch_size}, {args.channels}, {args.window_size}, {args.num_points})"
    )
    print(f"partition size: {partition}")
    print(f"baseline latency: {baseline_ms:.4f} ms")
    print(f"rewrite latency: {rewrite_ms:.4f} ms")
    print(f"speedup: {speedup:.3f}x")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")
    print(f"allclose(atol=1e-6, rtol=1e-5): {allclose}")


if __name__ == "__main__":
    main()
