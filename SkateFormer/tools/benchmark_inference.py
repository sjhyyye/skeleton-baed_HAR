import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.SkateFormer import SkateFormer_


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model_args = dict(
        num_classes=args.num_classes,
        num_people=args.num_people,
        num_points=args.num_points,
        kernel_size=args.kernel_size,
        num_heads=args.num_heads,
        attn_drop=args.attn_drop,
        head_drop=args.head_drop,
        rel=args.rel,
        drop_path=args.drop_path,
        type_1_size=args.type_1_size,
        type_2_size=args.type_2_size,
        type_3_size=args.type_3_size,
        type_4_size=args.type_4_size,
        mlp_ratio=args.mlp_ratio,
        index_t=args.index_t,
    )
    return SkateFormer_(**model_args)


def create_inputs(
    batch_size: int,
    window_size: int,
    num_points: int,
    num_people: int,
    device: torch.device,
):
    data = torch.randn(batch_size, 3, window_size, num_points, num_people, device=device, dtype=torch.float32)
    index_t_1d = torch.linspace(-1.0, 1.0, steps=window_size, device=device, dtype=torch.float32)
    index_t = index_t_1d.unsqueeze(0).repeat(batch_size, 1)
    return data, index_t


def load_weights(model: torch.nn.Module, weights_path: str, strict: bool):
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    cleaned = {k.split("module.")[-1]: v for k, v in ckpt.items()}
    incompatible = model.load_state_dict(cleaned, strict=strict)
    if strict:
        return
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        print(f"[load_weights] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_weights] unexpected keys: {len(unexpected)}")


def benchmark(
    model: torch.nn.Module,
    data: torch.Tensor,
    index_t: torch.Tensor,
    iters: int,
    warmup: int,
):
    device = data.device
    use_cuda = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        if use_cuda:
            torch.cuda.synchronize()
        for _ in range(max(warmup, 0)):
            _ = model(data, index_t)
        if use_cuda:
            torch.cuda.synchronize()
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(iters):
                _ = model(data, index_t)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(data, index_t)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0
    return total_ms


def measure_flops(model: torch.nn.Module, data: torch.Tensor, index_t: torch.Tensor):
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception as e:
        return None, f"torch.profiler not available: {e}"

    activities = [ProfilerActivity.CPU]
    if data.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    try:
        with torch.no_grad():
            with profile(activities=activities, with_flops=True) as prof:
                _ = model(data, index_t)
    except TypeError as e:
        return None, f"profiler does not support FLOPs on this torch version: {e}"
    except Exception as e:
        return None, f"failed to profile FLOPs: {e}"

    total_flops = 0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", None)
        if flops:
            total_flops += flops

    if total_flops == 0:
        return None, "FLOPs unavailable (profiler returned 0); ops may not be supported"

    return int(total_flops), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--num-people", "--num_people", dest="num_people", type=int, default=1)
    parser.add_argument("--num-points", type=int, required=True)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--attn-drop", type=float, default=0.5)
    parser.add_argument("--head-drop", type=float, default=0.0)
    parser.add_argument("--rel", type=parse_bool, default=True)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--type-1-size", type=int, nargs=2, required=True)
    parser.add_argument("--type-2-size", type=int, nargs=2, required=True)
    parser.add_argument("--type-3-size", type=int, nargs=2, required=True)
    parser.add_argument("--type-4-size", type=int, nargs=2, required=True)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--index-t", type=parse_bool, default=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--strict-weights", action="store_true")
    parser.add_argument("--report-flops", type=parse_bool, default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(args).to(device)
    if args.weights:
        if not os.path.isabs(args.weights):
            weights_path = os.path.abspath(args.weights)
        else:
            weights_path = args.weights
        load_weights(model, weights_path, strict=args.strict_weights)

    data, index_t = create_inputs(
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_points=args.num_points,
        num_people=args.num_people,
        device=device,
    )

    total_ms = benchmark(
        model=model,
        data=data,
        index_t=index_t,
        iters=args.iters,
        warmup=args.warmup,
    )
    per_iter_ms = total_ms / max(args.iters, 1)
    samples_per_s = (args.batch_size / per_iter_ms) * 1000.0 if per_iter_ms > 0 else float("inf")

    print("===== SkateFormer Inference Benchmark =====")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"input: (B, C, T, V, M)=({args.batch_size}, 3, {args.window_size}, {args.num_points}, {args.num_people})")
    print(f"iters: {args.iters}, warmup: {args.warmup}")
    print(f"latency: {per_iter_ms:.3f} ms/iter")
    print(f"throughput: {samples_per_s:.2f} samples/s")

    if args.report_flops:
        flops, err = measure_flops(model=model, data=data, index_t=index_t)
        if err:
            print(f"FLOPs: unavailable ({err})")
        else:
            per_iter_s = per_iter_ms / 1000.0
            flops_per_s = flops / per_iter_s if per_iter_s > 0 else float("inf")
            print(f"GFLOPs (per forward, per sample): {flops / 1000000000}")

if __name__ == "__main__":
    main()
