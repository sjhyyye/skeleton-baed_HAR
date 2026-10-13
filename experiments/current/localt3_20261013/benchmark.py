
"""Matched B0 vs RCA vs RCA local temporal inference benchmark. Run from any directory in skateformer env."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SkateFormer"))
from model.SkateFormer import SkateFormer_
from tools.benchmark_inference import load_weights

def gpu_status():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True).strip()
    except Exception as exc:
        return str(exc)

def percentile(values, q):
    values = sorted(values)
    idx = (len(values)-1)*q
    lo = int(idx)
    hi = min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(idx-lo)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.json")))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required; no silent CPU fallback"
    assert args.iterations>0 and args.rounds>0 and args.warmup>=0
    torch.set_num_threads(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    result = {
        "environment": {"python": platform.python_version(), "torch":torch.__version__,
            "cuda_runtime":torch.version.cuda, "cudnn":torch.backends.cudnn.version(),
            "gpu":torch.cuda.get_device_name(0), "gpu_status_before":gpu_status()},
        "protocol":{"shape":[1,3,64,14,1],"dtype":"float32","tf32":False,
            "cpu_threads":1,"eval":True,"autograd":"no_grad","cudnn_benchmark":True,
            "warmup":args.warmup,"iterations_per_round":args.iterations,"rounds":args.rounds,
            "timing":"perf_counter per forward, synchronize before and after; GPU-resident inputs; includes Python dispatch",
            "scope":"classifier only, excludes data loading, H2D, pose extraction; no torch.compile or AMP",
            "flops":"CPU torch.profiler with_flops supported conv/matmul operator estimates; multiply/add counted separately; partial, not exhaustive"},
        "models":{},"order":[]}
    models={}
    for label,name,weight in [
        ("B0","baseline_14p_1person",None),
        ("RCA","rca_14p_1person_ffn4_attndrop01",None),
        ("LocalT3","rca_14p_1person_ffn4_attndrop01_localt3",None)]:
        source=ROOT/"SkateFormer/config/train/acceleration"/(name+".yaml")
        raw=source.read_bytes()
        cfg=yaml.safe_load(raw)
        Path(__file__).with_name(label+".config.yaml").write_bytes(raw)
        assert cfg["model_args"]["num_points"]==14 and cfg["model_args"]["num_people"]==1
        torch.manual_seed(1)
        model=SkateFormer_(**cfg["model_args"]).eval()
        if weight:
            load_weights(model,weight,strict=True)
        entry={"config":str(source.relative_to(ROOT)),
               "config_sha256":hashlib.sha256(raw).hexdigest(),
               "model_args":cfg["model_args"],
               "weights":str(Path(weight).resolve()) if weight else "random_seed_1",
               "parameters":sum(p.numel() for p in model.parameters()),
               "rounds":[]}
        cpu_x=torch.randn(1,3,64,14,1)
        cpu_t=torch.linspace(-1,1,64).unsqueeze(0)
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU],with_flops=True,record_shapes=True) as prof:
                output=model(cpu_x,cpu_t)
        assert output.shape==(1,60) and torch.isfinite(output).all()
        ops=[]
        for event in prof.key_averages():
            if getattr(event,"flops",0):
                ops.append({"operator":event.key,"calls":event.count,"flops":int(event.flops)})
        entry["counted_flop_operators"]=ops
        entry["gflops_per_sample"]=sum(o["flops"] for o in ops)/1e9
        entry["flops_unaccounted_operators"]=[e.key for e in prof.key_averages() if not getattr(e,"flops",0)]
        model=model.cuda()
        models[label]=model
        result["models"][label]=entry
        print(label,"params",entry["parameters"],"profiler GFLOPs",entry["gflops_per_sample"],flush=True)
    torch.manual_seed(123)
    data=torch.randn(1,3,64,14,1,device="cuda")
    index=torch.linspace(-1,1,64,device="cuda").unsqueeze(0)
    with torch.no_grad():
        for model in models.values():
            for _ in range(args.warmup):
                model(data,index)
        torch.cuda.synchronize()
        for r in range(args.rounds):
            from itertools import permutations
            order=list(list(permutations(models))[r % 6])
            result["order"].append(order)
            for label in order:
                model=models[label]
                for _ in range(10):
                    model(data,index)
                torch.cuda.synchronize()
                times=[]
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    model(data,index)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                entry={"round":r+1,"wall_ms":times,"mean_ms":statistics.mean(times),
                       "median_ms":statistics.median(times),"p95_ms":percentile(times,.95)}
                result["models"][label]["rounds"].append(entry)
                print(label,"round",r+1,"mean ms",round(entry["mean_ms"],4),flush=True)
    for label,entry in result["models"].items():
        times=[v for r in entry["rounds"] for v in r["wall_ms"]]
        entry["summary"]={"mean_ms":statistics.mean(times),"median_ms":statistics.median(times),
            "p95_ms":percentile(times,.95),"throughput_samples_s":1000/statistics.mean(times),
            "round_mean_min_ms":min(r["mean_ms"] for r in entry["rounds"]),
            "round_mean_max_ms":max(r["mean_ms"] for r in entry["rounds"])}
    result["comparison"] = {}
    for ref in ["B0", "RCA"]:
        baseline=result["models"][ref]["summary"]["mean_ms"]
        new=result["models"]["LocalT3"]["summary"]["mean_ms"]
        result["comparison"]["localt3_vs_"+ref+"_latency_change_percent"]=100*(new/baseline-1)
    result["environment"]["gpu_status_after"]=gpu_status()
    dest=Path(args.output)
    assert not dest.exists(), "Use a fresh output filename to preserve measurements"
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v["summary"] for k,v in result["models"].items()},indent=2),flush=True)
    print("Comparison:",json.dumps(result["comparison"]),flush=True)
    print("Saved:",dest,flush=True)

if __name__=="__main__":
    main()
