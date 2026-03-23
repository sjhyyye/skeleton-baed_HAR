import argparse
import os
import sys
from typing import Any, Dict, Optional
import yaml
import torch
import torch.utils.data
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import import_class, init_seed


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)}")
    return data


def _parse_device(value: Optional[str]) -> torch.device:
    if value is None or value.strip() == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = value.strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value)
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    raise ValueError(f"unsupported --device value: {value}")


def _load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    weights = torch.load(weights_path, map_location=device)
    if isinstance(weights, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in weights and isinstance(weights[k], dict):
                weights = weights[k]
                break
    if not isinstance(weights, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(weights)}")
    cleaned = {k.split("module.")[-1]: v for k, v in weights.items()}
    model.load_state_dict(cleaned, strict=False)


def _build_test_loader(cfg: Dict[str, Any]):
    Feeder = import_class(cfg["feeder"])
    return torch.utils.data.DataLoader(
        dataset=Feeder(**cfg["test_feeder_args"]),
        batch_size=int(cfg.get("test_batch_size", 256)),
        shuffle=False,
        num_workers=int(cfg.get("num_worker", 0)),
        drop_last=False,
        worker_init_fn=init_seed,
    )


def _build_model(cfg: Dict[str, Any]):
    Model = import_class(cfg["model"])
    return Model(**cfg["model_args"])


def _zero_input_accuracy(model: torch.nn.Module, loader, device: torch.device, max_batches: int, zero_index_t: bool):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, index_t, label, _index) in enumerate(tqdm(loader, desc="zero-input-eval")):
            data = data.float().to(device)
            index_t = index_t.float().to(device)
            label = label.long().to(device)

            data.zero_()
            if zero_index_t:
                index_t.zero_()

            output = model(data, index_t)
            pred = torch.argmax(output, dim=1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())

            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

    return correct / max(total, 1), correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-batches", type=int, default=-1)
    ap.add_argument("--zero-index-t", type=str, default="true")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.abspath(args.config)
    cfg = _load_config(cfg_path)

    weights_path = None
    if args.weights:
        weights_path = args.weights if os.path.isabs(args.weights) else os.path.abspath(args.weights)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(weights_path)

    device = _parse_device(args.device)
    zero_index_t = args.zero_index_t.lower() in {"true", "1", "yes", "y", "t"}

    init_seed(int(cfg.get("seed", 1)))

    if weights_path is None:
        print("warning: --weights not provided; evaluating with random-initialized weights")

    loader = _build_test_loader(cfg)
    model = _build_model(cfg).to(device)
    if weights_path is not None:
        _load_weights(model, weights_path, device=device)

    acc, correct, total = _zero_input_accuracy(model, loader, device, int(args.max_batches), zero_index_t)
    print(f"zero-input top1: {acc * 100:.4f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
