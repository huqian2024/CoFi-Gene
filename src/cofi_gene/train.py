"""Upstream link-prediction trainer for B0-B5 (M2/M3) with a reproducible CLI.

    python -m cofi_gene.train --config configs/b5.yaml --seed 42

Deliberate protocol choices (all recorded in the run JSON):

- Negatives come from the processed dataset, so every model B0-B7 is scored on the
  *same* negative sample; no model gets an easier negative draw.
- Validation decides early stopping; the test split is scored once, with the
  best-validation checkpoint, and never used for tuning.
- Every run writes config, seed, dataset hash, git commit, package versions and
  the full metric history.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from .config import dump_config, load_config, resolved_config_hash
from .dataset import CoFiDataset
from .manifest import git_commit, runtime_versions, sha256_file, utc_timestamp
from .metrics import binary_summary

__all__ = ["set_seed", "train_link_prediction", "evaluate_split", "main"]

SPLITS = ("train", "val", "test")


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch identically for every model."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is optional for data-only work
        pass


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("training needs torch: pip install -e '.[train]'") from exc
    return torch


def _resolve_device(requested: str):
    torch = _require_torch()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    return torch.device(requested)


def _edge_tensor(array, device):
    torch = _require_torch()
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"edge array must be [2, E], got {array.shape}")
    return torch.as_tensor(array, dtype=torch.long, device=device)


def evaluate_split(model, x, adjacencies, dataset: CoFiDataset, split: str) -> dict:
    """AUPRC/AUROC per channel on one split, computed with the numpy metrics."""
    torch = _require_torch()
    model.eval()
    result = {}
    with torch.no_grad():
        embeddings = model(x, adjacencies)
        for channel in model.channels:
            part = dataset.channel(channel)
            positives = _edge_tensor(getattr(part, split), x.device)
            negatives = _edge_tensor(getattr(part, f"neg_{split}"), x.device)
            embeddings_channel = embeddings[channel]
            positive_scores = model.score_edges(embeddings_channel, positives,
                                                channel=channel).cpu().numpy()
            negative_scores = model.score_edges(embeddings_channel, negatives,
                                                channel=channel).cpu().numpy()
            scores = np.concatenate([positive_scores, negative_scores])
            labels = np.concatenate([np.ones(positive_scores.size), np.zeros(negative_scores.size)])
            summary = binary_summary(scores, labels)
            summary["n_positives"] = int(positive_scores.size)
            summary["n_negatives"] = int(negative_scores.size)
            result[channel] = summary
    model.train()
    return result


def _mean_primary(metrics: dict, key: str = "average_precision") -> float:
    values = [channel[key] for channel in metrics.values() if key in channel]
    if not values:
        raise ValueError(f"no channel reported {key!r}")
    return float(np.mean(values))


def train_link_prediction(dataset: CoFiDataset, config: dict, device: str | None = None,
                          out_dir: str | Path | None = None,
                          evaluate_test: bool = True) -> dict:
    """Full training loop; returns the metrics/provenance record written to disk.

    ``evaluate_test=False`` skips the final test-split scoring. It exists so that a
    Week 3 *smoke* run can be honest about never touching the test split; the
    default (``True``) is the unchanged Week 2 behaviour.
    """
    torch = _require_torch()
    from torch.nn import functional as F

    from .models import build_adjacencies, build_model

    runtime = config["runtime"]
    optim_cfg = config["optim"]
    model_cfg = config["model"]
    seed = int(runtime["seed"])
    set_seed(seed)
    device_obj = _resolve_device(device or runtime["device"])

    dataset.validate()
    model = build_model(model_cfg["name"], feature_dim=dataset.num_features,
                        hidden_dim=int(model_cfg["hidden_dim"]),
                        num_layers=int(model_cfg["num_layers"]),
                        dropout=float(model_cfg["dropout"]),
                        fusion_dim=int(model_cfg["fusion_dim"]),
                        edge_head=model_cfg["edge_head"],
                        # B5 only; absent keys keep the module defaults so the
                        # B0-B4 resolved configs (and their run names) are unchanged.
                        attention_heads=int(model_cfg.get("attention_heads", 4)),
                        attention_dropout=model_cfg.get("attention_dropout"),
                        attention_residual=bool(model_cfg.get("attention_residual", False)),
                        ).to(device_obj)

    x = torch.as_tensor(dataset.X, dtype=torch.float32, device=device_obj)
    encode_edges, encode_weights = {}, {}
    for channel in model.channels:
        part = dataset.channel(channel)
        encode_edges[channel] = _edge_tensor(part.encode_edges, device_obj)
        encode_weights[channel] = torch.as_tensor(part.encode_weights, dtype=torch.float32,
                                                  device=device_obj)
    adjacencies = build_adjacencies(encode_edges, encode_weights, dataset.num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_cfg["lr"]),
                                 weight_decay=float(optim_cfg["weight_decay"]))
    epochs = int(optim_cfg["epochs"])
    eval_every = max(1, int(optim_cfg["eval_every"]))
    patience = int(optim_cfg["early_stop_patience"])
    grad_clip = float(optim_cfg["grad_clip"])

    history: list[dict] = []
    best = {"epoch": 0, "val_ap": float("-inf"), "state": None}
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model(x, adjacencies)
        loss_sum = 0.0
        for channel in model.channels:
            part = dataset.channel(channel)
            positives = _edge_tensor(part.train, device_obj)
            negatives = _edge_tensor(part.neg_train, device_obj)
            logits = torch.cat([
                model.score_edges(embeddings[channel], positives, channel=channel),
                model.score_edges(embeddings[channel], negatives, channel=channel)])
            labels = torch.cat([torch.ones(positives.shape[1], device=device_obj),
                                torch.zeros(negatives.shape[1], device=device_obj)])
            loss_sum = loss_sum + F.binary_cross_entropy_with_logits(logits, labels)
        loss = loss_sum / len(model.channels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        record = {"epoch": epoch, "train_loss": float(loss.detach())}
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate_split(model, x, adjacencies, dataset, "val")
            record["val"] = val_metrics
            val_ap = _mean_primary(val_metrics)
            record["val_mean_ap"] = val_ap
            if val_ap > best["val_ap"]:
                best = {"epoch": epoch, "val_ap": val_ap,
                        "state": {key: value.detach().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
        history.append(record)
        if stale * eval_every >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    val_metrics = evaluate_split(model, x, adjacencies, dataset, "val")
    test_metrics = (evaluate_split(model, x, adjacencies, dataset, "test")
                    if evaluate_test else None)

    target_dir = Path(out_dir or runtime["out_dir"])
    run_name = runtime.get("run_name") or (
        f"{model_cfg['name']}_seed{seed}_{resolved_config_hash(config)}")
    run_dir = target_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    processed = Path(config["data"]["processed_dir"])
    dataset_npz = processed / "dataset.npz"
    record = {
        "run_name": run_name,
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "versions": runtime_versions(),
        "config": config,
        "model": {
            "name": model_cfg["name"],
            "channels": list(model.channels),
            "parameter_count": model.parameter_count(),
            "output_dim": model.output_dim,
            "fusion_kind": model.fusion_kind,
            "edge_heads": ("per_channel" if model.channel_heads is not None
                           else "shared"),
        },
        "dataset": {
            "processed_dir": str(processed),
            "dataset_sha256": sha256_file(dataset_npz) if dataset_npz.exists() else None,
            "num_nodes": dataset.num_nodes,
            "num_features": dataset.num_features,
            "meta": dataset.meta,
        },
        "best_epoch": best["epoch"],
        "best_val_mean_ap": best["val_ap"],
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "notes": [
            ("test split scored once with the best-validation checkpoint"
             if evaluate_test else
             "test split NOT scored: smoke run launched with evaluate_test=False"),
            "negatives reused from the dataset build so all models see the same draw",
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    dump_config(config, run_dir / "config.json")
    torch.save({"state_dict": model.state_dict(), "config": config,
                "model_name": model_cfg["name"], "seed": seed},
               run_dir / "model.pt")
    return record


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m cofi_gene.train --config ... --seed ...``."""
    parser = argparse.ArgumentParser(
        prog="cofi_gene.train",
        description="Train one upstream model (B0-B5) on a processed CoFi-Gene dataset.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    parser.add_argument("--dataset", default=None, help="override data.processed_dir")
    parser.add_argument("--device", default=None, help="override runtime.device (cpu/cuda)")
    parser.add_argument("--out-dir", default=None, help="override runtime.out_dir")
    parser.add_argument("--run-name", default=None, help="override runtime.run_name")
    parser.add_argument("--epochs", type=int, default=None, help="override optim.epochs")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config and dataset, print a summary, then exit")
    parser.add_argument("--skip-test", action="store_true",
                        help="do not score the test split at all (Week 3 smoke runs)")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.seed is not None:
        overrides.setdefault("runtime", {})["seed"] = args.seed
    if args.device is not None:
        overrides.setdefault("runtime", {})["device"] = args.device
    if args.out_dir is not None:
        overrides.setdefault("runtime", {})["out_dir"] = args.out_dir
    if args.run_name is not None:
        overrides.setdefault("runtime", {})["run_name"] = args.run_name
    if args.epochs is not None:
        overrides.setdefault("optim", {})["epochs"] = args.epochs
    if args.dataset is not None:
        overrides.setdefault("data", {})["processed_dir"] = args.dataset

    config = load_config(args.config, overrides)
    dataset = CoFiDataset.load(config["data"]["processed_dir"])
    summary = dataset.validate()
    if args.dry_run:
        print(json.dumps({"config": config, "dataset": summary}, indent=2, sort_keys=True))
        return 0

    try:
        record = train_link_prediction(dataset, config, device=args.device,
                                       out_dir=args.out_dir,
                                       evaluate_test=not args.skip_test)
    except ImportError as exc:  # torch missing
        print(f"[cofi_gene.train] {exc}", file=sys.stderr)
        return 2

    test_ap = ("skipped (--skip-test)" if record["test"] is None else
               {channel: metrics.get("average_precision")
                for channel, metrics in record["test"].items()})
    print(f"[cofi_gene.train] run={record['run_name']} "
          f"best_epoch={record['best_epoch']} "
          f"val_mean_ap={record['best_val_mean_ap']:.4f} "
          f"test_ap={json.dumps(test_ap, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


