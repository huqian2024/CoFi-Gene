"""Evaluate a saved upstream checkpoint on a processed dataset (M5.1 Stage 3).

Stage 3 of the M5.1 experiment matrix evaluates *already trained* checkpoints on a
hard-negative candidate set instead of retraining everything. That needs a code
path that

* rebuilds the model from the checkpoint's own recorded config (so B0-B7 keep
  their exact definitions and hidden dimensions),
* applies the same visibility protocol / edge removal as the training run,
* loads the frozen weights and scores the dataset's fixed positives and negatives.

Nothing is optimised here and the test split is only scored, never selected on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import CoFiDataset
from .metrics import binary_summary

__all__ = ["load_checkpoint", "build_model_from_checkpoint", "evaluate_checkpoint",
           "score_split"]


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a ``model.pt`` written by ``cofi_gene.train``."""
    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a cofi_gene checkpoint (missing state_dict)")
    return payload


def score_split(model, x, adjacencies, dataset: CoFiDataset, split: str) -> dict:
    """AUPRC/AUROC per channel on one split (mirrors ``train.evaluate_split``)."""
    import torch

    from .train import _edge_tensor

    model.eval()
    result: dict = {}
    with torch.no_grad():
        embeddings = model(x, adjacencies)
        for channel in model.channels:
            part = dataset.channel(channel)
            positives = _edge_tensor(getattr(part, split), x.device)
            negatives = _edge_tensor(getattr(part, f"neg_{split}"), x.device)
            vectors = embeddings[channel]
            positive_scores = model.score_edges(vectors, positives,
                                                channel=channel).cpu().numpy()
            negative_scores = model.score_edges(vectors, negatives,
                                                channel=channel).cpu().numpy()
            if positive_scores.size == 0 or negative_scores.size == 0:
                result[channel] = {"status": "N/A",
                                   "n_positives": int(positive_scores.size),
                                   "n_negatives": int(negative_scores.size)}
                continue
            scores = np.concatenate([positive_scores, negative_scores])
            labels = np.concatenate([np.ones(positive_scores.size),
                                     np.zeros(negative_scores.size)])
            summary = binary_summary(scores, labels)
            summary["n_positives"] = int(positive_scores.size)
            summary["n_negatives"] = int(negative_scores.size)
            result[channel] = summary
    return result


def build_model_from_checkpoint(checkpoint: dict, feature_dim: int,
                                device: str = "cpu"):
    """Rebuild the exact B0-B7 module recorded inside a checkpoint.

    Shared by frozen evaluation and the Week 7 feedback-case export so both use
    the model's *own* recorded config instead of a hand-copied constructor call.
    """
    import torch

    from .models import build_model

    run_config = checkpoint.get("config")
    if not run_config:
        raise ValueError("checkpoint has no config; cannot rebuild the model")
    model_cfg = run_config["model"]
    model = build_model(model_cfg["name"], feature_dim=feature_dim,
                        hidden_dim=int(model_cfg["hidden_dim"]),
                        num_layers=int(model_cfg["num_layers"]),
                        dropout=float(model_cfg["dropout"]),
                        fusion_dim=int(model_cfg["fusion_dim"]),
                        edge_head=model_cfg["edge_head"],
                        attention_heads=int(model_cfg.get("attention_heads", 4)),
                        attention_dropout=model_cfg.get("attention_dropout"),
                        attention_residual=bool(model_cfg.get("attention_residual",
                                                              False)),
                        feedback_alpha=float(model_cfg.get("feedback_alpha", 0.1)),
                        feedback_margin=float(model_cfg.get("feedback_margin", 0.5)),
                        ).to(torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    return model, run_config


def evaluate_checkpoint(dataset: CoFiDataset, checkpoint_path: str | Path,
                        config: dict | None = None, split: str = "test",
                        device: str = "cpu", protocol: str | None = None,
                        edge_removal_mode: str | None = None,
                        edge_drop_fraction: float | None = None,
                        drop_channels=None) -> dict:
    """Score one frozen checkpoint under the requested protocol.

    ``config`` defaults to the config stored inside the checkpoint, which is what
    makes the evaluation reproduce the training conditions.
    """
    import torch

    from .models import build_adjacencies
    from .perturbation import apply_data_perturbations

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if config is not None:   # explicit override keeps the public API of M5.1
        checkpoint = {**checkpoint, "config": config}
    model, run_config = build_model_from_checkpoint(checkpoint, dataset.num_features,
                                                    device=device)
    model.eval()

    perturb_config = {"data": {},
                      "runtime": {"seed": int(run_config["runtime"]["seed"])}}
    if protocol is not None:
        perturb_config["data"]["visibility_protocol"] = protocol
    if edge_removal_mode is not None:
        perturb_config["data"]["edge_removal_mode"] = edge_removal_mode
    if edge_drop_fraction is not None:
        perturb_config["data"]["edge_drop_fraction"] = float(edge_drop_fraction)
    if drop_channels is not None:
        perturb_config["data"]["drop_channels"] = list(drop_channels)
    prepared, perturbation = apply_data_perturbations(dataset, perturb_config)

    x = torch.as_tensor(prepared.X, dtype=torch.float32, device=torch.device(device))
    encode_edges = {channel: torch.as_tensor(prepared.channel(channel).encode_edges,
                                             dtype=torch.long, device=x.device)
                    for channel in model.channels}
    encode_weights = {channel: torch.as_tensor(prepared.channel(channel).encode_weights,
                                               dtype=torch.float32, device=x.device)
                      for channel in model.channels}
    adjacencies = build_adjacencies(encode_edges, encode_weights, prepared.num_nodes)
    if getattr(model, "uses_feedback", False):
        model.set_encoding_graphs(encode_edges, encode_weights, prepared.num_nodes)

    metrics = score_split(model, x, adjacencies, prepared, split)
    return {
        "checkpoint": str(checkpoint_path),
        "model": model.name,
        "split": split,
        "protocol": perturbation["visibility_protocol"],
        "edge_removal_mode": perturbation["edge_removal_mode"],
        "edge_drop_fraction": perturbation["edge_drop_fraction"],
        "changed": perturbation["changed"],
        "perturbation": perturbation,
        "visibility": perturbation["visibility"],
        "metrics": metrics,
    }

