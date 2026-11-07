"""YAML configuration loading with ``base:`` inheritance and strict validation.

A run must be reproducible from one file, so validation is strict: unknown keys
raise instead of being silently ignored, and every value that ends up in the run
log is explicitly named here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

__all__ = ["DEFAULTS", "load_config", "validate_config", "dump_config", "resolved_config_hash"]

DEFAULTS: dict = {
    "model": {
        "name": "b3",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "channels": ["A", "B"],
        "fusion_dim": 128,
        "edge_head": "dot",
    },
    "optim": {
        "lr": 0.01,
        "weight_decay": 0.0,
        "epochs": 200,
        "negatives_ratio": 1.0,
        "eval_every": 10,
        "early_stop_patience": 50,
        "grad_clip": 5.0,
    },
    "data": {
        "processed_dir": "data/processed/synthetic_v1",
        "weight_mode": "none",
    },
    "runtime": {
        "seed": 42,
        "device": "cpu",
        "out_dir": "runs",
        "run_name": None,
        "log_every": 10,
    },
}

_ALLOWED = {
    "model": {"name", "hidden_dim", "num_layers", "dropout", "channels", "fusion_dim",
              "edge_head",
              # Optional Week 3 keys (B5). They are deliberately *not* in DEFAULTS:
              # keeping them out means the resolved B0-B4 configs - and therefore
              # their run names and recorded hashes - stay unchanged.
              "attention_heads", "attention_dropout", "attention_residual"},
    "optim": {"lr", "weight_decay", "epochs", "negatives_ratio", "eval_every",
              "early_stop_patience", "grad_clip"},
    "data": {"processed_dir", "weight_mode"},
    "runtime": {"seed", "device", "out_dir", "run_name", "log_every"},
}

MODEL_NAMES = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")
CHANNEL_SETS = {
    "b0": ("A", "B"),   # feature baseline: evaluated on both graphs, no message passing
    "b1": ("A",),
    "b2": ("B",),
    "b3": ("A", "B"),
    "b4": ("A", "B"),
    "b5": ("A", "B"),
    "b6": ("A", "B"),
    "b7": ("A", "B"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load ``path`` (following an optional ``base:`` chain) and merge defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    base_path = loaded.pop("base", None)
    config = DEFAULTS
    if base_path:
        config = load_config((path.parent / base_path).resolve(), None)
    config = _deep_merge(config, loaded)
    if overrides:
        config = _deep_merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Strictly check structure and value ranges; raise ``ValueError`` otherwise."""
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    unknown_sections = set(config) - set(_ALLOWED)
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")
    for section, allowed in _ALLOWED.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"section {section!r} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown key(s) in {section}: {sorted(unknown)}")

    model = config["model"]
    if model["name"] not in MODEL_NAMES:
        raise ValueError(f"model.name must be one of {MODEL_NAMES}, got {model['name']!r}")
    expected_channels = CHANNEL_SETS[model["name"]]
    if tuple(model["channels"]) != expected_channels:
        raise ValueError(f"model {model['name']} uses channels {expected_channels}, "
                         f"config says {tuple(model['channels'])}")
    for key in ("hidden_dim", "num_layers", "fusion_dim"):
        if not isinstance(model[key], int) or model[key] <= 0:
            raise ValueError(f"model.{key} must be a positive int")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if model["edge_head"] not in ("dot", "mlp"):
        raise ValueError("model.edge_head must be 'dot' or 'mlp'")
    # Optional Week 3 keys: validated only when present (see _ALLOWED comment).
    if "attention_heads" in model:
        heads = model["attention_heads"]
        if not isinstance(heads, int) or heads < 1:
            raise ValueError("model.attention_heads must be a positive int")
        if model["hidden_dim"] % heads:
            raise ValueError(f"model.hidden_dim ({model['hidden_dim']}) must be divisible "
                             f"by model.attention_heads ({heads})")
    if "attention_dropout" in model and model["attention_dropout"] is not None:
        if not 0.0 <= float(model["attention_dropout"]) < 1.0:
            raise ValueError("model.attention_dropout must be in [0, 1)")
    if "attention_residual" in model and model["attention_residual"] is not None:
        if not isinstance(model["attention_residual"], bool):
            raise ValueError("model.attention_residual must be true or false")

    optim = config["optim"]
    if float(optim["lr"]) <= 0:
        raise ValueError("optim.lr must be > 0")
    if float(optim["weight_decay"]) < 0:
        raise ValueError("optim.weight_decay must be >= 0")
    if int(optim["epochs"]) < 1:
        raise ValueError("optim.epochs must be >= 1")
    if float(optim["negatives_ratio"]) <= 0:
        raise ValueError("optim.negatives_ratio must be > 0")
    if float(optim["grad_clip"]) < 0:
        raise ValueError("optim.grad_clip must be >= 0")

    if config["data"]["weight_mode"] not in ("none", "minmax"):
        raise ValueError("data.weight_mode must be 'none' or 'minmax'")
    if not str(config["data"]["processed_dir"]):
        raise ValueError("data.processed_dir must not be empty")
    if not str(config["runtime"]["out_dir"]):
        raise ValueError("runtime.out_dir must not be empty")


def dump_config(config: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolved_config_hash(config: dict) -> str:
    """Short stable hash of the fully resolved config (used in run names)."""
    import hashlib

    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
