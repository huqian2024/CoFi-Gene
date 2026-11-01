"""Config loading: defaults, `base:` inheritance, strict validation, hashing."""

import json

import pytest
import yaml

from cofi_gene.config import (CHANNEL_SETS, DEFAULTS, dump_config, load_config,
                              resolved_config_hash, validate_config)


def _write(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_every_shipped_config_is_valid():
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for name in ("b0", "b1", "b2", "b3"):
        config = load_config(config_dir / f"{name}.yaml")
        assert config["model"]["name"] == name
        assert tuple(config["model"]["channels"]) == CHANNEL_SETS[name]


def test_base_inheritance_merges_nested_sections(tmp_path):
    _write(tmp_path / "_base.yaml", {"optim": {"epochs": 7}, "runtime": {"seed": 11}})
    child = _write(tmp_path / "b1.yaml", {"base": "_base.yaml",
                                          "model": {"name": "b1", "channels": ["A"]},
                                          "optim": {"lr": 0.5}})
    config = load_config(child)
    assert config["optim"] == {**DEFAULTS["optim"], "epochs": 7, "lr": 0.5}
    assert config["runtime"]["seed"] == 11
    assert config["model"]["name"] == "b1"


def test_cli_overrides_win_over_file_values(tmp_path):
    path = _write(tmp_path / "b0.yaml", {"model": {"name": "b0"}})
    config = load_config(path, {"runtime": {"seed": 123}, "optim": {"epochs": 2}})
    assert config["runtime"]["seed"] == 123
    assert config["optim"]["epochs"] == 2


def test_unknown_keys_are_rejected_instead_of_ignored(tmp_path):
    path = _write(tmp_path / "bad.yaml", {"model": {"name": "b1", "channels": ["A"],
                                                    "hidden_dim": 64, "widht": 3}})
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)
    path = _write(tmp_path / "bad2.yaml", {"modl": {"name": "b1"}})
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(path)


def test_channel_mismatch_and_planned_models_are_rejected(tmp_path):
    path = _write(tmp_path / "b2.yaml", {"model": {"name": "b2", "channels": ["A"]}})
    with pytest.raises(ValueError, match="uses channels"):
        load_config(path)
    path = _write(tmp_path / "b5.yaml", {"model": {"name": "b5", "channels": ["A", "B"]}})
    config = load_config(path)          # config layer accepts planned models...
    assert config["model"]["name"] == "b5"
    pytest.importorskip("torch")        # ...but the model factory refuses to fake them
    from cofi_gene.models import build_model

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        build_model("b5", feature_dim=8)


def test_value_ranges_are_validated():
    bad = json.loads(json.dumps(DEFAULTS))
    bad["optim"]["lr"] = 0
    with pytest.raises(ValueError, match="optim.lr"):
        validate_config(bad)
    bad = json.loads(json.dumps(DEFAULTS))
    bad["model"]["dropout"] = 1.5
    with pytest.raises(ValueError, match="dropout"):
        validate_config(bad)
    bad = json.loads(json.dumps(DEFAULTS))
    bad["model"]["edge_head"] = "transformer"
    with pytest.raises(ValueError, match="edge_head"):
        validate_config(bad)


def test_missing_file_and_bad_payload(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(broken)


def test_config_hash_and_dump_are_stable(tmp_path):
    base = json.loads(json.dumps(DEFAULTS))
    base["model"]["name"] = "b1"
    base["model"]["channels"] = ["A"]
    validate_config(base)
    dumped = dump_config(base, tmp_path / "out" / "config.json")
    assert json.loads(dumped.read_text())["model"]["name"] == "b1"
    assert resolved_config_hash(base) == resolved_config_hash(json.loads(json.dumps(base)))
    changed = json.loads(json.dumps(base))
    changed["optim"]["lr"] = 0.02
    assert resolved_config_hash(base) != resolved_config_hash(changed)
