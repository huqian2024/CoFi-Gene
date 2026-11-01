#!/usr/bin/env python3
"""Print a non-sensitive software/hardware probe; --strict requires CUDA+PyG."""
import argparse
import importlib
import json
import platform
import sys


def probe():
    report = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torch_geometric"):
        try:
            mod = importlib.import_module(name)
            report[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["device"] = torch.cuda.get_device_name(0)
                report["compute_capability"] = list(torch.cuda.get_device_capability(0))
                x = torch.tensor([1.0], device="cuda")
                report["cuda_tensor_smoke"] = float((x + 1).item()) == 2.0
        except Exception as exc:
            report["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Fail unless torch, PyG, CUDA tensor smoke all work")
    args = parser.parse_args()
    report = probe()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not (isinstance(report.get("torch"), str)
                            and isinstance(report.get("torch_geometric"), str)
                            and report.get("cuda_tensor_smoke") is True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
