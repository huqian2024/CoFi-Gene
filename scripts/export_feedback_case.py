#!/usr/bin/env python3
"""Export one local graph-feedback case (B6/B7) for the Week 7 case figure.

    python scripts/export_feedback_case.py \\
        --checkpoint runs/week7/stage1/s1_b7_xfer_drop00_adj_seed42/model.pt \\
        --dataset data/processed/real_v1_1 --channel A --top-k 12 \\
        --out runs/week7/case_b7_A.json

The numbers are produced by the same feedback formula the model trains with
(``cofi_gene.models.feedback.feedback_weights``) on the encoder graph the model is
allowed to see. Nothing is trained or written back to the dataset; the output JSON
is the input of ``scripts/make_figures.py --kind feedback``.

``--focus-genes`` restricts the case to edges incident to those symbols, which is
what a report figure normally wants (otherwise the top-|Δw| edges may come from
four unrelated corners of the graph).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.dataset import CoFiDataset  # noqa: E402
from cofi_gene.manifest import git_commit, sha256_file, utc_timestamp  # noqa: E402
from cofi_gene.perturbation import apply_data_perturbations  # noqa: E402

# ``cofi_gene.evaluate`` / ``feedback_report`` pull in torch only when their
# functions run, and are imported inside ``main()`` for the same reason: this
# module must stay importable (``--help``, tests) on a torch-less CI job.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="model.pt of a B6/B7 run")
    parser.add_argument("--dataset", required=True, help="processed dataset directory")
    parser.add_argument("--channel", default="A", help="channel to report (default A)")
    parser.add_argument("--top-k", type=int, default=12,
                        help="keep this many largest |Δw| edges (default 12)")
    parser.add_argument("--focus-genes", nargs="+", default=None,
                        help="only keep edges incident to these gene symbols")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protocol", default=None,
                        help="optional visibility protocol to apply before reporting")
    parser.add_argument("--edge-removal-mode", default=None,
                        help="optional edge-removal mode to apply before reporting")
    parser.add_argument("--edge-drop-fraction", type=float, default=None)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    import torch

    from cofi_gene.evaluate import build_model_from_checkpoint, load_checkpoint
    from cofi_gene.feedback_report import feedback_edge_table

    dataset = CoFiDataset.load(args.dataset)
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    model, run_config = build_model_from_checkpoint(checkpoint, dataset.num_features,
                                                    device=args.device)

    config = {"data": {}, "runtime": {"seed": int(run_config["runtime"]["seed"])}}
    if args.protocol:
        config["data"]["visibility_protocol"] = args.protocol
    if args.edge_removal_mode:
        config["data"]["edge_removal_mode"] = args.edge_removal_mode
    if args.edge_drop_fraction is not None:
        config["data"]["edge_drop_fraction"] = float(args.edge_drop_fraction)
    prepared, perturbation = apply_data_perturbations(dataset, config)

    device = torch.device(args.device)
    x = torch.as_tensor(prepared.X, dtype=torch.float32, device=device)
    edges = {channel: torch.as_tensor(prepared.channel(channel).encode_edges,
                                      dtype=torch.long, device=device)
             for channel in model.channels}
    weights = {channel: torch.as_tensor(prepared.channel(channel).encode_weights,
                                        dtype=torch.float32, device=device)
               for channel in model.channels}
    model.set_encoding_graphs(edges, weights, prepared.num_nodes)

    case = feedback_edge_table(model, x, args.channel, prepared.num_nodes,
                               gene_ids=prepared.gene_ids, top_k=args.top_k,
                               focus_genes=args.focus_genes)
    npz = Path(args.dataset) / "dataset.npz"
    payload = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "model": model.name,
        "alpha": case["alpha"],
        "margin": case["margin"],
        "dataset": {"processed_dir": str(args.dataset),
                    "sha256": sha256_file(npz) if npz.exists() else None},
        "protocol": {"visibility_protocol": perturbation["visibility_protocol"],
                     "edge_removal_mode": perturbation["edge_removal_mode"],
                     "edge_drop_fraction": perturbation["edge_drop_fraction"]},
        "case": case,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")

    summary = case["summary"]
    print(f"[feedback-case] model={model.name} channel={args.channel} "
          f"selected={summary['num_selected_edges']}/"
          f"{summary['num_candidate_edges']} "
          f"up={summary['num_up']} down={summary['num_down']} "
          f"max|Δw|={summary['max_abs_delta']:.4f} "
          f"(α={case['alpha']}, m={case['margin']})")
    print(f"[feedback-case] wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
