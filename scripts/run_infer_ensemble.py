#!/usr/bin/env python3
"""Minimal usage example: predict one ROOT file with the fold-routed release.

    python3 scripts/run_infer_ensemble.py --root_file data/sig_VBF.root --max_events 500
    python3 scripts/run_infer_ensemble.py --root_file data/sig_VBF.root --calibrate
    python3 scripts/run_infer_ensemble.py --root_file other.root --branch_map my_branch_map.yaml

Each event is predicted by ONE member, k = event % 5 (see vbfnet_ensemble.routing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbfnet_ensemble import VBFNetEnsemble  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_file", required=True, nargs="+")
    ap.add_argument("--tree_name", default="Events")
    ap.add_argument("--branch_map", default=None,
                    help="YAML branch map for files whose branches have other names (see branch_map.yaml).")
    ap.add_argument("--max_events", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="Switch quantile calibration ON (each row with its routed member's tables).")
    args = ap.parse_args()

    net = VBFNetEnsemble(
        checkpoint=args.models, device=args.device, use_quantile_calibration=args.calibrate,
    )

    out = net.predict_root(
        root_files=args.root_file,
        tree_name=args.tree_name,
        max_events=args.max_events,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        decode=True,
        branch_map=args.branch_map,
        require_truth=False,
    )

    n = out["pred_log_full"].shape[0]
    print(f"\nevents predicted: {n}  (acceptance: {out['acceptance']})")
    print(f"route: member = {out['route']}   members: {out['member_ids']}")
    print("events per member: "
          + ", ".join(f"fold{k}={int(np.sum(out['fold_id'] == k))}" for k in out["member_ids"]))
    print(f"quantile crossing rate: {out['quantile_crossing_rate']:.3g}  (expect 0)")

    # Quantiles: use the calibrated dict when calibration is on. The point head
    # is identical in both, so point-only consumers can ignore the switch.
    pred_dict = out["pred_phys_cal"] if "pred_phys_cal" in out else out["pred_phys"]
    label = "calibrated" if "pred_phys_cal" in out else "raw"
    print(f"\nquantiles: {label}")
    print(f"\n{'target':8s} {'point':>10s} {'q16':>10s} {'q50':>10s} {'q84':>10s} {'band/2':>10s}")
    for key in net.target_keys:
        pred = pred_dict[key]
        half_band = 0.5 * (pred["q84"] - pred["q16"])          # ~1 sigma, per event
        print(
            f"{key:8s} {np.median(pred['point']):10.2f} {np.median(pred['q16']):10.2f} "
            f"{np.median(pred['q50']):10.2f} {np.median(pred['q84']):10.2f} "
            f"{np.median(half_band):10.2f}"
        )

    print("\nderived observables (median over events, point head):")
    for key in ("mjj", "deta", "eta_prod", "ptsum"):
        if key in out["pred_phys"]:
            print(f"  {key:10s} {np.median(out['pred_phys'][key]['point']):12.2f}")

    print(f"\nfirst rows: event_index {out['event_index'][:5]}  event {out['event'][:5]}  "
          f"fold {out['fold_id'][:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
