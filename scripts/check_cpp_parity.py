#!/usr/bin/env python3
"""Check that the C++ library reproduces the Python package on a ROOT file.

Runs the compiled ``vbfnet_predict_tree`` (``cpp/``) and ``vbfnet_ensemble.VBFNet``
on the first entries of the same file and compares, event by event:

* the selection (which entries get a prediction) and the routed member: must be identical;
* every target and head in model space, raw and calibrated;
* every key in physical units (targets and the p4-derived observables).

Both sides compute in float32, so the values agree to float32 precision, not
bit for bit. Most differences are ~1e-6 in model space. A few events are more
sensitive: e.g. a jet pair whose invariant mass m_ij comes from E^2 - p^2 with
E >> m loses digits in float32 in BOTH implementations (torch's vector math and
glibc round differently), and the network passes that on. Hence two criteria:
the bulk (99.9 % of values) must agree tightly, the worst value loosely. A
calibrated quantile can also differ by a whole calibration step when the raw
q50 lies within float32 precision of a bin edge; that must stay rare.

Usage
-----
    cmake -S cpp -B cpp/build && cmake --build cpp/build -j
    python3 scripts/export_cpp_weights.py
    python3 scripts/check_cpp_parity.py --root_file /path/to/signal.root --max_entries 2000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TOOL = REPO_ROOT / "cpp" / "build" / "vbfnet_predict_tree"


def run_cpp(args, out_file: Path) -> None:
    cmd = [str(args.tool), "--input", args.root_file, "--output", str(out_file), "--repo", str(REPO_ROOT),
           "--tree", args.tree_name, "--max-entries", str(args.max_entries), "--model-space"]
    if args.targets:
        cmd += ["--sets", ",".join(args.targets)]
    if not args.no_calibrate:
        cmd.append("--calibrate")
    if args.no_acceptance:
        cmd.append("--no-acceptance")
    if args.branch_map:
        cmd += ["--branch-map", args.branch_map]
    print("[parity] C++   :", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_file", required=True)
    ap.add_argument("--tree_name", default="Events")
    ap.add_argument("--max_entries", type=int, default=2000, help="Compare the first N entries of the tree.")
    ap.add_argument("--tool", default=str(DEFAULT_TOOL), help="Path of the compiled vbfnet_predict_tree.")
    ap.add_argument("--targets", nargs="+", default=None, help="Model sets to compare (default: all).")
    ap.add_argument("--branch_map", default=None)
    ap.add_argument("--no_calibrate", action="store_true")
    ap.add_argument("--no_acceptance", action="store_true")
    ap.add_argument("--tol_bulk", type=float, default=5e-4,
                    help="Bound on the 99.9th percentile of the model-space differences.")
    ap.add_argument("--tol_max", type=float, default=1e-2,
                    help="Bound on the largest model-space difference.")
    ap.add_argument("--max_cal_step_fraction", type=float, default=1e-3,
                    help="Largest fraction of calibrated values allowed in a neighbouring calibration bin.")
    args = ap.parse_args()

    if not Path(args.tool).exists():
        print(f"[parity] {args.tool} not found; build it: cmake -S cpp -B cpp/build && cmake --build cpp/build",
              file=sys.stderr)
        return 1

    import uproot

    from vbfnet_ensemble import VBFNet

    calibrate = not args.no_calibrate
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "cpp.root"
        run_cpp(args, out_file)
        with uproot.open(out_file) as f:
            cpp = f["vbfnet"].arrays(library="np")

    accepted = cpp["vbfnet_accepted"].astype(bool)
    entries = cpp["vbfnet_entry"][accepted].astype(np.int64)
    if not len(entries):
        print("[parity] no accepted events in the compared entries", file=sys.stderr)
        return 1

    print(f"[parity] Python: VBFNet(targets={args.targets or 'all'}).predict_root(max_events={len(entries)})",
          flush=True)
    net = VBFNet(targets=args.targets, use_quantile_calibration=calibrate, verbose=False)
    py = net.predict_root(args.root_file, tree_name=args.tree_name, max_events=len(entries), num_workers=0,
                          branch_map=args.branch_map, acceptance=None if args.no_acceptance else "default")

    failures = []
    if not np.array_equal(py["event_index"], entries):
        failures.append("the selected entries differ")
        print(f"[parity] selection differs: Python {len(py['event_index'])} rows, C++ {len(entries)}; "
              f"only in Python: {sorted(set(py['event_index']) - set(entries))[:10]}, "
              f"only in C++: {sorted(set(entries) - set(py['event_index']))[:10]}", file=sys.stderr)
        return 1
    if not np.array_equal(py["fold_id"], cpp["vbfnet_fold"][accepted]):
        failures.append("the routed member differs")

    heads = list(net.head_names)
    print(f"\n[parity] {len(entries)} events compared (of {len(accepted)} entries)\n")
    print(f"{'key':12s} {'set':4s} {'p99.9|dlog|':>11s} {'max|dlog|':>10s} {'max|dphys|':>11s} "
          f"{'max|dphys| cal':>14s} {'cal steps':>9s}")
    all_dlog, n_steps, n_cal = [], 0, 0
    for ti, key in enumerate(net.target_keys):
        def col(name, cal):
            return np.stack([cpp[f"vbfnet_{name}_{h}{'_cal' if cal else ''}"][accepted] for h in heads], axis=1)
        dlog = np.abs(py["pred_log_full"][:, ti, :] - col(f"log_{key}", False))
        dphys = np.abs(np.stack([py["pred_phys"][key][h] for h in heads], 1) - col(key, False))
        all_dlog.append(dlog.reshape(-1))
        line = (f"{key:12s} {net.ensemble_of[key]:4s} {np.nanquantile(dlog, 0.999):11.1e} {np.nanmax(dlog):10.1e}"
                f" {np.nanmax(dphys):11.3g}")
        if calibrate:
            dlog_cal = np.abs(py["pred_log_cal_full"][:, ti, :] - col(f"log_{key}", True))
            dphys_cal = np.abs(np.stack([py["pred_phys_cal"][key][h] for h in heads], 1) - col(key, True))
            # a neighbouring calibration bin: the calibrated value moves by much more than the raw one
            steps = int(np.sum(dlog_cal > dlog + args.tol_bulk))
            n_steps += steps
            n_cal += dlog_cal.size
            line += f" {np.nanmax(dphys_cal):14.3g} {steps:9d}"
        else:
            line += f" {'-':>14s} {'-':>9s}"
        print(line)
    for key in py["pred_phys"]:
        if key in net.target_keys:
            continue
        d = np.abs(np.stack([py["pred_phys"][key][h] for h in heads], 1) -
                   np.stack([cpp[f"vbfnet_{key}_{h}"][accepted] for h in heads], 1))
        line = f"{key:12s} {net.ensemble_of[key]:4s} {'(derived)':>11s} {'':10s} {np.nanmax(d):11.3g}"
        if calibrate:
            dc = np.abs(np.stack([py["pred_phys_cal"][key][h] for h in heads], 1) -
                        np.stack([cpp[f"vbfnet_{key}_{h}_cal"][accepted] for h in heads], 1))
            line += f" {np.nanmax(dc):14.3g}"
        print(line)

    all_dlog = np.concatenate(all_dlog)
    bulk, worst, median = (float(np.nanquantile(all_dlog, q)) for q in (0.999, 1.0, 0.5))
    if bulk > args.tol_bulk:
        failures.append(f"99.9 % of the model-space differences are not below {args.tol_bulk:.0e} ({bulk:.1e})")
    if worst > args.tol_max:
        failures.append(f"largest model-space difference {worst:.1e} > {args.tol_max:.0e}")
    if calibrate and n_steps > args.max_cal_step_fraction * n_cal:
        failures.append(f"{n_steps}/{n_cal} calibrated values moved to another calibration bin")

    print()
    if failures:
        print("[parity] FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    steps = f", {n_steps}/{n_cal} calibrated values in a neighbouring calibration bin" if calibrate else ""
    print(f"[parity] OK: same events and members; model-space differences: median {median:.1e}, "
          f"99.9 % below {bulk:.1e}, largest {worst:.1e}{steps}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
