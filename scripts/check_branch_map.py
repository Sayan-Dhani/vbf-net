#!/usr/bin/env python3
"""Check that a ROOT file provides every input the model needs, through a branch map.

    python3 scripts/check_branch_map.py --root_file your.root
    python3 scripts/check_branch_map.py --root_file your.root --branch_map my_branch_map.yaml
    python3 scripts/check_branch_map.py --root_file your.root --branch_map my_branch_map.yaml --require_truth

For every branch the model reads it prints the name the model expects, the branch
it will read in your tree, and whether that branch exists with the right shape
(one value per event, or one value per jet). Exits 1 if a required branch is
missing or has the wrong shape. It cannot check that a branch MEANS the same
thing as the model's input; see the comments in branch_map.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbfnet_ensemble.branch_map import check_branch_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_file", required=True)
    ap.add_argument("--branch_map", default=None,
                    help="YAML branch map (default: none, i.e. the model's own names)")
    ap.add_argument("--tree_name", default="Events")
    ap.add_argument("--require_truth", action="store_true",
                    help="Also require the generator-level branches (needed for require_truth=True).")
    args = ap.parse_args()

    rows = check_branch_map(args.root_file, args.branch_map, args.tree_name, args.require_truth)

    def shape(jagged):
        return "-" if jagged is None else ("per jet" if jagged else "per event")

    width = max(len(r["logical"]) for r in rows)
    print(f"{'model expects':<{width}}  {'reads from your tree':<30}  {'status'}")
    n_bad = 0
    for r in rows:
        if r["ok"]:
            status = "ok"
        elif not r["present"]:
            status = "MISSING"
        else:
            status = f"WRONG SHAPE: {shape(r['jagged_found'])}, expected {shape(r['jagged_expected'])}"
        if not r["required"]:
            status += "  (optional: truth only)"
        elif not r["ok"]:
            n_bad += 1
        print(f"{r['logical']:<{width}}  {r['actual']:<30}  {status}")

    if n_bad:
        print(f"\n{n_bad} required branch(es) missing or wrong. Map them in your branch map "
              "(left: the model's name, right: your branch).")
        return 1
    print("\nAll required branches found. Make sure each one means the same thing as the "
          "model's input (objects, algorithm, units).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
