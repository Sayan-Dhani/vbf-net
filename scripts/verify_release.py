#!/usr/bin/env python3
"""Verify a VBFNet_Ensemble release against its manifest.

Recomputes the sha256 of every member checkpoint and every shipped source file
and compares them to ``RELEASE_MANIFEST.json``. Exits non-zero on any drift.

Two independent things are checked, and both matter:

* the **weights** — has a checkpoint been swapped, truncated, or left as an
  unfetched git-lfs pointer?
* the **code** — is the decode path you are about to run the one that produced
  the published numbers? The single-model release hashes only its checkpoint and
  calibration files, so it cannot answer this.

Usage
-----
    python3 scripts/verify_release.py
    python3 scripts/verify_release.py --strict      # also load every checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbfnet_ensemble.manifest import (  # noqa: E402
    MANIFEST_NAME,
    load_manifest,
    state_dict_sha256,
    verify_files,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO_ROOT / MANIFEST_NAME))
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--no_code", action="store_true", help="Check weights only.")
    ap.add_argument("--strict", action="store_true",
                    help="Also load each checkpoint and re-hash its state dict.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[verify] no manifest at {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    root = Path(args.root)

    release = manifest.get("release", {})
    shared = manifest.get("shared", {})
    print(
        f"[verify] {release.get('name')} v{release.get('version')} "
        f"({len(manifest.get('members', []))} members, "
        f"config_hash {shared.get('config_hash')}, "
        f"route: member = {manifest.get('routing', {}).get('rule')})"
    )

    problems = verify_files(manifest, root, include_code=not args.no_code)

    if args.strict and not problems:
        import torch

        for member in manifest["members"]:
            path = root / member["file"]
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            digest = state_dict_sha256(ckpt["model"])
            if digest != member["state_dict_sha256"]:
                problems.append(
                    f"{member['file']}: state_dict_sha256 {digest[:16]}… != "
                    f"manifest {member['state_dict_sha256'][:16]}…"
                )
            if "optimiser" in ckpt and member.get("stripped_keys"):
                problems.append(f"{member['file']}: optimiser state present but manifest says stripped")
            del ckpt
            print(f"[verify]   fold {member['fold_id']}: weights OK")

    if problems:
        print(f"\n[verify] FAILED — {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    n_files = len(manifest.get("files", {})) if not args.no_code else 0
    print(f"[verify] OK — {len(manifest['members'])} member(s) and {n_files} source file(s) match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
