#!/usr/bin/env python3
"""Verify a VBFNet_Ensemble release against its manifests.

Recomputes the sha256 of every shipped source file, every model-set manifest,
every member checkpoint and every calibration file, and compares them to the
manifests. Exits non-zero on any drift.

Two independent things are checked, and both matter:

* the **weights** — has a checkpoint been swapped, truncated, or left as an
  unfetched git-lfs pointer?
* the **code** — is the decode path you are about to run the one that produced
  the published numbers? The single-model release hashes only its checkpoint and
  calibration files, so it cannot answer this.

The top-level ``RELEASE_MANIFEST.json`` hashes the code and pins each set
manifest (``ensembles/<set>/RELEASE_MANIFEST.json``), which hashes that set's
members and calibration. A set manifest can also be checked on its own with
``--manifest``/``--root``.

Usage
-----
    python3 scripts/verify_release.py
    python3 scripts/verify_release.py --strict          # also load every checkpoint
    python3 scripts/verify_release.py --ensemble hl     # one set (and the code)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbfnet_ensemble.manifest import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    MANIFEST_NAME,
    PACKAGE_SCHEMA_VERSION,
    load_manifest,
    state_dict_sha256,
    verify_files,
    verify_package,
)


def strict_check(manifest: dict, root: Path, label: str) -> list[str]:
    """Load every checkpoint of one set and re-hash its state dict."""
    import torch

    problems: list[str] = []
    for member in manifest["members"]:
        path = root / member["file"]
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        digest = state_dict_sha256(ckpt["model"])
        if digest != member["state_dict_sha256"]:
            problems.append(
                f"{label}{member['file']}: state_dict_sha256 {digest[:16]}… != "
                f"manifest {member['state_dict_sha256'][:16]}…"
            )
        if "optimiser" in ckpt and member.get("stripped_keys"):
            problems.append(f"{label}{member['file']}: optimiser state present but manifest says stripped")
        del ckpt
        print(f"[verify]   {label}fold {member['fold_id']}: weights OK")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO_ROOT / MANIFEST_NAME))
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--ensemble", action="append", default=None,
                    help="Check only this model set (repeatable). Default: every set.")
    ap.add_argument("--no_code", action="store_true", help="Check weights only.")
    ap.add_argument("--strict", action="store_true",
                    help="Also load each checkpoint and re-hash its state dict.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[verify] no manifest at {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path, schema=None)
    root = Path(args.root)
    release = manifest.get("release", {})

    if int(manifest.get("schema_version", 0)) == PACKAGE_SCHEMA_VERSION:
        index = manifest.get("ensembles", {}) or {}
        order = sorted(index, key=lambda n: (n != DEFAULT_ENSEMBLE, n))   # p4 first, as in the package
        names = order if args.ensemble is None else list(args.ensemble)
        print(
            f"[verify] {release.get('name')} v{release.get('version')} "
            f"({len(index)} model sets: {', '.join(order)})"
        )
        for name in names:
            entry = index.get(name, {})
            print(f"[verify]   set {name}: {entry.get('n_members')} members, "
                  f"config_hash {entry.get('config_hash')}, route: member = {entry.get('routing')}, "
                  f"targets {entry.get('target_keys')}")
        problems = verify_package(manifest, root, include_code=not args.no_code, ensembles=names)
        if args.strict and not problems:
            for name in names:
                sub_root = root / index[name]["dir"]
                problems += strict_check(load_manifest(sub_root / MANIFEST_NAME), sub_root, f"[{name}] ")
        n_members = sum(int(index[n].get("n_members") or 0) for n in names if n in index)
    else:
        shared = manifest.get("shared", {})
        print(
            f"[verify] {release.get('name')} v{release.get('version')} "
            f"({len(manifest.get('members', []))} members, "
            f"config_hash {shared.get('config_hash')}, "
            f"route: member = {manifest.get('routing', {}).get('rule')})"
        )
        problems = verify_files(manifest, root, include_code=not args.no_code)
        if args.strict and not problems:
            problems += strict_check(manifest, root, "")
        n_members = len(manifest.get("members", []))

    if problems:
        print(f"\n[verify] FAILED — {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    n_files = len(manifest.get("files", {})) if not args.no_code else 0
    print(f"[verify] OK — {n_members} member(s) and {n_files} source file(s) match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
