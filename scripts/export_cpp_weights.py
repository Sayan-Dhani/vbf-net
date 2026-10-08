#!/usr/bin/env python3
"""Export a model set to the weight file read by the C++ library (``cpp/``).

The C++ library cannot read PyTorch checkpoints, so each set is converted once
into ONE binary file holding everything the C++ side needs:

* the weights of all K members (float32, exactly the values in the ``.pt``);
* each member's calibration tables (float64, exactly the values in the JSON);
* the metadata that defines how the members are used: targets and their decode
  rules, heads, feature names and order, architecture switches, routing, the
  default acceptance gate, and the sha256 of every source file.

Nothing is fitted or rounded: the export is a change of container. Every source
file is checked against the set's ``RELEASE_MANIFEST.json`` BEFORE it is read,
so a file can only be exported from the released checkpoints and calibrations.
The output is deterministic (no timestamps), so two exports of the same release
are byte-identical and ``--check`` can verify an existing file.

Needs only ``torch`` and ``numpy`` (no torch-geometric).

Usage
-----
    python3 scripts/export_cpp_weights.py                  # every set -> ensembles/<set>/cpp/
    python3 scripts/export_cpp_weights.py --ensemble hl    # one set
    python3 scripts/export_cpp_weights.py --check          # verify existing files, write nothing

File layout (little-endian)
---------------------------
    char[8]   magic "VBFNETW\\0"
    uint32    format version (FORMAT_VERSION)
    uint64    length of the metadata text, then the text (UTF-8, one
              "key<TAB>value<TAB>value..." record per line)
    uint64    number of tensors, then for each tensor:
                uint32 name length, name (UTF-8)
                uint8  dtype (1 = float32, 2 = float64)
                uint8  ndim, then uint64 shape[ndim]
                the values, row-major
    char[8]   end marker "VBFNETE\\0"

Tensor names: ``member<k>.<state_dict key>`` (the PyTorch parameter names) and
``calib<k>.<target>.<head>.edges`` / ``.content`` (one clamped binning on the
member's raw physical q50 per target and quantile head).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

MAGIC = b"VBFNETW\0"
END_MAGIC = b"VBFNETE\0"
FORMAT_VERSION = 1
EXPORTER_VERSION = "1.0.0"
OUTPUT_DIRNAME = "cpp"
DTYPE_CODES = {np.dtype("float32"): 1, np.dtype("float64"): 2}

#: The calibration JSON written for every target of a member.
CALIBRATION_SUFFIX = "_local_shift_correctionlib.json"

#: Defaults of nn.LayerNorm / nn.BatchNorm1d, which PyGVBFGNN uses unchanged.
LAYERNORM_EPS = 1e-5
BATCHNORM_EPS = 1e-5


def _load_module(name: str, path: Path):
    """Load one pure-Python module of the package by path.

    ``import vbfnet_ensemble`` would pull in torch-geometric through the package
    ``__init__``; ``manifest.py`` and ``transforms.py`` need only the standard
    library and numpy, and are the single source of the hashing and decode rules.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest_mod = _load_module("_vbfnet_manifest", REPO_ROOT / "vbfnet_ensemble" / "manifest.py")
transforms_mod = _load_module("_vbfnet_transforms", REPO_ROOT / "vbfnet_ensemble" / "transforms.py")


class ExportError(RuntimeError):
    pass


def default_output(set_dir: Path, name: str) -> Path:
    return set_dir / OUTPUT_DIRNAME / f"vbfnet_{name}.bin"


def _checked_sha256(path: Path, expected: str, what: str) -> str:
    if not path.exists():
        raise ExportError(f"{what} {path} is missing")
    digest = manifest_mod.sha256_file(path)
    if digest != expected:
        raise ExportError(
            f"{what} {path} does not match the release manifest "
            f"(sha256 {digest[:16]}… != {expected[:16]}…). Refusing to export it."
        )
    return digest


def _same(values, what: str):
    """All members must agree on ``what``; return the shared value."""
    first = json.dumps(values[0], sort_keys=True)
    for k, v in enumerate(values[1:], start=1):
        if json.dumps(v, sort_keys=True) != first:
            raise ExportError(f"Member {k} differs from member 0 in {what}.")
    return values[0]


def _binning(correction: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(edges, content)`` of a flat clamped binning node (the shape the fit writes)."""
    data = correction.get("data", {})
    name = correction.get("name", "?")
    if data.get("nodetype") != "binning" or data.get("flow") != "clamp":
        raise ExportError(f"{name}: expected a clamped binning node, got "
                          f"{data.get('nodetype')!r}/{data.get('flow')!r}")
    content = data["content"]
    if not all(isinstance(c, (int, float)) for c in content):
        raise ExportError(f"{name}: nested content is not supported")
    edges = np.asarray(data["edges"], dtype=np.float64)
    content = np.asarray(content, dtype=np.float64)
    if edges.ndim != 1 or len(edges) != len(content) + 1 or np.any(np.diff(edges) <= 0):
        raise ExportError(f"{name}: malformed edges ({len(edges)}) / content ({len(content)})")
    return edges, content


def _meta_line(key: str, *values) -> str:
    fields = [str(key)] + [str(v) for v in values]
    for f in fields:
        if "\t" in f or "\n" in f:
            raise ExportError(f"metadata field {f!r} contains a tab or newline")
    return "\t".join(fields)


def build_export(set_dir: Path, name: str, package_version: str) -> bytes:
    """Read one set (verified against its manifest) and return the file content."""
    import torch

    manifest_path = set_dir / manifest_mod.MANIFEST_NAME
    manifest = manifest_mod.load_manifest(manifest_path)
    members = sorted(manifest.get("members", []), key=lambda m: int(m["fold_id"]))
    n_folds = int((manifest.get("routing", {}) or {}).get("n_folds") or len(members))
    if [int(m["fold_id"]) for m in members] != list(range(n_folds)):
        raise ExportError(
            f"{manifest_path}: routing needs exactly one member per fold 0..{n_folds - 1}; "
            f"got {[m['fold_id'] for m in members]}."
        )

    # ── members ──────────────────────────────────────────────────────────────
    tensors: list[tuple[str, np.ndarray]] = []
    metas = []
    member_lines = []
    for m in members:
        k = int(m["fold_id"])
        path = set_dir / m["file"]
        manifest_mod.check_not_lfs_pointer(path)  # names git-lfs instead of a hash mismatch
        _checked_sha256(path, m["sha256"], "Checkpoint")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        args_fold = (ckpt.get("args", {}) or {}).get("fold_id")
        if args_fold is not None and int(args_fold) != k:
            raise ExportError(f"{path}: trained as fold {args_fold}, manifest says fold {k}.")
        for key, t in ckpt["model"].items():
            if key.endswith("num_batches_tracked"):
                continue  # BatchNorm bookkeeping, unused at inference
            if t.dtype != torch.float32:
                raise ExportError(f"{path}: {key} is {t.dtype}, expected float32")
            tensors.append((f"member{k}.{key}", t.detach().cpu().contiguous().numpy()))
        metas.append({
            "config": {s: ckpt["config"].get(s) for s in ("model", "output", "features", "targets")},
            "head_names": list(ckpt["head_names"]),
            "target_specs": list(ckpt["target_specs"]),
            "keys": sorted(k_ for k_ in ckpt["model"] if not k_.endswith("num_batches_tracked")),
            "features": [list(ckpt[f"{g}_feature_names"]) for g in ("node", "edge", "global")],
        })
        member_lines.append(_meta_line("member", k, m["sha256"], m["file"], m.get("epoch", "?")))
        del ckpt

    cfg = _same([x["config"] for x in metas], "config (model/output/features/targets)")
    head_names = _same([x["head_names"] for x in metas], "head_names")
    specs = _same([x["target_specs"] for x in metas], "target_specs")
    _same([x["keys"] for x in metas], "parameter names")
    node_f, edge_f, global_f = _same([x["features"] for x in metas], "feature names")

    target_keys = transforms_mod._target_keys_from_specs(specs)
    decode_rules = [transforms_mod._decode_rule_for_spec(s) for s in specs]
    for key, rule in zip(target_keys, decode_rules):
        if rule not in {"identity", "sinh", "expm1", "signed_expm1", "exp"}:
            raise ExportError(f"target {key}: decode rule {rule!r} is not supported by the C++ library")
    shared = manifest.get("shared", {}) or {}
    if list(shared.get("target_keys", target_keys)) != target_keys:
        raise ExportError(f"{manifest_path}: target_keys differ from the checkpoints'")

    model = cfg["model"]
    aggregation = model.get("aggregation", ["sum", "mean", "max"])
    if isinstance(aggregation, str):
        aggregation = [aggregation]
    norm = model.get("norm", "LayerNorm")
    acc = manifest.get("acceptance", None) or {"min_jets": 2, "jet_min_pt": 50.0, "jet_max_abs_eta": 4.7}

    # ── calibration ──────────────────────────────────────────────────────────
    cal = manifest.get("calibration", {}) or {}
    cal_lines = []
    q_heads = [h for h in head_names if str(h).startswith("q")]
    if cal.get("shipped"):
        released = {int(m["fold_id"]): m.get("source_sha256") for m in members}
        for k in range(n_folds):
            entry = (cal.get("members", {}) or {}).get(str(k))
            if entry is None:
                raise ExportError(f"{manifest_path}: no calibration for fold {k}")
            if entry.get("member_source_sha256") != released[k]:
                raise ExportError(f"calibration of fold {k} was fitted on another checkpoint")
            files = entry.get("files", {}) or {}
            for target in target_keys:
                rel = f"{entry['dir']}/{target}{CALIBRATION_SUFFIX}"
                if rel not in files:
                    raise ExportError(f"{manifest_path}: {rel} is not in the manifest")
                path = set_dir / rel
                digest = _checked_sha256(path, files[rel], "Calibration file")
                by_name = {c["name"]: c for c in json.loads(path.read_text()).get("corrections", [])}
                for head in q_heads:
                    corr = by_name.get(f"{target}_local_shift_{head}")
                    if corr is None:
                        raise ExportError(f"{rel}: no correction {target}_local_shift_{head}")
                    edges, content = _binning(corr)
                    tensors.append((f"calib{k}.{target}.{head}.edges", edges))
                    tensors.append((f"calib{k}.{target}.{head}.content", content))
                cal_lines.append(_meta_line("calibration_file", k, digest, rel))

    # ── metadata ─────────────────────────────────────────────────────────────
    lines = [
        _meta_line("format", "vbfnet-cpp-weights", FORMAT_VERSION),
        _meta_line("exporter", "scripts/export_cpp_weights.py", EXPORTER_VERSION),
        _meta_line("package_version", package_version),
        _meta_line("set", name),
        _meta_line("set_manifest_sha256", manifest_mod.sha256_file(manifest_path)),
        _meta_line("config_hash", shared.get("config_hash", "?")),
        _meta_line("n_folds", n_folds),
        _meta_line("route", "event % n_folds"),
        _meta_line("targets", *target_keys),
        _meta_line("decode", *decode_rules),
        _meta_line("heads", *head_names),
        _meta_line("node_features", *node_f),
        _meta_line("edge_features", *edge_f),
        _meta_line("global_features", *global_f),
        _meta_line("model.n_layers", int(model.get("n_layers", 6))),
        _meta_line("model.aggregation", *[str(a).lower() for a in aggregation]),
        _meta_line("model.pool", str(model.get("pool", "mean+max")).lower().strip()),
        _meta_line("model.activation", str(model.get("activation", "GELU")).lower()),
        _meta_line("model.norm", "none" if norm is None else str(norm).lower() or "none"),
        _meta_line("model.dropout_layers", int(float(model.get("dropout", 0.0)) > 0.0)),
        _meta_line("model.input_batchnorm", int(bool(model.get("input_batchnorm", False)))),
        _meta_line("model.use_edge_pair_summary", int(bool(model.get("use_edge_pair_summary", True)))),
        _meta_line("model.layernorm_eps", repr(LAYERNORM_EPS)),
        _meta_line("model.batchnorm_eps", repr(BATCHNORM_EPS)),
        _meta_line("acceptance", int(acc["min_jets"]), repr(float(acc["jet_min_pt"])),
                   repr(float(acc["jet_max_abs_eta"]))),
        _meta_line("calibration", "per_member" if cal_lines else "none"),
        *member_lines,
        *cal_lines,
    ]
    meta = ("\n".join(lines) + "\n").encode("utf-8")

    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<I", FORMAT_VERSION))
    buf.write(struct.pack("<Q", len(meta)))
    buf.write(meta)
    buf.write(struct.pack("<Q", len(tensors)))
    for tname, arr in tensors:
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in DTYPE_CODES:
            raise ExportError(f"{tname}: unsupported dtype {arr.dtype}")
        raw = tname.encode("utf-8")
        buf.write(struct.pack("<I", len(raw)))
        buf.write(raw)
        buf.write(struct.pack("<BB", DTYPE_CODES[arr.dtype], arr.ndim))
        buf.write(struct.pack(f"<{arr.ndim}Q", *arr.shape))
        buf.write(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    buf.write(END_MAGIC)
    return buf.getvalue()


def read_export_meta(path: str | Path) -> dict[str, list[list[str]]]:
    """The metadata records of a weight file, without reading the tensors."""
    with open(path, "rb") as fh:
        head = fh.read(20)
        if head[:8] != MAGIC:
            raise ExportError(f"{path}: not a VBF-Net C++ weight file")
        (version,) = struct.unpack_from("<I", head, 8)
        if version != FORMAT_VERSION:
            raise ExportError(f"{path}: format version {version}, this script reads {FORMAT_VERSION}")
        (meta_len,) = struct.unpack_from("<Q", head, 12)
        text = fh.read(meta_len).decode("utf-8")
    meta: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        if line:
            key, *values = line.split("\t")
            meta.setdefault(key, []).append(values)
    return meta


def check_against_manifest(path: str | Path, set_dir: Path) -> list[str]:
    """Problems that make an exported file stale for the set in ``set_dir`` (fast: header only)."""
    manifest_path = set_dir / manifest_mod.MANIFEST_NAME
    manifest = manifest_mod.load_manifest(manifest_path)
    try:
        meta = read_export_meta(path)
    except (OSError, ExportError) as exc:
        return [str(exc)]
    problems = []
    if meta.get("set_manifest_sha256", [[None]])[0][0] != manifest_mod.sha256_file(manifest_path):
        problems.append("was exported from another version of the set manifest")
    exported = {int(r[0]): r[1] for r in meta.get("member", [])}
    released = {int(m["fold_id"]): m["sha256"] for m in manifest.get("members", [])}
    if exported != released:
        problems.append("its members are not the released checkpoints")
    cal = (manifest.get("calibration", {}) or {})
    files = {rel: digest for e in (cal.get("members", {}) or {}).values() for rel, digest in (e.get("files", {}) or {}).items()}
    for _, digest, rel in meta.get("calibration_file", []):
        if files.get(rel) != digest:
            problems.append(f"its calibration {rel} is not the released one")
    if bool(cal.get("shipped")) != (meta.get("calibration", [["none"]])[0][0] == "per_member"):
        problems.append("calibration shipped in the release but not in the file, or vice versa")
    return [f"{Path(path).name}: {p}; re-run scripts/export_cpp_weights.py" for p in problems]


def read_export(path: str | Path) -> tuple[dict[str, list[list[str]]], dict[str, np.ndarray]]:
    """Read a weight file back: ``(metadata records by key, tensors by name)``.

    Used by the tests and by ``verify_release.py`` (which reads only the
    metadata). Records are lists of fields, since some keys repeat (``member``).
    """
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ExportError(f"{path}: not a VBF-Net C++ weight file")
    (version,) = struct.unpack_from("<I", data, 8)
    if version != FORMAT_VERSION:
        raise ExportError(f"{path}: format version {version}, this script reads {FORMAT_VERSION}")
    (meta_len,) = struct.unpack_from("<Q", data, 12)
    pos = 20 + meta_len
    meta: dict[str, list[list[str]]] = {}
    for line in data[20:pos].decode("utf-8").splitlines():
        if line:
            key, *values = line.split("\t")
            meta.setdefault(key, []).append(values)
    (n,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    tensors: dict[str, np.ndarray] = {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", data, pos)
        tname = data[pos + 4 : pos + 4 + ln].decode("utf-8")
        pos += 4 + ln
        code, ndim = struct.unpack_from("<BB", data, pos)
        pos += 2
        shape = struct.unpack_from(f"<{ndim}Q", data, pos)
        pos += 8 * ndim
        dtype = np.dtype("<f4") if code == 1 else np.dtype("<f8")
        count = int(np.prod(shape)) if ndim else 1
        tensors[tname] = np.frombuffer(data, dtype=dtype, count=count, offset=pos).reshape(shape)
        pos += count * dtype.itemsize
    if data[pos:] != END_MAGIC:
        raise ExportError(f"{path}: truncated or corrupt (no end marker)")
    return meta, tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", action="append", default=None,
                    help="Model set to export (repeatable). Default: every set.")
    ap.add_argument("--ensembles_dir", default=str(REPO_ROOT / manifest_mod.ENSEMBLES_DIRNAME),
                    help="Directory holding one directory per set.")
    ap.add_argument("--out", default=None,
                    help="Output file (one set only). Default: ensembles/<set>/cpp/vbfnet_<set>.bin")
    ap.add_argument("--check", action="store_true",
                    help="Re-export in memory and compare with the existing files; write nothing.")
    args = ap.parse_args()

    root = Path(args.ensembles_dir)
    names = args.ensemble or manifest_mod.available_ensembles(root)
    if not names:
        print(f"[export] no model sets in {root}", file=sys.stderr)
        return 1
    if args.out and len(names) != 1:
        print("[export] --out needs exactly one --ensemble", file=sys.stderr)
        return 1

    top = REPO_ROOT / manifest_mod.MANIFEST_NAME
    package_version = (
        json.loads(top.read_text()).get("release", {}).get("version", "?") if top.exists() else "?"
    )

    failed = 0
    for name in names:
        set_dir = manifest_mod.ensemble_dir(name, root)
        out = Path(args.out) if args.out else default_output(set_dir, name)
        try:
            blob = build_export(set_dir, name, package_version)
        except ExportError as exc:
            print(f"[export] {name}: FAILED — {exc}", file=sys.stderr)
            failed += 1
            continue
        if args.check:
            if not out.exists():
                print(f"[export] {name}: {out} does not exist", file=sys.stderr)
                failed += 1
            elif out.read_bytes() != blob:
                print(f"[export] {name}: {out} differs from a fresh export of the release", file=sys.stderr)
                failed += 1
            else:
                print(f"[export] {name}: {out} matches the release")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(out)
        print(f"[export] {name}: wrote {out} ({len(blob) / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
