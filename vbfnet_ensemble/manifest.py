"""Release manifests: hashing, loading and verification.

The package ships several model sets ("ensembles"), each a complete fold-routed
k-fold release in its own directory ``ensembles/<name>/`` with its own
``RELEASE_MANIFEST.json`` (schema 1). A set manifest pins the **weights**, by
``sha256`` of each file and by ``state_dict_sha256``, a hash over the tensor
contents themselves (the file hash changes whenever the file is rewritten, e.g.
by stripping ``optimiser``; the state-dict hash does not, so it is what proves
the weights survived a rebuild), and the set's calibration files. Its paths are
relative to the set directory.

The top-level ``RELEASE_MANIFEST.json`` (schema 2) pins the **code**, by
``sha256`` of every shipped source and doc file, and indexes the sets, pinning
each set manifest by its ``sha256``. Verifying it therefore checks everything:
code, set manifests, weights and calibrations.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Iterable

MANIFEST_NAME = "RELEASE_MANIFEST.json"
#: Schema of a set manifest (``ensembles/<name>/RELEASE_MANIFEST.json``).
SCHEMA_VERSION = 1
#: Schema of the top-level package manifest that indexes the sets.
PACKAGE_SCHEMA_VERSION = 2

#: Directory, relative to the repository root, that holds one directory per set.
ENSEMBLES_DIRNAME = "ensembles"
#: The set :class:`~vbfnet_ensemble.predictor.VBFNetEnsemble` loads when none is
#: named, and the one listed first: the p4 regressor of v4.0.0 (GitHub v1.0.0).
DEFAULT_ENSEMBLE = "p4"


def package_root() -> Path:
    """The repository root: the directory that holds ``vbfnet_ensemble/``."""
    return Path(__file__).resolve().parent.parent


def available_ensembles(ensembles_dir: str | Path | None = None) -> list[str]:
    """Names of the sets in ``ensembles_dir``, the default set first, then by name.

    A set is a directory with a manifest. The order is the order of the targets
    in a merged prediction, so the p4 components keep their v4.0.0 positions.
    """
    root = Path(ensembles_dir) if ensembles_dir is not None else package_root() / ENSEMBLES_DIRNAME
    if not root.is_dir():
        return []
    names = [d.name for d in root.iterdir() if (d / MANIFEST_NAME).is_file()]
    return sorted(names, key=lambda n: (n != DEFAULT_ENSEMBLE, n))


def ensemble_dir(name: str | None = None, ensembles_dir: str | Path | None = None) -> Path:
    """Directory of set ``name`` (default :data:`DEFAULT_ENSEMBLE`); raises if absent."""
    name = DEFAULT_ENSEMBLE if name is None else str(name)
    root = Path(ensembles_dir) if ensembles_dir is not None else package_root() / ENSEMBLES_DIRNAME
    path = root / name
    if not (path / MANIFEST_NAME).is_file():
        raise ValueError(
            f"No model set {name!r}: {path / MANIFEST_NAME} does not exist. "
            f"Available sets: {available_ensembles(root)}"
        )
    return path

#: A git-lfs pointer file is a few hundred bytes of text. A real checkpoint is
#: tens of MB. Anything below this is a pointer, not a model.
LFS_POINTER_MAX_BYTES = 1024


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def state_dict_sha256(state: dict) -> str:
    """Hash the tensor *contents* of a state dict, independent of file framing.

    Stable across a re-save, so it survives stripping ``optimiser`` and is the
    invariant ``build_release.py`` checks after rewriting a checkpoint.
    """
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        tensor = state[name]
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def architecture_fingerprint(state: dict) -> str:
    """Hash parameter names/shapes/dtypes only — not values.

    Two members can share a ``config_hash`` and still have been produced by
    different code revisions. This catches that; ``config_hash`` cannot.
    """
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        tensor = state[name]
        h.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode())
    return h.hexdigest()


def check_not_lfs_pointer(path: str | Path) -> None:
    """Raise a message naming git-lfs if ``path`` is an unfetched pointer.

    Without this the failure is an opaque pickle error from ``torch.load`` on a
    130-byte text file, which sends people looking in the wrong place.
    """
    path = Path(path)
    size = path.stat().st_size
    if size > LFS_POINTER_MAX_BYTES:
        return
    try:
        head = path.read_text(errors="ignore")[:200]
    except OSError:
        head = ""
    if "git-lfs" in head or size < LFS_POINTER_MAX_BYTES:
        raise RuntimeError(
            f"{path} is {size} B — this is a git-lfs pointer, not a checkpoint.\n"
            "The model weights were not fetched. Run:\n"
            "    git lfs install && git lfs pull\n"
            "(or clone without GIT_LFS_SKIP_SMUDGE=1)."
        )


def load_manifest(path: str | Path, schema: int | None = SCHEMA_VERSION) -> dict:
    """Load a manifest and check its schema (``None`` accepts schema 1 and 2)."""
    with open(path, "r") as fh:
        manifest = json.load(fh)
    version = int(manifest.get("schema_version", 0))
    expected = (SCHEMA_VERSION, PACKAGE_SCHEMA_VERSION) if schema is None else (schema,)
    if version not in expected:
        raise ValueError(
            f"{path}: manifest schema_version {version}, this code expects "
            f"{' or '.join(map(str, expected))}."
        )
    return manifest


def verify_files(
    manifest: dict,
    root: str | Path,
    *,
    include_code: bool = True,
) -> list[str]:
    """Recompute every hash in the manifest. Returns a list of problems."""
    root = Path(root)
    problems: list[str] = []

    for member in manifest.get("members", []):
        rel = member["file"]
        path = root / rel
        if not path.exists():
            problems.append(f"missing member file: {rel}")
            continue
        size = path.stat().st_size
        if size <= LFS_POINTER_MAX_BYTES:
            problems.append(
                f"{rel} is {size} B — a git-lfs pointer, not a checkpoint. "
                "Run `git lfs pull`."
            )
            continue
        if size != int(member.get("size_bytes", size)):
            problems.append(
                f"{rel}: size {size} != manifest {member['size_bytes']}"
            )
        digest = sha256_file(path)
        if digest != member["sha256"]:
            problems.append(f"{rel}: sha256 {digest[:16]}… != manifest {member['sha256'][:16]}…")

    calibration = manifest.get("calibration", {}) or {}
    if calibration.get("shipped"):
        released = {str(m["fold_id"]): m.get("source_sha256") for m in manifest.get("members", [])}
        for fold, entry in (calibration.get("members", {}) or {}).items():
            if entry.get("member_source_sha256") != released.get(str(fold)):
                problems.append(
                    f"calibration for fold {fold} was fitted on a different checkpoint "
                    "than the released member — use the calibrations/ and models/ of the same release"
                )
            for rel, expected in (entry.get("files", {}) or {}).items():
                path = root / rel
                if not path.exists():
                    problems.append(f"missing calibration file: {rel}")
                elif sha256_file(path) != expected:
                    problems.append(f"{rel}: calibration sha256 does not match the manifest")

    if include_code:
        for rel, expected in (manifest.get("files", {}) or {}).items():
            path = root / rel
            if not path.exists():
                problems.append(f"missing tracked file: {rel}")
                continue
            digest = sha256_file(path)
            if digest != expected:
                problems.append(
                    f"{rel}: sha256 {digest[:16]}… != manifest {expected[:16]}… "
                    "(shipped code has been modified since the release was built)"
                )

    return problems


def provenance_text(manifest: dict) -> str:
    """Render the manifest in the grep-able ``<sha256>  <path>`` form.

    The single-model release ships ``PROVENANCE.txt`` in this format; keeping it
    means existing habits and scripts still work against the new package.
    """
    shared = manifest.get("shared", {})
    release = manifest.get("release", {})
    lines = [
        f"release: {release.get('name', 'VBFNet_ensemble')} "
        f"v{release.get('version', '?')} "
        f"({len(manifest.get('members', []))} members, "
        f"route: member = {manifest.get('routing', {}).get('rule', '?')})",
        f"config_hash: {shared.get('config_hash', '?')}   "
        f"acceptance: {manifest.get('acceptance', '?')}",
        f"built: {release.get('created', '?')}",
        "",
    ]
    for member in manifest.get("members", []):
        lines.append(
            f"# fold {member.get('fold_id')}  epoch {member.get('epoch')}  "
            f"selection_score {member.get('selection_score')}  "
            f"source {member.get('source_run')}"
        )
        lines.append(f"{member['sha256']}  {member['file']}")
    calibration = manifest.get("calibration", {}) or {}
    if calibration.get("shipped"):
        lines.append("")
        lines.append(
            f"# calibration: {calibration.get('layout')} {calibration.get('type')}, "
            f"n_bins {calibration.get('n_bins')}, min_events {calibration.get('min_events')}"
        )
        for fold, entry in sorted((calibration.get("members", {}) or {}).items()):
            lines.append(f"# fold {fold}: fitted on {entry.get('n_fit_events')} held-out events "
                         f"from {entry.get('fit_export')}")
            for rel, digest in sorted(entry.get("files", {}).items()):
                lines.append(f"{digest}  {rel}")
    lines.append("")
    for rel, digest in sorted((manifest.get("files", {}) or {}).items()):
        lines.append(f"{digest}  {rel}")
    return "\n".join(lines) + "\n"


#: Source directories of the C++ library under ``cpp/`` (its ``build/`` is not shipped).
CPP_SOURCE_DIRS = ("include", "src", "examples")


def iter_code_files(root: str | Path) -> Iterable[Path]:
    """The shipped files whose hashes go into ``manifest["files"]``."""
    root = Path(root)
    cpp = root / "cpp"
    for rel in sorted(
        [p.relative_to(root) for p in (root / "vbfnet_ensemble").rglob("*.py")]
        + [p.relative_to(root) for p in (root / "scripts").rglob("*.py")]
        + [p.relative_to(root) for p in (root / "example_bdt_input").glob("*")
           if p.suffix in {".py", ".yaml", ".md"}]
        + [p.relative_to(root) for d in CPP_SOURCE_DIRS for p in (cpp / d).rglob("*") if p.is_file()]
        + [p.relative_to(root) for p in (cpp / "CMakeLists.txt", cpp / "README.md") if p.is_file()]
    ):
        if "__pycache__" in rel.parts:
            continue
        yield rel
    for name in ("README.md", "MODEL_CARD.md"):
        if (root / name).exists():
            yield Path(name)


# ── the package manifest: code hashes + an index of the sets ─────────────────

def ensemble_index(root: str | Path) -> dict:
    """The ``ensembles`` section of the package manifest, read from the sets on disk.

    Pins each set manifest by its sha256; the set manifest in turn pins the
    set's weights and calibration, so the chain covers every shipped file.
    """
    root = Path(root)
    index: dict = {}
    for name in available_ensembles(root / ENSEMBLES_DIRNAME):
        rel_dir = Path(ENSEMBLES_DIRNAME) / name
        path = root / rel_dir / MANIFEST_NAME
        sub = load_manifest(path)
        shared = sub.get("shared", {}) or {}
        index[name] = {
            "dir": str(rel_dir),
            "manifest": str(rel_dir / MANIFEST_NAME),
            "manifest_sha256": sha256_file(path),
            "target_keys": list(shared.get("target_keys", [])),
            "config_hash": shared.get("config_hash"),
            "n_members": len(sub.get("members", [])),
            "routing": (sub.get("routing", {}) or {}).get("rule"),
            "calibration_shipped": bool((sub.get("calibration", {}) or {}).get("shipped")),
        }
    return index


def verify_package(
    manifest: dict,
    root: str | Path,
    *,
    include_code: bool = True,
    ensembles: Iterable[str] | None = None,
) -> list[str]:
    """Verify a package manifest (schema 2): code, every set manifest, and through
    them every member and calibration file. ``ensembles`` limits the sets checked.
    Returns a list of problems."""
    root = Path(root)
    problems: list[str] = []
    index = manifest.get("ensembles", {}) or {}

    for name in available_ensembles(root / ENSEMBLES_DIRNAME):
        if name not in index:
            problems.append(f"{ENSEMBLES_DIRNAME}/{name}/ is not in the package manifest")

    for name in (list(index) if ensembles is None else list(ensembles)):
        entry = index.get(name)
        if entry is None:
            problems.append(f"no model set {name!r} in the package manifest (it has {sorted(index)})")
            continue
        path = root / entry["manifest"]
        if not path.exists():
            problems.append(f"missing set manifest: {entry['manifest']}")
            continue
        if sha256_file(path) != entry.get("manifest_sha256"):
            problems.append(f"{entry['manifest']}: sha256 does not match the package manifest")
            continue
        sub = load_manifest(path)
        keys = list((sub.get("shared", {}) or {}).get("target_keys", []))
        if keys != list(entry.get("target_keys", [])):
            problems.append(f"{entry['manifest']}: target_keys {keys} != package manifest {entry.get('target_keys')}")
        problems.extend(
            f"[{name}] {p}" for p in verify_files(sub, root / entry["dir"], include_code=False)
        )

    if include_code:
        problems.extend(verify_files({"files": manifest.get("files", {})}, root, include_code=True))
    return problems


def package_provenance_text(manifest: dict) -> str:
    """``PROVENANCE.txt`` of the package: the sets, then ``<sha256>  <path>`` lines."""
    release = manifest.get("release", {})
    lines = [
        f"release: {release.get('name', 'VBFNet_Ensemble')} v{release.get('version', '?')} "
        f"({len(manifest.get('ensembles', {}))} model sets)",
        f"built: {release.get('created', '?')}",
        "",
    ]
    for name, entry in (manifest.get("ensembles", {}) or {}).items():
        lines.append(
            f"# set {name}: {entry.get('n_members')} members, route: member = {entry.get('routing')}, "
            f"config_hash {entry.get('config_hash')}, targets {entry.get('target_keys')}"
        )
        lines.append(f"{entry.get('manifest_sha256')}  {entry.get('manifest')}")
    lines.append("")
    for rel, digest in sorted((manifest.get("files", {}) or {}).items()):
        lines.append(f"{digest}  {rel}")
    return "\n".join(lines) + "\n"


def build_package_manifest(root: str | Path, *, version: str, builder: dict | None = None) -> dict:
    """The top-level manifest (schema 2) for the tree at ``root``, from what is on disk."""
    root = Path(root)
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "release": {
            "name": "VBFNet_Ensemble",
            "version": str(version),
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "builder": dict(builder or {}),
        "ensembles": ensemble_index(root),
        "files": {str(rel): sha256_file(root / rel) for rel in iter_code_files(root)},
    }


def write_package_manifest(
    root: str | Path, *, version: str, builder: dict | None = None
) -> tuple[dict, bool]:
    """(Re)write ``<root>/RELEASE_MANIFEST.json`` and ``PROVENANCE.txt``.

    Maintainer helper (``maintainer/build_release.py``, ``install_calibrations.py``).
    Returns ``(manifest, changed)``; nothing is written when only the timestamp
    and builder record would change, so re-running it is a no-op.
    """
    root = Path(root)
    path = root / MANIFEST_NAME
    new = build_package_manifest(root, version=version, builder=builder)
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except ValueError:
            old = {}
        same = (
            old.get("schema_version") == new["schema_version"]
            and old.get("ensembles") == new["ensembles"]
            and old.get("files") == new["files"]
            and (old.get("release", {}) or {}).get("version") == new["release"]["version"]
        )
        if same:
            return old, False
    path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    (root / "PROVENANCE.txt").write_text(package_provenance_text(new))
    return new, True
