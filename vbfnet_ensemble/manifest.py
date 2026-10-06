"""Release manifest: hashing, loading and verification.

``RELEASE_MANIFEST.json`` is the authoritative record of what a release contains.
It pins two independent things:

* the **weights**, by ``sha256`` of each file and by ``state_dict_sha256``, a
  hash over the tensor contents themselves. The file hash changes whenever the
  file is rewritten (stripping ``optimiser`` rewrites it); the state-dict hash
  does not, so it is what proves the weights survived a rebuild.
* the **code**, by ``sha256`` of every shipped ``.py``. The single-model release
  hashes only its checkpoint and calibration files, so it cannot tell you whether
  the decode path that produced a published number is the one you are running.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

MANIFEST_NAME = "RELEASE_MANIFEST.json"
SCHEMA_VERSION = 1

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


def load_manifest(path: str | Path) -> dict:
    with open(path, "r") as fh:
        manifest = json.load(fh)
    version = int(manifest.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: manifest schema_version {version}, this code expects "
            f"{SCHEMA_VERSION}."
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


def iter_code_files(root: str | Path) -> Iterable[Path]:
    """The shipped files whose hashes go into ``manifest["files"]``."""
    root = Path(root)
    for rel in sorted(
        [p.relative_to(root) for p in (root / "vbfnet_ensemble").rglob("*.py")]
        + [p.relative_to(root) for p in (root / "scripts").rglob("*.py")]
        + [p.relative_to(root) for p in (root / "example_bdt_input").glob("*")
           if p.suffix in {".py", ".yaml", ".md"}]
    ):
        if "__pycache__" in rel.parts:
            continue
        yield rel
    for name in ("README.md", "MODEL_CARD.md"):
        if (root / name).exists():
            yield Path(name)
