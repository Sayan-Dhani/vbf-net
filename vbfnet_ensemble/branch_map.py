"""Input branch map: the branch names the model expects -> the names in your tree.

The model reads its inputs under fixed *logical* branch names (``VBFJet_pt``,
``b1_pt``, ``met_phi``, ...). A file that stores the same quantities under other
names is read through a branch map ``{logical: actual}``; the reader then loads
``actual`` and hands it to the model as ``logical``. Only the names change, so an
identity map gives bit-identical inputs.

The template ``branch_map.yaml`` at the repository root lists every logical name
with an identity mapping. Copy it, edit the right-hand side, and pass the file
(or the dict) as ``branch_map=`` to :meth:`VBFNetEnsemble.predict_root`, or as
``--branch_map`` to the scripts. ``scripts/check_branch_map.py`` checks a map
against a ROOT file before you run anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .pyg_vbf_dataset import _EVENT_ID_BRANCHES, _FEATURE_BRANCHES, _TRUTH_BRANCHES

#: CMS event id. Required: each event is routed to member ``event % 5``.
EVENT_ID_BRANCHES: tuple[str, ...] = tuple(_EVENT_ID_BRANCHES)
#: Reco-level model inputs. Required.
FEATURE_BRANCHES: tuple[str, ...] = tuple(_FEATURE_BRANCHES)
#: Generator-level branches, needed only with ``require_truth=True``.
TRUTH_BRANCHES: tuple[str, ...] = tuple(_TRUTH_BRANCHES)
#: Every logical name a branch map may contain.
MODEL_BRANCHES: tuple[str, ...] = EVENT_ID_BRANCHES + FEATURE_BRANCHES + TRUTH_BRANCHES

#: Per-event collections (one value per jet / LHE particle); every other branch
#: holds one value per event.
JAGGED_PREFIXES: tuple[str, ...] = ("VBFJet_", "LHEPart_")


def is_jagged(logical: str) -> bool:
    """True if the logical branch holds one value per jet / LHE particle."""
    return logical.startswith(JAGGED_PREFIXES)


def identity_branch_map(include_truth: bool = True) -> dict[str, str]:
    """``{name: name}`` for every logical branch (the template's content)."""
    names = EVENT_ID_BRANCHES + FEATURE_BRANCHES + (TRUTH_BRANCHES if include_truth else ())
    return {name: name for name in names}


def validate_branch_map(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Check a ``{logical: actual}`` dict and return a plain copy.

    Refuses unknown logical names (typos would otherwise be ignored silently),
    empty or non-string targets, and two logical names mapped to the same
    branch (two different model inputs cannot come from one branch).
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"branch_map must be a mapping {{logical: actual}}, got {type(mapping).__name__}")
    out: dict[str, str] = {}
    unknown = sorted(str(k) for k in mapping if k not in MODEL_BRANCHES)
    if unknown:
        raise ValueError(
            f"branch_map has unknown logical name(s) {unknown}. The left-hand side must be one of "
            f"the names the model expects; see branch_map.yaml for the full list."
        )
    for logical, actual in mapping.items():
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError(f"branch_map[{logical!r}] must be a non-empty branch name, got {actual!r}")
        out[str(logical)] = actual.strip()
    seen: dict[str, str] = {}
    for logical, actual in out.items():
        if actual in seen:
            raise ValueError(
                f"branch_map maps both {seen[actual]!r} and {logical!r} to the branch {actual!r}; "
                "each model input needs its own branch."
            )
        seen[actual] = logical
    return out


def load_branch_map(source: str | Path | Mapping[str, Any] | None) -> dict[str, str]:
    """Load and validate a branch map.

    ``source`` is ``None`` (no renaming), a dict, or a YAML file holding either a
    top-level ``branch_map:`` mapping (the template layout) or a bare mapping.
    """
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return validate_branch_map(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"branch map file not found: {path}")
    import yaml

    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(payload).__name__}")
    mapping = payload.get("branch_map", payload)
    if mapping is None:
        return {}
    return validate_branch_map(mapping)


def check_branch_map(
    root_file: str | Path,
    branch_map: str | Path | Mapping[str, Any] | None = None,
    tree_name: str = "Events",
    require_truth: bool = False,
) -> list[dict[str, Any]]:
    """Check a branch map against one ROOT file, without running the model.

    Returns one row per logical branch the model would read: ``logical``,
    ``actual``, ``required`` (False for truth branches unless ``require_truth``),
    ``present``, ``jagged_expected``, ``jagged_found`` and ``ok``. A row is ok when
    the branch exists and holds one value per event or one per object, as the
    logical branch does. It cannot check that the quantity means the same thing.
    """
    import uproot

    mapping = load_branch_map(branch_map)
    rows: list[dict[str, Any]] = []
    with uproot.open(str(root_file)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not found in {root_file}. Available keys: {list(f.keys())}")
        tree = f[tree_name]
        available = set(tree.keys())
        for logical in MODEL_BRANCHES:
            actual = mapping.get(logical, logical)
            required = logical not in TRUTH_BRANCHES or require_truth
            present = actual in available
            jagged_found = None
            if present:
                typename = str(tree[actual].typename)
                jagged_found = typename.endswith("[]") or "vector" in typename
            ok = present and jagged_found == is_jagged(logical)
            rows.append({
                "logical": logical,
                "actual": actual,
                "required": required,
                "present": present,
                "jagged_expected": is_jagged(logical),
                "jagged_found": jagged_found,
                "ok": ok,
            })
    return rows
