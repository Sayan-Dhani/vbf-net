#!/usr/bin/env python3
"""Make BDT input ROOT files from VBFNet_Ensemble (5 k-fold members, fold-routed).

What this script does
=====================
For one SIGNAL and one BACKGROUND ROOT file it writes, per event:

1. **Classical ("old") VBF-pair selection** — a reco-level baseline, no ML.
   Among the pT-sorted ``VBFJet`` collection, pick the jet pair with the largest
   invariant mass that has ``|deta| > deta_cut``; the event passes if that
   mass is ``> mjj_cut``. Branches (13):

       mjj_old  deta_old  eta_prod_old  pt_sum_old  old_pass
       q1_{E,px,py,pz}_old   q2_{E,px,py,pz}_old

   ``q1`` is the more forward jet (higher eta) — the same ordering as the truth
   quarks the GNN regresses, so ``q1_E_old`` and ``q1_E_gnn_point`` describe the
   same object.

2. **GNN regression** — the generator-level VBF quark four-vectors. Each event
   is predicted by ONE member, ``gnn_fold = event % 5`` (the rule the training
   split used), so on the training signal samples every value is the
   out-of-fold prediction. Branches (60):

       {mjj,deta,eta_prod,pt_sum}_gnn_point                     4   derived
       {q1,q2}_{E,px,py,pz}_gnn_point                           8   regressed
       {q1,q2}_{E,px,py,pz}_gnn_{q16,q50,q84}_raw              24   regressed
       {q1,q2}_{E,px,py,pz}_gnn_{q16,q50,q84}_cal              24   regressed

   * ``_gnn_point``: the routed member's central estimate.
   * ``_raw``: the routed member's uncalibrated quantiles.
   * ``_cal``: calibrated quantiles — the routed member's own shifts, fitted
     on its out-of-fold rows (see ../README.md, "Quantile calibration").
     If calibration is switched off, ``_cal`` is a copy of ``_raw``, exactly as
     in the old script.
   * The four derived observables are built from the ``point`` four-vectors
     only. No quantiles are written for them: mjj computed from the q16
     components is NOT the 16% quantile of mjj.

   **HL releases** (``model.target_set: hl``): the members regress
   ``mjj, deta, eta_prod, ptsum`` DIRECTLY, so every observable gets the full
   set and its quantiles are true quantiles of that observable. Branches (28):

       {mjj,deta,eta_prod,pt_sum}_gnn_point                     4   regressed
       {mjj,deta,eta_prod,pt_sum}_gnn_{q16,q50,q84}_raw        12   regressed
       {mjj,deta,eta_prod,pt_sum}_gnn_{q16,q50,q84}_cal        12   regressed

   The point branch names are the same as the p4 release's derived ones, so a
   BDT reading ``mjj_gnn_point`` works with either release. Total with the old
   selection and bookkeeping: **47 branches**.

3. Six bookkeeping branches: ``event_idx`` (ROOT entry), ``label`` (1 = signal,
   0 = bkg), the CMS event id ``run`` / ``luminosityBlock`` / ``event``, and
   ``gnn_fold`` (the member that made the GNN values, = ``event % 5``).

Total: **79 branches** per tree in the combined file (contract v2 = the 75 of
the previous single-model BDT-input script, plus ``run``, ``luminosityBlock``,
``event``, ``gnn_fold``).
:func:`expected_branches` rebuilds it from the config and the script refuses to
write a tree that differs from it by a single name.

Which events get a row: the deployment acceptance gate (``acceptance`` in the
config): at least ``min_jets`` VBF jets with pT >= ``jet_min_pt`` and
|eta| <= ``jet_max_abs_eta``. It is a gate only -- the graph keeps every jet.
No generator-level cut is applied.

Output files (in ``outputs.outdir``), each with a ``sig`` and a ``bkg`` TTree:

    bdt_inputs_old_selection.root   6 bookkeeping + the 13 old branches
    bdt_inputs_gnn.root             6 bookkeeping + the 60 GNN branches
    bdt_inputs_combined.root        all 79
    bdt_inputs_provenance.json      what produced the files (model, cuts, counts)

What changed compared with the old script
=========================================
Same branches, same YAML layout, same command-line flags. Four things differ:

* **Model**: the 5 k-fold members, each event routed to member ``event % 5``,
  instead of the single fold-1 model.
  ``model.checkpoint`` now means a *directory of members* (``null`` = the
  bundled release), and ``model.calibration_dir`` a per-member calibration
  directory (``null`` = the bundled one).

* **Row alignment (bug fix).** The GNN dataset drops events it cannot turn into
  a graph (fewer than 2 VBF jets), but the old selection keeps one row per raw
  ROOT entry. The old script lined the two up by truncating both to the shorter
  length, so after the first dropped event every row paired the GNN values of
  one event with the old-selection values of another. Measured on
  ``sig_VBF.root``: 2,998 of the first 3,000 rows were misaligned. Rows are now
  matched by the original ROOT entry index, and ``event_idx`` holds that index
  (it used to be 0, 1, 2, ... whatever the event).

* **No generator-truth cuts (bug fix).** The old script let the dataset
  auto-enable its training-time truth cuts (LHE quarks, nLHEPart == 6, Hbb
  validity, quark pT/eta) whenever the file contained truth branches. Those
  select events on information real data does not have, and they removed
  98.6% of the DY background (2,368 events kept from 175,121) and 67% of the
  signal. Inference now uses reco-level inputs only (``require_truth=False``).

* **``max_events`` counts events the GNN keeps**, not raw entries (that is how
  the dataset has always interpreted it). The old selection is read over
  exactly the raw window those kept events come from.

Usage
=====
From the ``VBFNet_Ensemble`` directory (the script adds it to ``sys.path``, so
no ``pip install`` is needed)::

    # needs PyROOT, torch and torch_geometric in the environment
    python3 example_bdt_input/make_bdt_inputs.py \\
        --config example_bdt_input/bdt_inputs_config.yaml

    # quick test on 2000 events per file
    python3 example_bdt_input/make_bdt_inputs.py \\
        --config example_bdt_input/bdt_inputs_config.yaml --max_events 2000 \\
        --outdir bdt_inputs_test

Command-line values override the YAML; the YAML overrides the defaults in
:data:`DEFAULT_BDT_CONFIG`.

Requirements: numpy, uproot, PyYAML, torch, torch_geometric, and **PyROOT**
(``TLorentzVector`` is used for every kinematic calculation, exactly as in the
old script, so the numbers are computed the same way).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# Make `import vbfnet_ensemble` work without installing the package: this file
# lives in <repo>/example_bdt_input/, the package in <repo>/vbfnet_ensemble/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

#: Every setting the script uses, with its default. The YAML file (section
#: ``bdt_inputs``) is merged on top of this, then the command line on top of that.
DEFAULT_BDT_CONFIG: dict[str, Any] = {
    "inputs": {
        "signal": None,                 # signal ROOT file (required)
        "background": None,             # background ROOT file (required)
        "tree_name": "Events",
        "branch_map": None,             # YAML file or {logical: actual}; None = the model's own names
        "max_events": None,             # per file, counted as events the GNN keeps
        "signal_max_events": None,      # overrides max_events for the signal only
        "background_max_events": None,  # overrides max_events for the background only
    },
    "runtime": {
        "batch_size": 512,
        "num_workers": 4,               # DataLoader workers for the GNN
        "device": None,                 # None = cuda if available, else cpu
    },
    # Deployment gate: which events get a GNN prediction (and a row). Reco only.
    "acceptance": {"min_jets": 2, "jet_min_pt": 50.0, "jet_max_abs_eta": 4.7},
    "model": {
        "checkpoint": None,             # directory of member .pt files; None = bundled
        "calibration_dir": None,        # per-member calibration dir; None = bundled
        "use_quantile_calibration": True,   # fills the *_cal branches
        # What the members regress: "p4" = the 8 q{1,2}_{E,px,py,pz} (observables
        # derived from the point p4), "hl" = mjj/deta/eta_prod/ptsum directly.
        "target_set": "p4",
    },
    "old_selection": {
        "deta_cut": 3.0,                # require |eta1 - eta2| > deta_cut
        "mjj_cut": 50.0,                # require best mjj > mjj_cut [GeV]
        "branches": ["nVBFJet", "VBFJet_pt", "VBFJet_eta", "VBFJet_phi", "VBFJet_mass"],
        "outputs": {                    # logical quantity -> branch name
            "mjj": "mjj_old",
            "deta": "deta_old",
            "eta_prod": "eta_prod_old",
            "ptsum": "pt_sum_old",
            "pass": "old_pass",
            "q1_E": "q1_E_old", "q1_px": "q1_px_old", "q1_py": "q1_py_old", "q1_pz": "q1_pz_old",
            "q2_E": "q2_E_old", "q2_px": "q2_px_old", "q2_py": "q2_py_old", "q2_pz": "q2_pz_old",
        },
    },
    # Derived observables written as <branch_prefix>_gnn_point.
    "targets": [
        {"key": "mjj", "vbfnet_key": "mjj", "branch_prefix": "mjj"},
        {"key": "deta", "vbfnet_key": "deta", "branch_prefix": "deta"},
        {"key": "eta_prod", "vbfnet_key": "eta_prod", "branch_prefix": "eta_prod"},
        {"key": "ptsum", "vbfnet_key": "ptsum", "branch_prefix": "pt_sum"},
    ],
    "outputs": {
        "outdir": "bdt_inputs_ensemble",
        "fill_value": -999.0,           # written wherever a value is NaN/inf
        "tree_names": {"signal": "sig", "background": "bkg"},
        "old_filename": "bdt_inputs_old_selection.root",
        "gnn_filename": "bdt_inputs_gnn.root",
        "combined_filename": "bdt_inputs_combined.root",
        "provenance_filename": "bdt_inputs_provenance.json",
        # Branch-name prefixes that go into the old-only / gnn-only files.
        "old_prefixes": [
            "mjj_old", "deta_old", "eta_prod_old", "pt_sum_old", "old_pass",
            "q1_E_old", "q1_px_old", "q1_py_old", "q1_pz_old",
            "q2_E_old", "q2_px_old", "q2_py_old", "q2_pz_old",
        ],
        "gnn_prefixes": [
            "mjj_gnn", "deta_gnn", "eta_prod_gnn", "pt_sum_gnn",
            "q1_E_gnn", "q1_px_gnn", "q1_py_gnn", "q1_pz_gnn",
            "q2_E_gnn", "q2_px_gnn", "q2_py_gnn", "q2_pz_gnn",
        ],
    },
}

#: The eight regressed quark four-vector components, in model output order.
P4_TARGET_KEYS = ("q1_E", "q1_px", "q1_py", "q1_pz", "q2_E", "q2_px", "q2_py", "q2_pz")

#: Observables that can be derived from the two regressed four-vectors.
BDT_DERIVED_KEYS = ("mjj", "deta", "eta_prod", "ptsum")

#: Quantile heads written for the regressed components.
QUANTILE_HEADS = ("q16", "q50", "q84")

#: Branches that are not physics quantities.
BOOKKEEPING_BRANCHES = ("event_idx", "label", "run", "luminosityBlock", "event", "gnn_fold")

#: ``model.target_set`` values: what the ensemble members regress.
TARGET_SETS = ("p4", "hl")


def target_set(cfg: dict[str, Any]) -> str:
    """``model.target_set``: ``p4`` (default) or ``hl``."""
    value = str(get_path(cfg, "model.target_set", "p4") or "p4").lower()
    if value not in TARGET_SETS:
        raise ValueError(f"model.target_set must be one of {TARGET_SETS}, got {value!r}")
    return value


def required_member_keys(cfg: dict[str, Any], targets: list[dict[str, str]]) -> list[str]:
    """Target keys the members must regress for this config."""
    if target_set(cfg) == "hl":
        return [t["vbfnet_key"] for t in targets]
    return list(P4_TARGET_KEYS)


# ── small YAML helpers ──

def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML file; ``{}`` when no path is given."""
    if path is None or str(path).strip() == "":
        return {}
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Top-level YAML content must be a mapping, got {type(payload).__name__}")
    return payload


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``base`` recursively updated with ``updates``.

    Dicts are merged key by key; everything else — including lists — is
    REPLACED. So a YAML ``old_prefixes`` list replaces the default list rather
    than extending it.
    """
    out = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def get_path(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted path such as ``"model.checkpoint"``."""
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(payload: dict[str, Any], path: str, value: Any, *, skip_none: bool = True) -> None:
    """Set a dotted path in place. ``None`` does not override unless ``skip_none=False``."""
    if skip_none and value is None:
        return
    cur = payload
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def none_or_int(value: Any) -> int | None:
    """``None`` / ``"null"`` / ``""`` -> ``None``; anything else -> ``int``."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else int(text)


def as_bool(value: Any) -> bool:
    """Parse the usual YAML/CLI spellings of a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def target_list(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Validate and normalise the ``targets`` list.

    ``p4``: the observables derived from the point p4 (``vbfnet_key`` unused).
    ``hl``: the regressed observables; ``vbfnet_key`` is the member's target key.
    """
    targets = cfg.get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError("bdt_inputs.targets must be a non-empty list")
    out = []
    for item in targets:
        if not isinstance(item, dict) or "key" not in item:
            raise ValueError(f"Invalid target entry: {item!r}")
        key = str(item["key"])
        if key not in BDT_DERIVED_KEYS:
            raise ValueError(
                f"Target '{key}' cannot be derived from the regressed four-vectors; "
                f"allowed: {list(BDT_DERIVED_KEYS)}"
            )
        out.append({
            "key": key,
            "vbfnet_key": str(item.get("vbfnet_key", key)),
            "branch_prefix": str(item.get("branch_prefix", key)),
        })
    return out


# ═════════════════════════════════════════════════════════════════════════════
# The branch contract
# ═════════════════════════════════════════════════════════════════════════════

def expected_old_branches(cfg: dict[str, Any]) -> list[str]:
    """The 13 old-selection branch names, from ``old_selection.outputs``."""
    names = cfg["old_selection"]["outputs"]
    return [names[k] for k in ("mjj", "deta", "eta_prod", "ptsum", "pass", *P4_TARGET_KEYS)]


def expected_gnn_branches(cfg: dict[str, Any]) -> list[str]:
    """GNN branch names.

    ``p4`` (60): 4 derived point + 8 x (point + 3 raw + 3 cal).
    ``hl`` (28): 4 regressed observables x (point + 3 raw + 3 cal).
    """
    if target_set(cfg) == "hl":
        names = []
        for t in target_list(cfg):
            names.append(f"{t['branch_prefix']}_gnn_point")
            for q in QUANTILE_HEADS:
                names.append(f"{t['branch_prefix']}_gnn_{q}_raw")
                names.append(f"{t['branch_prefix']}_gnn_{q}_cal")
        return names
    names = [f"{t['branch_prefix']}_gnn_point" for t in target_list(cfg)]
    for key in P4_TARGET_KEYS:
        names.append(f"{key}_gnn_point")
        for q in QUANTILE_HEADS:
            names.append(f"{key}_gnn_{q}_raw")
            names.append(f"{key}_gnn_{q}_cal")
    return names


def expected_branches(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Exact branch list of each output file, sorted.

    With the default config this reproduces the old script's output branch for
    branch (``tests/test_bdt_example.py`` pins it against a frozen list).
    """
    old = expected_old_branches(cfg)
    gnn = expected_gnn_branches(cfg)
    book = list(BOOKKEEPING_BRANCHES)
    return {
        "combined": sorted(book + old + gnn),
        "old": sorted(book + [b for b in old if _matches(b, cfg["outputs"]["old_prefixes"])]),
        "gnn": sorted(book + [b for b in gnn if _matches(b, cfg["outputs"]["gnn_prefixes"])]),
    }


def _matches(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def check_schema(tree: dict[str, np.ndarray], expected: list[str], what: str) -> None:
    """Refuse to write a tree whose branch set is not exactly ``expected``."""
    got = sorted(tree)
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise RuntimeError(
            f"{what}: branch set differs from the BDT input contract.\n"
            f"  missing: {missing}\n  unexpected: {extra}\n"
            "Refusing to write: a BDT trained on the old branches would silently "
            "read the wrong thing."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Part 1 — classical ("old") VBF jet-pair selection
# ═════════════════════════════════════════════════════════════════════════════

def _lorentz(pt: float, eta: float, phi: float, mass: float):
    """``TLorentzVector`` from (pt, eta, phi, m); negative masses are set to 0."""
    import ROOT

    v = ROOT.TLorentzVector()
    v.SetPtEtaPhiM(pt, eta, phi, max(mass, 0.0))
    return v


def inv_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2) -> float:
    """Invariant mass of two jets, floored at 0 (as in the old script)."""
    return max((_lorentz(pt1, eta1, phi1, m1) + _lorentz(pt2, eta2, phi2, m2)).M(), 0.0)


def select_vbf_pair(
    jets: list[tuple[float, float, float, float]],
    deta_cut: float,
    mjj_cut: float,
) -> tuple | None:
    """Pick the VBF jet pair for ONE event, or ``None`` if the event fails.

    ``jets`` are ``(pt, eta, phi, mass)`` tuples. The rule, unchanged from the
    old script:

    1. sort jets by pT, descending;
    2. over all pairs (j < k) with ``|eta_j - eta_k| > deta_cut``, keep the pair
       with the largest mjj (the first one found wins a tie);
    3. the event passes if that mjj is ``> mjj_cut``.

    Returns ``(mjj, deta, eta_prod, pt_sum, jet_a, jet_b)``.
    """
    if len(jets) < 2:
        return None
    jets = sorted(jets, key=lambda x: x[0], reverse=True)

    best = None
    for j in range(len(jets) - 1):
        for k in range(j + 1, len(jets)):
            pt1, eta1, phi1, m1 = jets[j]
            pt2, eta2, phi2, m2 = jets[k]
            deta = abs(eta1 - eta2)
            if deta <= deta_cut:
                continue
            mjj = inv_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2)
            if best is None or mjj > best[0]:
                best = (mjj, deta, eta1 * eta2, pt1 + pt2, jets[j], jets[k])

    if best is None or best[0] <= mjj_cut:
        return None
    return best


def compute_old_selection_for_file(
    root_file: str,
    tree_name: str = "Events",
    entry_stop: int | None = None,
    deta_cut: float = 3.0,
    mjj_cut: float = 50.0,
    branches: list[str] | None = None,
    output_names: dict[str, str] | None = None,
    branch_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Run the old selection on raw entries ``[0, entry_stop)`` of one file.

    Returns one array per output branch with ONE ROW PER RAW ENTRY (events that
    fail get NaN, and ``old_pass = 0``). The caller then picks the rows of the
    events the GNN kept, see :func:`align_old_to_gnn`.

    ``branches`` are logical names; ``branch_map`` ({logical: actual}) reads them
    from differently named branches, exactly as the GNN reader does.
    """
    import uproot

    branches = list(branches or DEFAULT_BDT_CONFIG["old_selection"]["branches"])
    names = {**DEFAULT_BDT_CONFIG["old_selection"]["outputs"], **(output_names or {})}

    print(f"[old] reading {root_file} (entries [0, {entry_stop if entry_stop is not None else 'EOF'}))")
    with uproot.open(root_file) as f:
        tree = f[tree_name]
        actual = {b: (branch_map or {}).get(b, b) for b in branches}
        missing = sorted(actual[b] for b in branches if actual[b] not in tree.keys())
        if missing:
            raise RuntimeError(f"Missing required branches in {root_file}: {missing} "
                               "(map them with --branch_map / inputs.branch_map)")
        # library="np" gives one small numpy array per event for the jagged
        # branches. Iterating those is much faster than indexing awkward
        # records event by event, and the values are the same.
        aliases = {b: a for b, a in actual.items() if a != b}
        arrays = tree.arrays(branches, aliases=aliases or None, entry_stop=entry_stop, library="np")

    scalar_keys = ("mjj", "deta", "eta_prod", "ptsum")
    out: dict[str, list[float]] = {names[k]: [] for k in (*scalar_keys, *P4_TARGET_KEYS, "pass")}

    n_raw = len(arrays["nVBFJet"])
    for i in range(n_raw):
        n_jets = int(arrays["nVBFJet"][i])
        jets = list(zip(
            arrays["VBFJet_pt"][i][:n_jets].tolist(),
            arrays["VBFJet_eta"][i][:n_jets].tolist(),
            arrays["VBFJet_phi"][i][:n_jets].tolist(),
            arrays["VBFJet_mass"][i][:n_jets].tolist(),
        ))

        best = select_vbf_pair(jets, deta_cut, mjj_cut) if n_jets >= 2 else None
        if best is None:
            for k in (*scalar_keys, *P4_TARGET_KEYS):
                out[names[k]].append(np.nan)
            out[names["pass"]].append(0.0)
            continue

        mjj, deta, eta_prod, pt_sum, jet_a, jet_b = best
        for k, value in zip(scalar_keys, (mjj, deta, eta_prod, pt_sum)):
            out[names[k]].append(value)

        # q1 = the more forward jet (higher eta), the same ordering as the truth quarks.
        forward, backward = (jet_a, jet_b) if jet_a[1] >= jet_b[1] else (jet_b, jet_a)
        for quark, jet in (("q1", forward), ("q2", backward)):
            v = _lorentz(*jet)
            for comp, value in (("E", v.E()), ("px", v.Px()), ("py", v.Py()), ("pz", v.Pz())):
                out[names[f"{quark}_{comp}"]].append(value)
        out[names["pass"]].append(1.0)

    result = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    n_pass = int(np.sum(result[names["pass"]] > 0.5))
    print(f"[old] raw events={n_raw:,}, pass={n_pass:,}, eff={100.0 * n_pass / max(n_raw, 1):.2f}%")
    return result


def align_old_to_gnn(old: dict[str, np.ndarray], raw_event_index: np.ndarray) -> dict[str, np.ndarray]:
    """Keep only the old-selection rows of the events the GNN kept, in GNN order.

    ``old`` has one row per raw entry; ``raw_event_index[i]`` is the raw entry
    of GNN row ``i``. Indexing by it makes the two row-aligned.
    """
    idx = np.asarray(raw_event_index, dtype=np.int64)
    aligned = {}
    for key, values in old.items():
        values = np.asarray(values)
        if idx.size and int(idx.max()) >= len(values):
            raise IndexError(
                f"old-selection branch '{key}' has {len(values)} rows but the GNN "
                f"references raw entry {int(idx.max())}."
            )
        aligned[key] = values[idx]
    return aligned


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 — GNN regression with the ensemble
# ═════════════════════════════════════════════════════════════════════════════

def _require(pred: dict, key: str, head: str) -> np.ndarray:
    """``pred[key][head]`` as float64, with a readable error if it is missing."""
    if key not in pred:
        raise KeyError(f"Target '{key}' not in the prediction; available: {sorted(pred)}")
    if head not in pred[key]:
        raise KeyError(f"Head '{head}' not found for '{key}'; available: {sorted(pred[key])}")
    return np.asarray(pred[key][head], dtype=np.float64)


def derive_observables_from_p4(pred: dict, head: str = "point") -> dict[str, np.ndarray]:
    """mjj, deta, eta_prod, ptsum from the regressed q1/q2 four-vectors.

    Each event's (E, px, py, pz) for q1 and q2 are loaded into
    ``TLorentzVector`` and the observables read back — the same calculation as
    the old script:

        mjj      = (v1 + v2).M(), floored at 0
        deta     = |eta1 - eta2|
        eta_prod = eta1 * eta2
        ptsum    = pT1 + pT2

    Use the ``point`` head only: applying this to the q16 components does not
    give the 16% quantile of mjj.
    """
    import ROOT

    comp = {k: _require(pred, k, head) for k in P4_TARGET_KEYS}
    n = len(comp["q1_E"])
    mjj, deta, eta_prod, ptsum = (np.empty(n) for _ in range(4))

    v1, v2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
    for i in range(n):
        v1.SetPxPyPzE(float(comp["q1_px"][i]), float(comp["q1_py"][i]),
                      float(comp["q1_pz"][i]), float(comp["q1_E"][i]))
        v2.SetPxPyPzE(float(comp["q2_px"][i]), float(comp["q2_py"][i]),
                      float(comp["q2_pz"][i]), float(comp["q2_E"][i]))
        mjj[i] = max((v1 + v2).M(), 0.0)
        eta1, eta2 = v1.Eta(), v2.Eta()
        deta[i] = abs(eta1 - eta2)
        eta_prod[i] = eta1 * eta2
        ptsum[i] = v1.Pt() + v2.Pt()

    return {"mjj": mjj, "deta": deta, "eta_prod": eta_prod, "ptsum": ptsum}


def build_gnn_branches(
    pred_raw: dict,
    pred_cal: dict | None,
    targets: list[dict[str, str]],
    target_set: str = "p4",
) -> dict[str, np.ndarray]:
    """Turn the ensemble's prediction dicts into the GNN branches (60 p4 / 28 hl).

    ``pred_raw`` is ``out["pred_phys"]``; ``pred_cal`` is
    ``out["pred_phys_cal"]`` or ``None`` when calibration is off, in which case
    the ``_cal`` branches are copies of the ``_raw`` ones.
    """
    gnn: dict[str, np.ndarray] = {}

    if target_set == "hl":
        # Regressed observables: point + raw quantiles + calibrated quantiles.
        for spec in targets:
            key, prefix = spec["vbfnet_key"], spec["branch_prefix"]
            gnn[f"{prefix}_gnn_point"] = _require(pred_raw, key, "point")
            for q in QUANTILE_HEADS:
                raw = _require(pred_raw, key, q)
                gnn[f"{prefix}_gnn_{q}_raw"] = raw
                gnn[f"{prefix}_gnn_{q}_cal"] = (_require(pred_cal, key, q)
                                                if pred_cal is not None else raw.copy())
        return gnn

    # Derived observables: point head only.
    derived = derive_observables_from_p4(pred_raw, "point")
    for spec in targets:
        gnn[f"{spec['branch_prefix']}_gnn_point"] = derived[spec["key"]]

    # Regressed components: point + raw quantiles + calibrated quantiles.
    for key in P4_TARGET_KEYS:
        gnn[f"{key}_gnn_point"] = _require(pred_raw, key, "point")
        for q in QUANTILE_HEADS:
            raw = _require(pred_raw, key, q)
            gnn[f"{key}_gnn_{q}_raw"] = raw
            gnn[f"{key}_gnn_{q}_cal"] = _require(pred_cal, key, q) if pred_cal is not None else raw.copy()

    return gnn


def predict_gnn_for_file(
    net,
    root_file: str,
    tree_name: str,
    max_events: int | None,
    batch_size: int,
    num_workers: int,
    targets: list[dict[str, str]],
    acceptance: dict | None = None,
    branch_map: dict[str, str] | None = None,
    target_set: str = "p4",
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Run the fold-routed predictor on one file.

    Returns ``(gnn_branches, raw_event_index, ids)``: ``raw_event_index[i]`` is
    the ROOT entry of GNN row ``i`` (events failing the acceptance gate are
    dropped); ``ids`` holds ``run``, ``luminosityBlock``, ``event`` and
    ``gnn_fold`` per row.
    """
    print(f"[gnn] predicting {root_file}")
    out = net.predict_root(
        root_files=root_file,
        tree_name=tree_name,
        max_events=max_events,
        batch_size=batch_size,
        num_workers=num_workers,
        decode=True,
        branch_map=branch_map,
        # Reco-level inputs only. The truth cuts exist to build TRAINING targets;
        # applied here they would select signal on generator information and
        # throw away almost the entire DY background.
        require_truth=False,
        acceptance=acceptance,
    )

    if "event_index" not in out:
        raise RuntimeError(
            "The predictor did not return event_index; rows cannot be matched "
            "to the old selection safely."
        )
    need = [t["vbfnet_key"] for t in targets] if target_set == "hl" else list(P4_TARGET_KEYS)
    if not all(k in out["pred_phys"] for k in need):
        raise RuntimeError(f"target_set={target_set} needs targets {need}; got {sorted(out['pred_phys'])}")

    gnn = build_gnn_branches(out["pred_phys"], out.get("pred_phys_cal"), targets, target_set)
    raw_idx = np.asarray(out["event_index"], dtype=np.int64)
    ids = {
        "run": np.asarray(out["run"], dtype=np.int64),
        "luminosityBlock": np.asarray(out["lumi"], dtype=np.int64),
        "event": np.asarray(out["event"], dtype=np.int64),
        "gnn_fold": np.asarray(out["fold_id"], dtype=np.int32),
    }
    if not np.array_equal(ids["gnn_fold"], ids["event"] % net.n_folds):
        raise RuntimeError("gnn_fold != event % n_folds: the routing is broken.")
    counts = ", ".join(f"{k}:{int(np.sum(ids['gnn_fold'] == k))}" for k in range(net.n_folds))
    print(f"[gnn] kept {len(raw_idx):,} events (raw entries [0, "
          f"{int(raw_idx.max()) + 1 if raw_idx.size else 0})); per member {counts}")
    return gnn, raw_idx, ids


# ═════════════════════════════════════════════════════════════════════════════
# Part 3 — assemble and write
# ═════════════════════════════════════════════════════════════════════════════

def as_root_float(arr: np.ndarray, fill_value: float = -999.0) -> np.ndarray:
    """float32 copy with NaN/inf replaced by ``fill_value``."""
    arr = np.asarray(arr, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.float32(fill_value)
    return arr


def make_tree(
    old: dict[str, np.ndarray],
    gnn: dict[str, np.ndarray],
    raw_event_index: np.ndarray,
    label: int,
    pass_name: str,
    fill_value: float = -999.0,
    ids: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """One TTree's worth of branches. ``old`` must already be aligned to ``gnn``."""
    n = len(raw_event_index)
    if ids is None:
        raise ValueError("make_tree needs the run/luminosityBlock/event/gnn_fold arrays (ids).")
    for name, arrays in (("old", old), ("gnn", gnn), ("ids", ids)):
        bad = {k: len(v) for k, v in arrays.items() if len(v) != n}
        if bad:
            raise ValueError(f"{name} branches are not row-aligned with the GNN ({n} rows): {bad}")

    tree: dict[str, np.ndarray] = {
        "event_idx": np.asarray(raw_event_index, dtype=np.int64),   # ROOT entry number
        "label": np.full(n, label, dtype=np.int32),
        "run": np.asarray(ids["run"], dtype=np.int64),
        "luminosityBlock": np.asarray(ids["luminosityBlock"], dtype=np.int64),
        "event": np.asarray(ids["event"], dtype=np.int64),
        "gnn_fold": np.asarray(ids["gnn_fold"], dtype=np.int32),   # member = event % 5
    }
    for key, value in old.items():
        if key == pass_name:
            tree[key] = (np.asarray(value) > 0.5).astype(np.int32)
        else:
            tree[key] = as_root_float(value, fill_value)
    for key, value in gnn.items():
        tree[key] = as_root_float(value, fill_value)
    return tree


def select_branches(tree: dict[str, np.ndarray], prefixes: list[str]) -> dict[str, np.ndarray]:
    """Bookkeeping branches plus every branch starting with one of ``prefixes``."""
    return {k: v for k, v in tree.items() if k in BOOKKEEPING_BRANCHES or _matches(k, prefixes)}


def write_two_tree_root(path: Path, sig_tree: dict, bkg_tree: dict, tree_names: dict) -> None:
    """Write a ROOT file with a signal and a background TTree."""
    import uproot

    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as fout:
        fout[str(tree_names.get("signal", "sig"))] = sig_tree
        fout[str(tree_names.get("background", "bkg"))] = bkg_tree
    print(f"[saved] {path}")


def process_file(net, path: str, label: int, cap: int | None, cfg: dict, targets: list,
                 branch_map: dict[str, str] | None = None) -> dict:
    """GNN first (to learn which events survive), then the old selection on
    exactly that raw window, then align the two by ROOT entry index."""
    inputs, runtime, old_cfg = cfg["inputs"], cfg["runtime"], cfg["old_selection"]
    names = old_cfg["outputs"]

    gnn, raw_idx, ids = predict_gnn_for_file(
        net, path, inputs["tree_name"], cap,
        int(runtime["batch_size"]), int(runtime["num_workers"]), targets,
        acceptance=cfg.get("acceptance"), target_set=target_set(cfg),
        branch_map=branch_map,
    )
    old = compute_old_selection_for_file(
        root_file=path,
        tree_name=inputs["tree_name"],
        entry_stop=int(raw_idx.max()) + 1 if raw_idx.size else 0,
        deta_cut=float(old_cfg["deta_cut"]),
        mjj_cut=float(old_cfg["mjj_cut"]),
        branches=list(old_cfg["branches"]),
        output_names=names,
        branch_map=branch_map,
    )
    old = align_old_to_gnn(old, raw_idx)
    return make_tree(old, gnn, raw_idx, label, names["pass"], float(cfg["outputs"]["fill_value"]), ids)


# ═════════════════════════════════════════════════════════════════════════════
# Command line
# ═════════════════════════════════════════════════════════════════════════════

def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """defaults <- YAML ``bdt_inputs`` section <- command line."""
    payload = load_yaml_config(args.config)
    section = payload.get("bdt_inputs", payload.get("bdt", {})) or {}
    cfg = deep_update(DEFAULT_BDT_CONFIG, section)

    overrides = {
        "inputs.signal": args.signal,
        "inputs.background": args.background,
        "inputs.tree_name": args.tree_name,
        "inputs.branch_map": args.branch_map,
        "inputs.max_events": args.max_events,
        "inputs.signal_max_events": args.signal_max_events,
        "inputs.background_max_events": args.background_max_events,
        "outputs.outdir": args.outdir,
        "outputs.fill_value": args.fill_value,
        "runtime.batch_size": args.batch_size,
        "runtime.num_workers": args.num_workers,
        "runtime.device": args.device,
        "model.checkpoint": args.checkpoint,
        "model.calibration_dir": args.calibration_dir,
        "old_selection.deta_cut": args.deta_cut,
        "old_selection.mjj_cut": args.mjj_cut,
    }
    for path, value in overrides.items():
        set_path(cfg, path, value)
    if args.no_calibration:
        set_path(cfg, "model.use_quantile_calibration", False, skip_none=False)
    return cfg


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BDT input maker: classical VBF selection + VBFNet_Ensemble (fold-routed) regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # All defaults are None so the YAML can supply them; the effective defaults
    # live in DEFAULT_BDT_CONFIG. The flags are the old script's, unchanged.
    p.add_argument("--config", default=None, help="YAML config (section 'bdt_inputs'). CLI overrides it.")
    p.add_argument("--signal", default=None, help="Signal ROOT file")
    p.add_argument("--background", default=None, help="Background ROOT file")
    p.add_argument("--tree_name", default=None)
    p.add_argument("--branch_map", default=None,
                   help="YAML branch map for files whose branches have other names (see ../branch_map.yaml)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--max_events", type=int, default=None,
                   help="Events per file, counted after the acceptance gate drops events")
    p.add_argument("--signal_max_events", type=int, default=None)
    p.add_argument("--background_max_events", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Directory of ensemble member .pt files (default: the bundled release)")
    p.add_argument("--calibration_dir", default=None,
                   help="Per-member calibration directory (default: the bundled one)")
    p.add_argument("--no_calibration", action="store_true",
                   help="Switch quantile calibration off (the *_cal branches then copy *_raw)")
    p.add_argument("--deta_cut", type=float, default=None)
    p.add_argument("--mjj_cut", type=float, default=None)
    p.add_argument("--fill_value", type=float, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    cfg = build_config(args)

    signal, background = get_path(cfg, "inputs.signal"), get_path(cfg, "inputs.background")
    if not signal or not background:
        raise SystemExit("Both signal and background are required (YAML inputs.* or --signal/--background).")

    targets = target_list(cfg)
    expected = expected_branches(cfg)       # validates the config before any work
    inputs, runtime, model_cfg, out_cfg = cfg["inputs"], cfg["runtime"], cfg["model"], cfg["outputs"]
    old_cfg = cfg["old_selection"]

    max_events = none_or_int(inputs.get("max_events"))
    sig_cap = none_or_int(inputs["signal_max_events"]) if inputs.get("signal_max_events") is not None else max_events
    bkg_cap = none_or_int(inputs["background_max_events"]) if inputs.get("background_max_events") is not None else max_events
    use_cal = as_bool(model_cfg.get("use_quantile_calibration", True))
    outdir = Path(out_cfg["outdir"])

    from vbfnet_ensemble.branch_map import load_branch_map

    branch_map = load_branch_map(inputs.get("branch_map"))   # validated before any work
    renamed = {k: v for k, v in branch_map.items() if k != v}

    print("=" * 80)
    print("BDT input production — VBFNet_Ensemble (fold-routed)")
    print(f"signal      : {signal}   (max_events={sig_cap})")
    print(f"background  : {background}   (max_events={bkg_cap})")
    print(f"outdir      : {outdir}")
    print(f"old cuts    : |deta| > {old_cfg['deta_cut']}, mjj > {old_cfg['mjj_cut']} GeV")
    print(f"target_set  : {target_set(cfg)}")
    print(f"calibration : {'ON' if use_cal else 'OFF (the *_cal branches will equal *_raw)'}")
    print(f"acceptance  : {cfg.get('acceptance')}")
    print(f"branch map  : {len(renamed)} renamed branch(es)" + (f" {renamed}" if renamed else ""))
    print(f"branches    : {len(expected['combined'])} per tree in the combined file")
    print("=" * 80)

    from vbfnet_ensemble import VBFNetEnsemble

    net = VBFNetEnsemble(
        checkpoint=model_cfg.get("checkpoint"),
        device=runtime.get("device"),
        use_quantile_calibration=use_cal,
        calibration_dir=model_cfg.get("calibration_dir"),
    )
    # Fail before reading any ROOT file if the release does not regress what
    # this config expects (e.g. an HL config pointed at the p4 release).
    need = required_member_keys(cfg, targets)
    missing = [k for k in need if k not in list(net.target_keys)]
    if missing:
        raise SystemExit(
            f"model.target_set={target_set(cfg)} needs member targets {need}, but the "
            f"release regresses {list(net.target_keys)} (missing {missing})."
        )

    sig_tree = process_file(net, signal, 1, sig_cap, cfg, targets, branch_map=branch_map)
    bkg_tree = process_file(net, background, 0, bkg_cap, cfg, targets, branch_map=branch_map)

    trees = {}
    for kind, prefixes_key in (("old", "old_prefixes"), ("gnn", "gnn_prefixes"), ("combined", None)):
        s = sig_tree if prefixes_key is None else select_branches(sig_tree, list(out_cfg[prefixes_key]))
        b = bkg_tree if prefixes_key is None else select_branches(bkg_tree, list(out_cfg[prefixes_key]))
        check_schema(s, expected[kind], f"{kind} / signal")
        check_schema(b, expected[kind], f"{kind} / background")
        trees[kind] = (s, b)

    tree_names = out_cfg.get("tree_names", {"signal": "sig", "background": "bkg"})
    files = {
        "old": outdir / out_cfg["old_filename"],
        "gnn": outdir / out_cfg["gnn_filename"],
        "combined": outdir / out_cfg["combined_filename"],
    }
    for kind, path in files.items():
        write_two_tree_root(path, *trees[kind], tree_names)

    # A small record of what produced these files.
    from vbfnet_ensemble import __version__

    provenance = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "vbfnet_ensemble_version": __version__,
        "release_manifest": (net.manifest or {}).get("release"),
        "members": [
            {"fold_id": m.get("fold_id"), "epoch": m.get("epoch"), "file": m.get("file")}
            for m in net.member_meta
        ],
        "route": f"member = {net.route}",
        "target_set": target_set(cfg),
        "member_target_keys": list(net.target_keys),
        "member_split_modes": ((net.manifest or {}).get("routing", {}) or {}).get("member_split_modes"),
        "acceptance": cfg.get("acceptance"),
        "branch_map_renamed": renamed,
        "calibration": net._calibration_label(),
        "config": cfg,
        "events": {
            "signal": int(len(sig_tree["event_idx"])),
            "background": int(len(bkg_tree["event_idx"])),
        },
        "branches": expected,
        "require_truth": False,
        "event_idx": "original ROOT entry number of the event in its input file",
        "gnn_fold": "the member that produced the GNN branches of the row (= event % n_folds)",
    }
    prov_path = outdir / out_cfg.get("provenance_filename", "bdt_inputs_provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"[saved] {prov_path}")

    print("=" * 80)
    print(f"signal events     : {len(sig_tree['event_idx']):,}")
    print(f"background events : {len(bkg_tree['event_idx']):,}")
    for path in files.values():
        print(f"  {path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
