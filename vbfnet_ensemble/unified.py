"""One entry point for every model set in the package: ask for targets, not models.

Each directory of ``ensembles/`` is a complete fold-routed k-fold release
(members, calibration, manifest) that regresses its own targets:

* ``p4``: the quark four-vectors ``q{1,2}_{E,px,py,pz}``, plus the observables
  derived from them (``q{1,2}_{pt,eta,phi,mass}``, ``mjj_p4``, ``deta_p4``,
  ``eta_prod_p4``, ``ptsum_p4``);
* ``hl``: ``mjj``, ``deta``, ``eta_prod``, ``ptsum``, regressed directly, so
  their quantiles are quantiles of the observable itself.

:class:`VBFNet` maps the requested targets to the sets that provide them (read
from the set manifests), loads only those sets, builds the event graphs ONCE
and runs every loaded set on them. The sets share inputs, cuts and routing, so
row i of every set is the same event, predicted by the same fold member
``event % 5`` of each set; the results are merged into one prediction.

**Name rule.** A plain key always means the set that REGRESSES it, so ``mjj`` is
the hl prediction whichever sets are loaded; the p4 set's derived version is
``mjj_p4``. A key never changes meaning with what else was requested.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .manifest import ENSEMBLES_DIRNAME, MANIFEST_NAME, available_ensembles, load_manifest, package_root
from .predictor import VBFNetEnsemble, graphs_from_events
from .pyg_vbf_dataset import DataListDataset
from .routing import resolve_acceptance
from .transforms import P4_DERIVED_KEYS, P4_TARGET_KEYS
from .validate import EnsembleCompatibilityError

__all__ = ["VBFNet", "resolve_targets", "target_catalogue"]

#: Config sections that decide the graphs and the event selection. Sets can share
#: one dataset only if these are identical; ``targets`` only changes the truth y.
SHARED_DATASET_SECTIONS = ("features", "dataset")

#: Per-row arrays every set must agree on, row by row, before they are merged.
_ROW_KEYS = ("fold_id", "event_index", "run", "lumi", "event")
#: Shared scalars and bookkeeping, taken from the first set.
_SHARED_KEYS = ("fold_id", "route", "n_members", "member_ids", "event_index",
                "run", "lumi", "event", "acceptance", "calibration")
#: Arrays concatenated over the sets, and their target axis.
_TARGET_AXIS = {
    "pred_log_full": 1, "pred_phys_full": 1,
    "pred_log_cal_full": 1, "pred_phys_cal_full": 1,
    "truth_log": 1, "pred_log_members": 2,
}


def target_catalogue(ensembles_dir: Union[str, Path, None] = None) -> dict[str, dict[str, list[str]]]:
    """``{set: {"regressed": [...], "derived": [...]}}``, read from the set manifests.

    Nothing is hard-coded per set: a set regressing all eight p4 components gets
    the p4-derived keys, and a new set is picked up from its manifest.
    """
    root = Path(ensembles_dir) if ensembles_dir is not None else package_root() / ENSEMBLES_DIRNAME
    catalogue: dict[str, dict[str, list[str]]] = {}
    for name in available_ensembles(root):
        shared = load_manifest(root / name / MANIFEST_NAME).get("shared", {}) or {}
        regressed = [str(k) for k in shared.get("target_keys", [])]
        derived = list(P4_DERIVED_KEYS) if all(k in regressed for k in P4_TARGET_KEYS) else []
        catalogue[name] = {"regressed": regressed, "derived": derived}
    return catalogue


def resolve_targets(
    targets: Union[str, Iterable[str], None],
    catalogue: dict[str, dict[str, list[str]]],
) -> tuple[list[str], dict[str, str]]:
    """Map requested targets to sets: ``(sets to load, {requested key: set})``.

    ``targets`` is ``None`` or ``"all"`` (every set), a set name (``"p4"``,
    ``"hl"``), a target key, or a list mixing them. A key goes to the set that
    regresses it, else to the set that derives it. Sets come back in catalogue
    order (p4 first), which is the order of the targets in the merged arrays.
    """
    if targets is None:
        targets = ["all"]
    elif isinstance(targets, str):
        targets = [targets]
    targets = [str(t) for t in targets]
    if not targets:
        raise ValueError("targets=[] asks for nothing; pass None or 'all' for every set.")

    wanted: set[str] = set()
    owner_of: dict[str, str] = {}
    for key in targets:
        if key == "all":
            wanted.update(catalogue)
            continue
        if key in catalogue:
            wanted.add(key)
            continue
        owners = [s for s, c in catalogue.items() if key in c["regressed"]]
        if not owners:
            owners = [s for s, c in catalogue.items() if key in c["derived"]]
        if len(owners) > 1:
            raise ValueError(
                f"Target {key!r} is provided by several sets {owners}; ask for the set by name."
            )
        if not owners:
            regressed = {s: c["regressed"] for s, c in catalogue.items()}
            derived = sorted({k for c in catalogue.values() for k in c["derived"]})
            raise ValueError(
                f"Unknown target {key!r}. Regressed: {regressed}. Derived (p4): {derived}. "
                f"Sets: {sorted(catalogue)} or 'all'."
            )
        wanted.add(owners[0])
        owner_of[key] = owners[0]
    return [s for s in catalogue if s in wanted], owner_of


class VBFNet:
    """Fold-routed VBF-Net over every model set that the requested targets need.

    Parameters
    ----------
    targets:
        What to predict: ``None``/``"all"`` (every set: the default), a set name
        (``"p4"``, ``"hl"``), target keys (``["q1_E", "mjj"]``) or a mix. Only
        the sets these need are loaded. Every target of a loaded set is returned.
    device:
        ``None`` picks CUDA when available.
    use_quantile_calibration:
        The ON/OFF switch for quantile calibration of every loaded set (each set
        with its own per-member tables). Settable later, or per call with
        ``calibrate=``.
    return_members, verify, strict, verbose:
        Passed to every set's :class:`~vbfnet_ensemble.predictor.VBFNetEnsemble`.
    ensembles_dir:
        Directory holding the sets. Default: the bundled ``ensembles/``.
    """

    def __init__(
        self,
        targets: Union[str, Sequence[str], None] = None,
        device: str | None = None,
        use_quantile_calibration: bool = False,
        *,
        return_members: bool = False,
        verify: bool | None = None,
        strict: bool = True,
        verbose: bool = True,
        ensembles_dir: Union[str, Path, None] = None,
    ):
        self.ensembles_dir = (
            Path(ensembles_dir) if ensembles_dir is not None else package_root() / ENSEMBLES_DIRNAME
        )
        self.catalogue = target_catalogue(self.ensembles_dir)
        if not self.catalogue:
            raise FileNotFoundError(f"No model sets in {self.ensembles_dir}.")
        names, self.requested = resolve_targets(targets, self.catalogue)
        self.verbose = bool(verbose)

        self.ensembles: dict[str, VBFNetEnsemble] = {}
        for name in names:
            self.ensembles[name] = VBFNetEnsemble(
                device=device,
                use_quantile_calibration=use_quantile_calibration,
                release_dir=self.ensembles_dir / name,
                return_members=return_members,
                verify=verify,
                strict=strict,
                verbose=verbose,
            )
        self._check_compatible()

        ref = self._first
        self.device = ref.device
        self.n_folds = ref.n_folds
        self.route = ref.route
        self.head_names = list(ref.head_names)
        self.num_heads = len(self.head_names)
        self.target_specs = [dict(s) for e in self.ensembles.values() for s in e.target_specs]
        self.target_keys = [k for e in self.ensembles.values() for k in e.target_keys]

        #: Which loaded set provides each key, regressed or derived.
        self.ensemble_of: dict[str, str] = {}
        for name, ens in self.ensembles.items():
            for key in list(ens.target_keys) + self.catalogue[name]["derived"]:
                self.ensemble_of.setdefault(key, name)

        # The dataset config: the members' (identical) features and cuts, with the
        # union of the sets' targets, so with require_truth=True one y serves all.
        self.config = copy.deepcopy(ref.config) if ref.config is not None else None
        if self.config is not None:
            self.config["targets"] = [dict(s) for s in self.target_specs]

        if self.verbose:
            sets = ", ".join(f"{n}: {e.target_keys}" for n, e in self.ensembles.items())
            print(f"[VBFNet] sets loaded: {sets}", flush=True)

    # ── loaded sets ──────────────────────────────────────────────────────────

    @property
    def _first(self) -> VBFNetEnsemble:
        return next(iter(self.ensembles.values()))

    @property
    def n_members(self) -> int:
        return self._first.n_members

    def _check_compatible(self) -> None:
        """Refuse sets whose rows could not be aligned: different graphs or cuts."""
        items = list(self.ensembles.items())
        ref_name, ref = items[0]

        def section(ens: VBFNetEnsemble, key: str) -> str:
            return json.dumps((ens.config or {}).get(key), sort_keys=True, default=str)

        seen = set(ref.target_keys)
        for name, ens in items[1:]:
            problems = [f"config section {s!r}" for s in SHARED_DATASET_SECTIONS
                        if section(ens, s) != section(ref, s)]
            for group in ("node", "edge", "global"):
                key = f"{group}_feature_names"
                if list(ens.member_meta[0].get(key) or []) != list(ref.member_meta[0].get(key) or []):
                    problems.append(key)
            if ens.n_folds != ref.n_folds:
                problems.append(f"n_folds ({ens.n_folds} vs {ref.n_folds})")
            if list(ens.head_names) != list(ref.head_names):
                problems.append(f"head_names ({ens.head_names} vs {ref.head_names})")
            clash = sorted(seen & set(ens.target_keys))
            if clash:
                problems.append(f"targets regressed by both: {clash}")
            seen |= set(ens.target_keys)
            if problems:
                raise EnsembleCompatibilityError(
                    f"Sets {ref_name!r} and {name!r} cannot share one dataset; they differ in "
                    f"{problems}. Load them separately with VBFNetEnsemble(ensemble=...)."
                )

    # ── calibration switch ───────────────────────────────────────────────────

    @property
    def use_quantile_calibration(self) -> bool:
        """ON/OFF switch for quantile calibration of every loaded set."""
        return all(e.use_quantile_calibration for e in self.ensembles.values())

    @use_quantile_calibration.setter
    def use_quantile_calibration(self, value: bool) -> None:
        for ens in self.ensembles.values():
            ens.use_quantile_calibration = value

    @property
    def calibration_available(self) -> bool:
        return all(e.calibration_available for e in self.ensembles.values())

    # ── merging ──────────────────────────────────────────────────────────────

    def _merge(self, results: dict[str, dict], *, merge_truth: bool = True) -> dict:
        """One prediction from the per-set ones (rows are checked to be the same events)."""
        names = list(results)
        ref = results[names[0]]
        for name in names[1:]:
            for key in _ROW_KEYS + ("input_index",):
                if key in ref and not np.array_equal(ref[key], results[name].get(key)):
                    raise RuntimeError(
                        f"Sets {names[0]!r} and {name!r} disagree on {key!r}: their rows "
                        "are not the same events and cannot be merged."
                    )

        out = {key: ref[key] for key in _SHARED_KEYS + ("input_index",) if key in ref}
        out["ensembles"] = names
        out["target_keys"] = list(self.target_keys)
        out["ensemble_of"] = dict(self.ensemble_of)

        for key, axis in _TARGET_AXIS.items():
            if key == "truth_log" and not merge_truth and len(names) > 1:
                continue
            parts = [results[n].get(key) for n in names]
            if all(p is not None for p in parts):
                out[key] = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=axis)

        for key in ("pred_phys", "pred_phys_cal"):
            parts = [results[n].get(key) for n in names]
            if not all(p is not None for p in parts):
                continue
            merged: dict = {}
            for name, part in zip(names, parts):
                clash = sorted(set(part) & set(merged))
                if clash:
                    raise RuntimeError(f"Set {name!r} returns keys already taken: {clash}.")
                merged.update(part)
            out[key] = merged

        # Rate over all (event, target) pairs = target-weighted mean of the set rates.
        rates = {n: float(results[n]["quantile_crossing_rate"]) for n in names}
        weights = {n: len(self.ensembles[n].target_keys) for n in names}
        out["quantile_crossing_rate"] = sum(rates[n] * weights[n] for n in names) / sum(weights.values())
        out["quantile_crossing_rate_by_set"] = rates
        return out

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_root(
        self,
        root_files,
        tree_name: str | None = None,
        max_events: int | None = None,
        batch_size: int = 512,
        num_workers: int = 4,
        decode: bool = True,
        branch_map: dict | str | Path | None = None,
        require_truth: bool | None = False,
        calibrate: bool | None = None,
        acceptance="default",
    ):
        """Run every loaded set on ROOT file(s); the graphs are built once.

        Same options and keys as :meth:`VBFNetEnsemble.predict_root`, merged
        over the sets: ``pred_phys`` holds every target of every loaded set,
        and the ``*_full`` arrays are concatenated along the target axis in
        ``net.target_keys`` order. ``ensemble_of`` says which set gave each key.
        """
        acceptance = resolve_acceptance(acceptance)
        ds = self._first.build_dataset(
            root_files, tree_name=tree_name, max_events=max_events, branch_map=branch_map,
            require_truth=require_truth, acceptance=acceptance, config=self.config,
        )
        results = {
            name: ens.predict_dataset(
                ds, batch_size=batch_size, num_workers=num_workers, decode=decode, calibrate=calibrate,
            )
            for name, ens in self.ensembles.items()
        }
        out = self._merge(results)
        out["acceptance"] = acceptance
        return out

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        decode: bool = True,
        calibrate: bool | None = None,
        *,
        truth_keys: Sequence[str] | None = None,
    ):
        """Run every loaded set on an arbitrary loader (graphs carry ``data.event``).

        With several sets, ``truth_log`` is returned only when ``truth_keys``
        names the columns of the graphs' ``y``.
        """
        results = {
            name: ens.predict_loader(loader, decode=decode, calibrate=calibrate, truth_keys=truth_keys)
            for name, ens in self.ensembles.items()
        }
        return self._merge(results, merge_truth=truth_keys is not None)

    def predict_events(
        self,
        events,
        batch_size: int = 512,
        num_workers: int = 0,
        decode: bool = True,
        calibrate: bool | None = None,
        acceptance="default",
    ):
        """Run every loaded set on in-memory events; see :meth:`VBFNetEnsemble.predict_events`."""
        graphs, kept = graphs_from_events(
            events, self.config, acceptance, route=self.route, verbose=self.verbose,
        )
        loader = DataLoader(
            DataListDataset(graphs), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=self.device.type == "cuda",
        )
        out = self.predict_loader(loader, decode=decode, calibrate=calibrate)
        out["input_index"] = np.asarray(kept, dtype=np.int64)
        return out
