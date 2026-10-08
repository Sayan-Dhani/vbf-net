"""Fold-routed inference over ONE set of k-fold VBF-Net checkpoints.

``VBFNetEnsemble`` runs one model set from ``ensembles/<name>/`` (``ensemble=``,
default ``p4``) or any release directory (``release_dir=``). It is the engine
behind :class:`vbfnet_ensemble.unified.VBFNet`, which picks the sets from the
requested targets and merges their predictions; use it directly for a single
set or for checkpoints outside the package. It keeps the constructor keyword
names and the ``predict_root`` / ``predict_events`` / ``predict_loader``
signatures of ``trained_model_VBFNet.VBFNet``.

Every event is predicted by EXACTLY ONE member (see :mod:`.routing`)::

    event -> acceptance gate (reco only) -> k = event % n_folds -> member k

On the training samples member k is the one model that never trained on the
event, so the deployed prediction is the out-of-fold prediction that was
measured. There is no median over members any more.

Differences from the single-model predictor, all deliberate:

* checkpoints load with ``weights_only=True`` (verified to work for every k-fold
  checkpoint), so a downloaded release cannot execute code on unpickling;
* ``correctionlib`` is imported lazily, so this package imports without it;
* ``self.model`` and ``self.ckpt`` are **not** defined. There are K models,
  and anything reaching for "the" model should get a loud ``AttributeError``
  rather than member 0 silently standing in for all of them.
"""

from __future__ import annotations

import gc
import glob as _glob
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .branch_map import load_branch_map
from .calibration import (
    CalibrationError,
    apply_quantile_calibration,
    calibrate_member_log,
    load_shift_tables,
)
from .manifest import (
    MANIFEST_NAME,
    check_not_lfs_pointer,
    ensemble_dir,
    load_manifest,
    sha256_file,
    state_dict_sha256,
)
from .pyg_vbf_dataset import (
    NUM_TARGETS,
    DataListDataset,
    VBFJetRootDataset,
    build_data_from_arrays,
    subset_dataset,
)
from .pyg_vbf_gnn import PyGVBFGNN
from .routing import (
    ROUTE_RULE,
    quantile_crossing_rate,
    resolve_acceptance,
    route_folds,
    sort_quantile_heads,
)
from .transforms import (
    DEFAULT_QUANTILES,
    LEGACY_TARGET_SPECS,
    LEGACY_TARGETS,
    decode_predictions,
    decode_predictions_array,
    predictions_array_to_dict,
    _head_names_from_metadata,
    _target_keys_from_specs,
    _target_specs_from_ckpt,
)
from .validate import (
    EnsembleCompatibilityError,
    assert_members_compatible,
    order_members,
)

__all__ = [
    "VBFNetEnsemble",
    "build_model_from_ckpt",
    "graphs_from_events",
    "decode_predictions",
    "decode_predictions_array",
    "predictions_array_to_dict",
    "apply_quantile_calibration",
]


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> PyGVBFGNN:
    """Instantiate and load a model from a checkpoint dict.

    This is ``VBFNet._load_model``'s body with the file I/O and the ``self``
    dependency removed, because an ensemble needs it N times without an instance.
    The legacy no-``config`` branch is kept so this stays a faithful mirror, even
    though every k-fold checkpoint takes the ``cfg`` path.
    """
    cfg = ckpt.get("config", None)
    ca = ckpt.get("args", {}) or {}

    if cfg:
        model = PyGVBFGNN.from_config(
            num_node_features=int(ckpt["num_node_features"]),
            num_edge_features=int(ckpt["num_edge_features"]),
            num_global_features=int(ckpt["num_global_features"]),
            cfg=cfg,
        )
    else:
        head_hidden = ca.get("mlp_hidden", None) or ca.get("head_hidden", None) or [256, 128]
        output_mode = ckpt.get("output_mode", ca.get("output_mode", "both"))
        quantiles = ckpt.get("quantiles", ca.get("quantiles", DEFAULT_QUANTILES))

        model = PyGVBFGNN(
            num_node_features=int(ckpt["num_node_features"]),
            num_edge_features=int(ckpt["num_edge_features"]),
            num_global_features=int(ckpt["num_global_features"]),
            node_dim=int(ca.get("node_dim", 128)),
            edge_dim=int(ca.get("edge_dim", 128)),
            global_dim=int(ca.get("global_dim", 128)),
            n_layers=int(ca.get("n_layers", 6)),
            node_encoder_hidden=tuple(ca.get("node_encoder_hidden", [128])),
            edge_encoder_hidden=tuple(ca.get("edge_encoder_hidden", [128])),
            global_encoder_hidden=tuple(ca.get("global_encoder_hidden", [128])),
            head_hidden=tuple(int(x) for x in head_hidden),
            mp_hidden=ca.get("mp_hidden", None),
            aggregation=tuple(ca.get("aggregation", ["sum", "mean", "max"])),
            dropout=float(ca.get("dropout", 0.0)),
            pool=str(ca.get("pool", "mean+max")),
            activation=str(ca.get("activation", "GELU")),
            norm=ca.get("norm", "LayerNorm"),
            num_targets=int(ckpt.get("num_targets", NUM_TARGETS)),
            output_mode=str(output_mode),
            quantiles=tuple(float(q) for q in quantiles),
            use_edge_pair_summary=bool(ca.get("use_edge_pair_summary", True)),
        )

    state = ckpt.get("model", None) or ckpt.get("model_state_dict", None)
    if state is None:
        raise KeyError("Checkpoint contains neither 'model' nor 'model_state_dict'.")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _resolve_member_paths(checkpoint, release_root: Path) -> list[Path]:
    """Resolve the ``checkpoint`` argument to a concrete list of files."""
    if checkpoint is None:
        checkpoint = release_root / "models"

    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        if path.is_dir():
            paths = sorted(path.glob("*.pt"))
            if not paths:
                raise FileNotFoundError(f"No .pt checkpoints found in {path}")
            return paths
        if any(ch in str(path) for ch in "*?["):
            paths = sorted(Path(p) for p in _glob.glob(str(path)))
            if not paths:
                raise FileNotFoundError(f"Glob matched no checkpoints: {checkpoint}")
            return paths
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return [path]

    paths = [Path(p) for p in checkpoint]
    if not paths:
        raise ValueError("checkpoint= was an empty sequence.")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return paths


class VBFNetEnsemble:
    """One set of K k-fold VBF-Net regressors, each event routed to one of them.

    Parameters
    ----------
    checkpoint:
        ``None`` (the set's ``models/`` directory), a directory, a glob or an
        explicit sequence of files. The members must be exactly folds
        ``0 .. n_folds-1`` of one k-fold run: every event needs its member.
    device:
        ``None`` picks CUDA when available.
    use_quantile_calibration:
        The ON/OFF switch for quantile calibration. Off by default, matching the
        single-model ``VBFNet``. When on, the rows routed to member k are
        calibrated with member k's OWN shifts (fitted on member k's out-of-fold
        rows, i.e. exactly the population routed to it), and the calibrated
        results are returned under the ``*_cal`` keys alongside the raw ones.
        Can be flipped later (``net.use_quantile_calibration = True``) or
        overridden per call (``predict_root(..., calibrate=True)``).
    calibration_dir:
        Directory holding the calibration. Default: the set's
        ``calibrations/``. Two layouts are understood:
        ``<dir>/fold{k}/*.json`` (one calibration per member — what ships) and
        ``<dir>/*.json`` (one calibration shared by every member).
    ensemble:
        Name of the bundled set to load (a directory of ``ensembles/``):
        ``"p4"`` (the default) or ``"hl"``.
    release_dir:
        Instead of ``ensemble``: any directory laid out like a set
        (``models/``, ``calibrations/``, ``RELEASE_MANIFEST.json``).
    return_members:
        Diagnostic: ALSO run every member on every event and return
        ``pred_log_members`` ``(M, N, n_targets, n_heads)``. Costs M forward
        passes instead of one; the routed prediction is unaffected.
    verify:
        Check each file's sha256 against the set's ``RELEASE_MANIFEST.json``
        *before* unpickling it. ``None`` means "verify if a manifest is present".
    """

    def __init__(
        self,
        checkpoint: Union[str, Path, Sequence[Union[str, Path]], None] = None,
        device: str | None = None,
        use_quantile_calibration: bool = False,
        calibration_dir: Union[str, Path, None] = None,
        *,
        ensemble: str | None = None,
        release_dir: Union[str, Path, None] = None,
        return_members: bool = False,
        verify: bool | None = None,
        strict: bool = True,
        manifest: Union[str, Path, None] = None,
        verbose: bool = True,
    ):
        if release_dir is not None and ensemble is not None:
            raise ValueError("Pass ensemble= or release_dir=, not both.")
        # The set directory: models/, calibrations/ and the manifest live here.
        self.release_root = (
            Path(release_dir).resolve() if release_dir is not None else ensemble_dir(ensemble)
        )
        self.ensemble = str(ensemble) if ensemble is not None else self.release_root.name

        self.return_members = bool(return_members)
        self.verbose = bool(verbose)

        if calibration_dir is None:
            calibration_dir = self.release_root / "calibrations"
        self.calibration_dir = Path(calibration_dir)
        self._calibration_tables: list[dict] | None = None
        self._use_quantile_calibration = False

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── manifest ──────────────────────────────────────────────────────────
        manifest_path = Path(manifest) if manifest is not None else self.release_root / MANIFEST_NAME
        self.manifest = load_manifest(manifest_path) if manifest_path.exists() else None
        if verify is None:
            verify = self.manifest is not None
        self.verified = bool(verify) and self.manifest is not None

        member_paths = _resolve_member_paths(checkpoint, self.release_root)

        expected_sha = {}
        if self.verified:
            expected_sha = {
                Path(m["file"]).name: m["sha256"] for m in self.manifest.get("members", [])
            }

        # ── load members, one at a time ───────────────────────────────────────
        models: list[PyGVBFGNN] = []
        metas: list[dict] = []
        for path in member_paths:
            check_not_lfs_pointer(path)

            if self.verified and path.name in expected_sha:
                digest = sha256_file(path)          # BEFORE torch.load
                if digest != expected_sha[path.name]:
                    raise EnsembleCompatibilityError(
                        f"{path} failed manifest verification: sha256 {digest[:16]}… "
                        f"!= {expected_sha[path.name][:16]}…. Refusing to load it."
                    )

            ckpt = torch.load(path, map_location="cpu", weights_only=True)

            meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimiser")}
            meta["file"] = str(path)
            meta["state_dict_sha256"] = state_dict_sha256(ckpt["model"])
            if meta.get("fold_id", None) is None:
                meta["fold_id"] = (ckpt.get("args", {}) or {}).get("fold_id", None)

            models.append(build_model_from_ckpt(ckpt, self.device))
            metas.append(meta)

            del ckpt
            gc.collect()

        assert_members_compatible(metas, strict=strict)

        # Order by fold_id, so self.models[k] IS the member for event % K == k.
        order = order_members(metas)
        self.models = [models[i] for i in order]
        self.member_meta = [metas[i] for i in order]
        self.member_ids = [m.get("fold_id") for m in self.member_meta]
        self.n_members = len(self.models)

        # Routing needs one member per fold, exactly folds 0..K-1.
        n_folds = {
            (m.get("n_folds") or (m.get("args", {}) or {}).get("n_folds"))
            for m in self.member_meta
        } - {None}
        self.n_folds = int(n_folds.pop()) if len(n_folds) == 1 else self.n_members
        if self.member_ids != list(range(self.n_folds)):
            raise EnsembleCompatibilityError(
                f"Fold routing needs exactly one member per fold 0..{self.n_folds - 1}; "
                f"got fold_ids {self.member_ids}. Events with event % {self.n_folds} "
                f"equal to a missing fold would have no member."
            )

        for model in self.models:
            model.eval()
            assert not model.training

        # ── shared metadata (identical across members by construction) ────────
        ref = self.member_meta[0]
        self.config = ref.get("config", None)
        self.target_specs = _target_specs_from_ckpt(ref)
        self.target_keys = _target_keys_from_specs(self.target_specs)
        self.output_mode = str(
            ref.get("output_mode", (self.config or {}).get("output", {}).get("mode", "both"))
        )
        self.quantiles = list(
            ref.get(
                "quantiles",
                (self.config or {}).get("output", {}).get("quantiles", DEFAULT_QUANTILES),
            )
        )
        self.head_names = _head_names_from_metadata(
            head_names=ref.get("head_names", None),
            output_mode=self.output_mode,
            quantiles=self.quantiles,
        )
        self.num_heads = len(self.head_names)
        self.route = ROUTE_RULE.replace("n_folds", str(self.n_folds))

        # Goes through the property setter: loads and validates the tables now,
        # so a missing or mismatched calibration fails at construction, not
        # halfway through a production run.
        self.use_quantile_calibration = use_quantile_calibration

        if self.verbose:
            members = ", ".join(
                f"fold{m.get('fold_id')}@e{m.get('epoch')}" for m in self.member_meta
            )
            print(
                f"[VBFNetEnsemble] set={self.ensemble}  members={self.n_members} [{members}]\n"
                f"[VBFNetEnsemble] route: member = {self.route}  device={self.device}  "
                f"manifest={'verified' if self.verified else 'none'}  "
                f"calibration={self._calibration_label()}\n"
                f"[VBFNetEnsemble] targets={self.target_keys}\n"
                f"[VBFNetEnsemble] heads={self.head_names}",
                flush=True,
            )

    # ── calibration switch ───────────────────────────────────────────────────

    @property
    def use_quantile_calibration(self) -> bool:
        """ON/OFF switch for quantile calibration. Settable at any time."""
        return self._use_quantile_calibration

    @use_quantile_calibration.setter
    def use_quantile_calibration(self, value: bool) -> None:
        value = bool(value)
        if value and self._calibration_tables is None:
            self._calibration_tables = self._load_calibration()
        self._use_quantile_calibration = value

    @property
    def calibration_available(self) -> bool:
        """True if a complete calibration for every member can be loaded."""
        if self._calibration_tables is not None:
            return True
        try:
            self._calibration_tables = self._load_calibration()
        except CalibrationError:
            return False
        return True

    def _calibration_label(self) -> str:
        if not self._use_quantile_calibration:
            return "off"
        return f"on ({self._calibration_layout})"

    def _calibration_dirs(self) -> tuple[str, list[Path]]:
        """Resolve one calibration directory per member, in member order."""
        root = self.calibration_dir
        per_member = [root / f"fold{fid}" for fid in self.member_ids]
        if all(d.is_dir() for d in per_member):
            return "per_member", per_member
        if any(d.is_dir() for d in per_member):
            missing = [str(d) for d in per_member if not d.is_dir()]
            raise CalibrationError(
                f"Per-member calibration in {root} is incomplete; missing {missing}. "
                "Every member needs its own calibration."
            )
        if root.is_dir() and any(root.glob("*_local_shift_correctionlib.json")):
            return "shared", [root] * self.n_members
        raise CalibrationError(
            f"No quantile calibration found in {root}. Expected {root}/fold{{k}}/ "
            f"for members {self.member_ids}, or JSONs directly in {root}."
        )

    def _load_calibration(self) -> list[dict]:
        layout, dirs = self._calibration_dirs()

        cal_manifest = (self.manifest or {}).get("calibration", {}) or {}
        released = {str(m.get("fold_id")): m for m in (self.manifest or {}).get("members", [])}
        bundled = self.calibration_dir.resolve() == (self.release_root / "calibrations").resolve()

        # For the bundled calibration, prove each file is the one installed for
        # this member, and that it was fitted on the checkpoint being loaded.
        if self.verified and bundled and cal_manifest.get("shipped"):
            root = self.release_root
            for fid, meta in zip(self.member_ids, self.member_meta):
                entry = (cal_manifest.get("members", {}) or {}).get(str(fid))
                if entry is None:
                    raise CalibrationError(f"Manifest has no calibration for fold {fid}.")
                if entry.get("member_source_sha256") != released.get(str(fid), {}).get("source_sha256"):
                    raise CalibrationError(
                        f"Calibration for fold {fid} was fitted on a different checkpoint "
                        "than the one released. Use the calibrations/ and models/ of the same release."
                    )
                for rel, expected in (entry.get("files", {}) or {}).items():
                    if sha256_file(root / rel) != expected:
                        raise CalibrationError(f"{rel} failed manifest verification.")

        tables = [
            load_shift_tables(d, self.target_keys, self.head_names) for d in dirs
        ]
        self._calibration_layout = layout
        return tables

    # ── internals ────────────────────────────────────────────────────────────

    def _check_dataset_features(self, ds) -> None:
        """Assert the built graphs carry the features the members were trained on.

        ``load_state_dict(strict=True)`` only requires the input *count* to match,
        so a dataset that silently reordered or renamed a feature would load
        cleanly and predict nonsense. The single-model release does not check
        this; it is the nastiest gap in the current deployment path.
        """
        ref = self.member_meta[0]
        for group in ("node", "edge", "global"):
            expected = ref.get(f"{group}_feature_names", None)
            actual = getattr(ds, f"{group}_feature_names", None)
            if expected is None or actual is None:
                continue
            if list(map(str, actual)) != list(map(str, expected)):
                raise EnsembleCompatibilityError(
                    f"{group} feature mismatch between the dataset and the trained "
                    f"members.\n  dataset:  {list(actual)}\n  members:  {list(expected)}\n"
                    "The graphs would be built from different inputs than the models "
                    "were trained on."
                )

    # ── inference ────────────────────────────────────────────────────────────

    def _run(self, model, loader, member_id) -> tuple[np.ndarray, list[np.ndarray]]:
        """One member over one loader: ``(pred (n, T, H) float64, truth chunks)``."""
        expected = len(self.target_keys) * self.num_heads
        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        for batch in loader:
            batch = batch.to(self.device)
            pred = model(batch)
            if pred.shape[1] != expected:
                raise RuntimeError(
                    f"Model output shape mismatch for member {member_id}: "
                    f"got {tuple(pred.shape)}, expected (B, {expected}) for "
                    f"{len(self.target_keys)} targets and {self.num_heads} heads."
                )
            preds.append(
                pred.view(pred.shape[0], len(self.target_keys), self.num_heads)
                .detach().cpu().numpy()
            )
            if getattr(batch, "y", None) is not None:
                truths.append(batch.y.detach().cpu().numpy())
        if not preds:
            return np.zeros((0, len(self.target_keys), self.num_heads)), truths
        return np.concatenate(preds, axis=0).astype(np.float64, copy=False), truths

    def _finish(self, pred_log, fold, truth, decode, calibrate, members_log=None) -> dict:
        """Everything after the forward passes: sort, calibrate, decode."""
        pred_log_raw = np.asarray(pred_log, dtype=np.float64)
        pred_log_full = sort_quantile_heads(pred_log_raw, self.head_names)

        result = {
            "pred_log_full": pred_log_full,
            "fold_id": np.asarray(fold, dtype=np.int64),
            "ensemble": self.ensemble,
            "target_keys": list(self.target_keys),
            "route": self.route,
            "n_members": self.n_members,
            "member_ids": list(self.member_ids),
            "quantile_crossing_rate": quantile_crossing_rate(pred_log_raw, self.head_names),
        }
        if truth is not None:
            result["truth_log"] = truth
        if members_log is not None:
            result["pred_log_members"] = members_log

        do_calibrate = self.use_quantile_calibration if calibrate is None else bool(calibrate)
        if do_calibrate:
            if self._calibration_tables is None:
                self._calibration_tables = self._load_calibration()
            # Rows routed to member k get member k's OWN shifts, keyed on member
            # k's own raw q50 -- the population and quantity they were fitted on.
            pred_log_cal = pred_log_raw.copy()
            for k in range(self.n_members):
                rows = np.flatnonzero(result["fold_id"] == k)
                if len(rows):
                    pred_log_cal[rows] = calibrate_member_log(
                        pred_log_raw[rows],
                        self._calibration_tables[k],
                        target_specs=self.target_specs,
                        target_keys=self.target_keys,
                        head_names=self.head_names,
                    )
            result["pred_log_cal_full"] = sort_quantile_heads(pred_log_cal, self.head_names)
            result["calibration"] = self._calibration_label()

        if decode:
            pred_phys_full = decode_predictions_array(
                pred_log_full, target_specs=self.target_specs
            )
            result["pred_phys_full"] = pred_phys_full
            # Derived observables (q1_pt, ..., mjj_p4, ...) from the routed member's p4.
            result["pred_phys"] = predictions_array_to_dict(
                pred_phys_full,
                target_keys=self.target_keys,
                head_names=self.head_names,
                include_p4_derived=True,
            )
            if do_calibrate:
                pred_phys_cal_full = decode_predictions_array(
                    result["pred_log_cal_full"], target_specs=self.target_specs
                )
                result["pred_phys_cal_full"] = pred_phys_cal_full
                result["pred_phys_cal"] = predictions_array_to_dict(
                    pred_phys_cal_full,
                    target_keys=self.target_keys,
                    head_names=self.head_names,
                    include_p4_derived=True,
                )
        return result

    def _truth_columns(self, truth_keys) -> list[int] | None:
        """Columns of a ``y`` laid out as ``truth_keys`` that hold this set's targets.

        ``None`` when ``y`` already is this set's own layout. A dataset built for
        several sets (see :class:`vbfnet_ensemble.unified.VBFNet`) carries the
        union of their targets, and each set takes its own columns by key.
        """
        if truth_keys is None:
            return None
        truth_keys = [str(k) for k in truth_keys]
        if truth_keys == list(self.target_keys):
            return None
        missing = [k for k in self.target_keys if k not in truth_keys]
        if missing:
            raise ValueError(
                f"The graphs' truth y holds {truth_keys}; set {self.ensemble!r} needs "
                f"{list(self.target_keys)} (missing {missing})."
            )
        return [truth_keys.index(k) for k in self.target_keys]

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        decode: bool = True,
        calibrate: bool | None = None,
        *,
        truth_keys: Sequence[str] | None = None,
    ):
        """Route every graph of an arbitrary loader to its member.

        Each graph must carry its CMS event number as ``data.event`` (the
        bundled dataset and ``predict_events`` set it). Every batch is split by
        ``event % n_folds`` and each part goes through its member only.
        ``calibrate`` overrides the ``use_quantile_calibration`` switch for this
        call only (``None`` = use the switch). ``truth_keys`` names the columns
        of the graphs' ``y`` when they are not this set's targets in order.
        """
        preds: list[np.ndarray] = []
        folds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        members: list[np.ndarray] = []

        for batch in loader:
            event = getattr(batch, "event", None)
            if event is None:
                raise ValueError(
                    "Graphs carry no `event` attribute: the fold-routed predictor "
                    "needs the CMS event number of every event (member = "
                    f"{self.route}). Set data.event on each graph."
                )
            fold = route_folds(event.detach().cpu().numpy().reshape(-1), self.n_folds)
            out = np.empty((len(fold), len(self.target_keys), self.num_heads))
            for k in np.unique(fold):
                rows = np.flatnonzero(fold == k)
                sub = Batch.from_data_list(batch.index_select(torch.as_tensor(rows)))
                out[rows], _ = self._run(self.models[k], [sub], k)
            preds.append(out)
            folds.append(fold)
            if getattr(batch, "y", None) is not None:
                truths.append(batch.y.detach().cpu().numpy())
            if self.return_members:
                members.append(np.stack(
                    [self._run(m, [batch], mid)[0] for m, mid in zip(self.models, self.member_ids)]
                ))

        pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(self.target_keys), self.num_heads))
        fold = np.concatenate(folds) if folds else np.zeros(0, dtype=np.int64)
        truth = np.concatenate(truths, axis=0) if truths else None
        if truth is not None:
            cols = self._truth_columns(truth_keys)
            if cols is not None:
                truth = truth[:, cols]
        members_log = np.concatenate(members, axis=1) if members else None
        return self._finish(pred, fold, truth, decode, calibrate, members_log)

    def build_dataset(
        self,
        root_files,
        tree_name: str | None = None,
        max_events: int | None = None,
        branch_map: dict | str | Path | None = None,
        require_truth: bool | None = False,
        acceptance="default",
        *,
        config: dict | None = None,
    ):
        """Build the graph dataset ``predict_root`` runs on (see its options).

        ``config`` replaces the members' own config. :class:`~vbfnet_ensemble.unified.VBFNet`
        passes one whose ``targets`` are the union of every loaded set's, so a
        single dataset serves all of them; the graph features are checked
        against this set's members here and in :meth:`predict_dataset`.
        """
        if isinstance(root_files, (str, Path)):
            root_files = [str(root_files)]
        ds = VBFJetRootDataset(
            root_files=[str(x) for x in root_files],
            tree_name=tree_name,
            max_events=max_events,
            verbose=self.verbose,
            config=self.config if config is None else config,
            branch_map=load_branch_map(branch_map),
            require_truth=require_truth,
            acceptance=resolve_acceptance(acceptance),
        )
        self._check_dataset_features(ds)
        return ds

    @torch.no_grad()
    def predict_dataset(
        self,
        ds,
        batch_size: int = 512,
        num_workers: int = 4,
        decode: bool = True,
        calibrate: bool | None = None,
    ):
        """Predict every row of a dataset from :meth:`build_dataset`.

        The rows of fold k go through member k only, one loader per fold, so the
        cost is ONE forward pass per event. The dataset is not modified, so the
        same one can be passed to several sets.
        """
        self._check_dataset_features(ds)

        event_ids = np.asarray(ds.event_ids, dtype=np.int64).reshape(-1, 3)
        fold = route_folds(event_ids[:, 2], self.n_folds)
        n = len(ds)
        pred = np.empty((n, len(self.target_keys), self.num_heads))
        truth = None
        pin = self.device.type == "cuda"

        for k in range(self.n_members):
            rows = np.flatnonzero(fold == k)
            if not len(rows):
                continue
            loader = DataLoader(
                subset_dataset(ds, rows), batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
            )
            pred[rows], truths = self._run(self.models[k], loader, k)
            if truths:
                if truth is None:
                    truth = np.empty((n,) + truths[0].shape[1:])
                truth[rows] = np.concatenate(truths, axis=0)

        if truth is not None:
            cols = self._truth_columns(getattr(ds, "target_keys", None) or None)
            if cols is not None:
                truth = truth[:, cols]

        members_log = None
        if self.return_members:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin)
            members_log = np.stack(
                [self._run(m, loader, mid)[0] for m, mid in zip(self.models, self.member_ids)]
            )

        result = self._finish(pred, fold, truth, decode, calibrate, members_log)
        # Raw entry index of each kept row (single file: the TTree entry), so
        # callers can realign per-raw-event arrays -- see make_bdt_inputs.
        result["event_index"] = np.asarray(ds.raw_event_indices, dtype=np.int64)
        result["run"] = event_ids[:, 0]
        result["lumi"] = event_ids[:, 1]
        result["event"] = event_ids[:, 2]
        result["acceptance"] = getattr(ds, "acceptance", None)
        return result

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
        """Run inference from ROOT file(s).

        ``require_truth=False`` (deployment default): no generator-level cuts,
        the only selection is ``acceptance`` -- ``"default"`` (>= 2 VBF jets
        with pT >= 50 GeV and |eta| <= 4.7; a gate only, the graph keeps every
        jet), a dict with the same keys, or ``None`` for no gate.
        ``require_truth=True`` applies the TRAINING cuts and returns
        ``truth_log``; with ``acceptance=None`` it reproduces the training event
        list, which is how a release is checked against its out-of-fold predictions.

        ``branch_map`` reads files whose branches have other names: a dict
        ``{logical: actual}`` or the path of a YAML file laid out like
        ``branch_map.yaml`` (see :mod:`vbfnet_ensemble.branch_map`).

        The dataset is built once (:meth:`build_dataset`); the rows of fold k go
        through member k only (:meth:`predict_dataset`), so the cost is ONE
        forward pass per event.
        """
        acceptance = resolve_acceptance(acceptance)
        ds = self.build_dataset(
            root_files, tree_name=tree_name, max_events=max_events, branch_map=branch_map,
            require_truth=require_truth, acceptance=acceptance,
        )
        result = self.predict_dataset(
            ds, batch_size=batch_size, num_workers=num_workers, decode=decode, calibrate=calibrate,
        )
        result["acceptance"] = acceptance
        return result

    def predict_events(
        self,
        events,
        batch_size: int = 512,
        num_workers: int = 0,
        decode: bool = True,
        calibrate: bool | None = None,
        acceptance="default",
    ):
        """Run inference from in-memory physics objects.

        Same event dict layout as ``trained_model_VBFNet.VBFNet.predict_events``
        plus a REQUIRED ``event`` key (the CMS event number, which picks the
        member). Events failing the acceptance gate get no prediction; the
        returned ``input_index`` says which inputs each row came from.
        """
        graphs, kept = graphs_from_events(
            events, self.config, acceptance, route=self.route, verbose=self.verbose,
        )
        loader = DataLoader(
            DataListDataset(graphs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        result = self.predict_loader(loader, decode=decode, calibrate=calibrate)
        result["input_index"] = np.asarray(kept, dtype=np.int64)
        return result


def graphs_from_events(events, cfg, acceptance="default", *, route=ROUTE_RULE, verbose=True):
    """Build the graphs of in-memory events: ``(graphs, kept input indices)``.

    Each event dict needs ``hbb``, ``htt``, ``met``, optionally ``vbf_jets``, and
    the CMS ``event`` number, which picks the member. Events failing the
    acceptance gate are skipped; raises if none is left.
    """
    acceptance = resolve_acceptance(acceptance)
    graphs, kept = [], []
    for i, ev in enumerate(events):
        if "event" not in ev:
            raise KeyError(
                f"events[{i}] has no 'event' key: the fold-routed predictor needs "
                f"the CMS event number of every event (member = {route})."
            )
        data = build_data_from_arrays(
            vbf_jets=ev.get("vbf_jets"),
            hbb=ev["hbb"],
            htt=ev["htt"],
            met=ev["met"],
            cfg=cfg,
            acceptance=acceptance,
        )
        if data is not None:
            data.event = torch.tensor([int(ev["event"])], dtype=torch.long)
            graphs.append(data)
            kept.append(i)

    if not graphs:
        raise ValueError("No usable events: every event failed the acceptance gate.")
    n_dropped = len(events) - len(graphs)
    if n_dropped and verbose:
        print(
            f"[VBFNetEnsemble] predict_events: {n_dropped} event(s) failed the "
            "acceptance gate and got no prediction.",
            flush=True,
        )
    return graphs, kept
