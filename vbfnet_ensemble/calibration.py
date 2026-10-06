"""Quantile-head calibration.

Two entry points:

* :func:`calibrate_member_log` — what :class:`VBFNetEnsemble` uses. Applies one
  member's calibration to that member's raw predictions, with a vectorised
  numpy lookup of the correctionlib ``binning`` nodes. No correctionlib needed.
* :func:`apply_quantile_calibration` — the single-model release's function,
  kept verbatim (correctionlib, per-event scalar calls) for API compatibility.

The calibration is a local q50-binned additive shift in PHYSICAL space::

    q_alpha_cal = q_alpha_raw + delta_alpha( bin of raw q50 )

The binning variable is the member's OWN raw q50. That is why, in an ensemble,
each member must be calibrated *before* the reduction: the shifts were fitted
against one member's q50 distribution and mean nothing against the median's.
The point head is never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import numpy as np

from .transforms import (
    LEGACY_TARGET_SPECS,
    _decode_rule_for_spec,
    _encode_array,
    _head_names_from_metadata,
    _target_keys_from_specs,
    decode_predictions_array,
)


def apply_quantile_calibration(
    pred_log_full: np.ndarray,
    calibration_dir: str | Path,
    *,
    target_specs: Sequence[dict] | None = None,
    target_keys: Sequence[str] | None = None,
    head_names: Sequence[str] | None = None,
    sort_quantiles: bool = True,
    skip_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply correctionlib quantile calibration to direct model targets.

    The corrections are additive shifts fitted and stored in **physical space**
    (GeV), keyed by the raw predicted physical q50 — exactly what
    the calibration fit produces. They are therefore applied in physical
    space (``q_phys_cal = q_phys_raw + delta``) and the result is re-encoded to
    model space; adding the physical shift to the model-space (e.g. signed_log1p)
    prediction and then decoding would blow up. The point head is left unchanged.
    Missing JSONs are skipped by default so p4 checkpoints can still be used when
    only derived BDT variables are needed.
    """
    try:
        import correctionlib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "correctionlib is required to apply quantile calibration "
            "(pip install correctionlib). It is an optional dependency of "
            "vbfnet_ensemble; VBFNetEnsemble itself does not need it."
        ) from exc

    calibration_dir = Path(calibration_dir)
    pred_log_full = np.asarray(pred_log_full, dtype=np.float64)

    nt, nh = pred_log_full.shape[1], pred_log_full.shape[2]

    if target_specs is None:
        target_specs = LEGACY_TARGET_SPECS[:nt]
    if target_keys is None:
        target_keys = _target_keys_from_specs(target_specs)
    if head_names is None:
        head_names = _head_names_from_metadata(output_mode="both")[:nh]

    pred_phys_raw = decode_predictions_array(pred_log_full, target_specs=target_specs)
    pred_phys_cal = pred_phys_raw.copy()

    q_head_indices = [i for i, h in enumerate(head_names) if str(h).startswith("q")]
    q50_index = head_names.index("q50") if "q50" in head_names else (q_head_indices[len(q_head_indices) // 2] if q_head_indices else None)

    if q50_index is None:
        return pred_log_full.copy(), pred_phys_raw

    missing: list[str] = []

    for ti, target in enumerate(target_keys):
        json_path = calibration_dir / f"{target}_local_shift_correctionlib.json"
        if not json_path.exists():
            if skip_missing:
                missing.append(str(json_path.name))
                continue
            raise FileNotFoundError(f"Missing calibration JSON for {target}: {json_path}")

        cset = correctionlib.CorrectionSet.from_file(str(json_path))
        raw_q50_phys = pred_phys_raw[:, ti, q50_index]   # binning variable (physical)

        for qi in q_head_indices:
            qname = str(head_names[qi])
            corr_name = f"{target}_local_shift_{qname}"
            if corr_name not in cset:
                if skip_missing:
                    missing.append(f"{json_path.name}:{corr_name}")
                    continue
                raise KeyError(f"Missing correction '{corr_name}' in {json_path}")

            delta = np.asarray(
                [cset[corr_name].evaluate(float(x)) for x in raw_q50_phys],
                dtype=np.float64,
            )
            # Physical shift applied in physical space.
            pred_phys_cal[:, ti, qi] = pred_phys_raw[:, ti, qi] + delta

    if missing:
        preview = ", ".join(missing[:6])
        more = "" if len(missing) <= 6 else f", ... ({len(missing)} missing entries total)"
        warnings.warn(
            "Some quantile calibration JSON/corrections were not found and were skipped: "
            f"{preview}{more}",
            RuntimeWarning,
        )

    # Enforce monotone quantiles in physical space.
    if sort_quantiles and len(q_head_indices) >= 2:
        pred_phys_cal[:, :, q_head_indices] = np.sort(pred_phys_cal[:, :, q_head_indices], axis=2)

    # Re-encode the calibrated quantile heads to model space so the returned
    # model-space array stays consistent with the physical one (point head kept as-is).
    pred_log_cal = pred_log_full.copy()
    for ti, spec in enumerate(target_specs):
        rule = _decode_rule_for_spec(spec)
        for qi in q_head_indices:
            pred_log_cal[:, ti, qi] = _encode_array(pred_phys_cal[:, ti, qi], rule)

    return pred_log_cal, pred_phys_cal



# ─────────────────────────────────────────────────────────────────────────────
# Vectorised per-member calibration (used by VBFNetEnsemble)
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_SUFFIX = "_local_shift_correctionlib.json"


class CalibrationError(ValueError):
    """Raised when a calibration is missing, malformed or does not fit the model."""


def _binning_node(correction: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(edges, content)`` from a flat correctionlib binning node.

    Only the exact shape the calibration fit writes is accepted — a
    single binning node over ``raw_q50`` with float content and ``clamp`` flow.
    Anything else is refused rather than approximated, because the vectorised
    lookup below reproduces correctionlib's semantics only for that shape.
    """
    data = correction.get("data", {})
    name = correction.get("name", "?")
    if data.get("nodetype") != "binning":
        raise CalibrationError(f"{name}: expected a binning node, got {data.get('nodetype')!r}")
    if data.get("flow") != "clamp":
        raise CalibrationError(f"{name}: expected flow='clamp', got {data.get('flow')!r}")
    edges = np.asarray(data["edges"], dtype=np.float64)
    content = data["content"]
    if not all(isinstance(c, (int, float)) for c in content):
        raise CalibrationError(f"{name}: nested content is not supported by the fast lookup")
    content = np.asarray(content, dtype=np.float64)
    if edges.ndim != 1 or len(edges) != len(content) + 1 or np.any(np.diff(edges) <= 0):
        raise CalibrationError(f"{name}: malformed edges ({len(edges)}) / content ({len(content)})")
    return edges, content


def lookup_shift(edges: np.ndarray, content: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a clamped binning node on an array. Matches correctionlib.

    Bin ``i`` holds ``edges[i] <= x < edges[i+1]``; values below the first edge
    use bin 0 and values at or above the last edge use the last bin (``clamp``).
    Non-finite inputs give NaN rather than a silently clamped shift.

    Verified against correctionlib on the shipped files. The one known
    difference: an input lying EXACTLY on an edge can land one bin to the left
    in correctionlib, whose parsed edge sits one ulp above the JSON value. That
    needs a network output to hit a bin edge to the last bit, which does not
    happen with continuous predictions.
    """
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(content) - 1)
    out = content[idx]
    return np.where(np.isfinite(x), out, np.nan)


def load_shift_tables(
    directory: str | Path,
    target_keys: Sequence[str],
    head_names: Sequence[str],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load ``{target: {quantile_head: (edges, content)}}`` from one directory.

    Every target and every quantile head must be present: a partially
    calibrated member would silently mix calibrated and raw quantiles.
    """
    import json

    directory = Path(directory)
    q_heads = [h for h in head_names if str(h).startswith("q")]
    tables: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    missing: list[str] = []

    for target in target_keys:
        path = directory / f"{target}{CALIBRATION_SUFFIX}"
        if not path.exists():
            missing.append(path.name)
            continue
        by_name = {c["name"]: c for c in json.loads(path.read_text()).get("corrections", [])}
        tables[target] = {}
        for head in q_heads:
            name = f"{target}_local_shift_{head}"
            if name not in by_name:
                missing.append(f"{path.name}:{name}")
                continue
            tables[target][head] = _binning_node(by_name[name])

    if missing:
        raise CalibrationError(
            f"Incomplete calibration in {directory}: missing {missing}. "
            "Refusing to calibrate some quantiles and not others."
        )
    return tables


def calibrate_member_log(
    member_log: np.ndarray,
    tables: dict,
    *,
    target_specs: Sequence[dict],
    target_keys: Sequence[str],
    head_names: Sequence[str],
) -> np.ndarray:
    """Calibrate ONE member's model-space predictions, ``(N, nt, nh)``.

    Decode -> add the physical-space shift keyed on this member's raw q50 ->
    sort the quantile heads -> re-encode the quantile heads. The point head is
    returned bit-identical (it is copied, never round-tripped through
    decode/encode). Same result as :func:`apply_quantile_calibration` with
    ``sort_quantiles=True``, minus the per-event Python loop.
    """
    member_log = np.asarray(member_log, dtype=np.float64)
    head_names = list(head_names)
    q_idx = [i for i, h in enumerate(head_names) if str(h).startswith("q")]
    if "q50" not in head_names or not q_idx:
        raise CalibrationError(f"Quantile calibration needs a q50 head; heads are {head_names}")
    i50 = head_names.index("q50")

    phys = decode_predictions_array(member_log, target_specs=target_specs)
    out = member_log.copy()

    for ti, target in enumerate(target_keys):
        raw_q50 = phys[:, ti, i50]
        cal = phys[:, ti, q_idx].copy()
        for j, qi in enumerate(q_idx):
            edges, content = tables[target][head_names[qi]]
            cal[:, j] = phys[:, ti, qi] + lookup_shift(edges, content, raw_q50)
        cal = np.sort(cal, axis=1)
        out[:, ti, q_idx] = _encode_array(cal, _decode_rule_for_spec(target_specs[ti]))

    return out
