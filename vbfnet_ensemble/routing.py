"""Fold routing: which member predicts which event.

Pure numpy: no torch, no torch_geometric.

Each event is predicted by EXACTLY ONE member, chosen by the same rule the
training split used (``vbf_kfold.assign_folds``, byte-identical to the split
code used in training)::

    k = event % n_folds        ->  member k predicts this event

For an event from the training samples, member k is the one model that never
trained on it, so the deployed prediction is exactly the out-of-fold (OOF)
prediction that was measured. Background and data are out-of-sample for every
member; the rule just gives them a deterministic, reproducible member.

This replaces the median over all members. A median on the training samples was
in-sample (K-1 of the K members had trained on every event), and it treated
signal (seen in training) and background (never seen) differently.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .vbf_kfold import assign_folds

#: The routing rule, as recorded in the manifest and returned as ``route``.
ROUTE_RULE = "event % n_folds"

#: Event-level reco gate applied at deployment (no truth cuts): at least
#: ``min_jets`` VBF jets with ``pt >= jet_min_pt`` GeV and
#: ``|eta| <= jet_max_abs_eta``. Gate only: the graph keeps ALL jets.
DEFAULT_ACCEPTANCE = {"min_jets": 2, "jet_min_pt": 50.0, "jet_max_abs_eta": 4.7}


def resolve_acceptance(acceptance) -> dict | None:
    """``"default"`` -> :data:`DEFAULT_ACCEPTANCE`; ``None``/``{}`` -> off; a dict -> checked copy."""
    if acceptance is None:
        return None
    if isinstance(acceptance, str):
        if acceptance != "default":
            raise ValueError(f"acceptance={acceptance!r}: use 'default', None or a dict.")
        return dict(DEFAULT_ACCEPTANCE)
    acceptance = dict(acceptance)
    if not acceptance:
        return None
    missing = set(DEFAULT_ACCEPTANCE) - set(acceptance)
    if missing:
        raise ValueError(f"acceptance is missing {sorted(missing)}; need {sorted(DEFAULT_ACCEPTANCE)}.")
    return {
        "min_jets": int(acceptance["min_jets"]),
        "jet_min_pt": float(acceptance["jet_min_pt"]),
        "jet_max_abs_eta": float(acceptance["jet_max_abs_eta"]),
    }


def route_folds(event, n_folds: int) -> np.ndarray:
    """Member index of every event: ``event % n_folds`` (raises on a missing event number)."""
    event = np.asarray(event, dtype=np.int64)
    if event.size == 0:
        return np.zeros(0, dtype=np.int64)
    return assign_folds(event, n_folds)


def sort_quantile_heads(
    pred_log_full: np.ndarray,
    head_names: Sequence[str],
) -> np.ndarray:
    """Return a copy with the quantile heads sorted ascending along the head axis.

    The point head is left in place. The model already sorts its quantiles, so
    this is a cheap guard; calibration re-sorts on its own as well.
    """
    pred_log_full = np.asarray(pred_log_full, dtype=np.float64)
    q_idx = [i for i, h in enumerate(head_names) if str(h).startswith("q")]
    if len(q_idx) < 2:
        return pred_log_full.copy()

    out = pred_log_full.copy()
    out[:, :, q_idx] = np.sort(out[:, :, q_idx], axis=2)
    return out


def quantile_crossing_rate(
    pred_log_full: np.ndarray,
    head_names: Sequence[str],
) -> float:
    """Fraction of (event, target) pairs whose quantile heads are out of order."""
    pred_log_full = np.asarray(pred_log_full, dtype=np.float64)
    q_idx = [i for i, h in enumerate(head_names) if str(h).startswith("q")]
    if len(q_idx) < 2 or pred_log_full.shape[0] == 0:
        return 0.0

    q = pred_log_full[:, :, q_idx]
    bad = np.any(np.diff(q, axis=2) < 0.0, axis=2)
    return float(np.mean(bad))
