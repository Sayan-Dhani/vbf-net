"""Pure-numpy target transforms, decoding and prediction reshaping.

Kept free of heavy imports so that the decode
path can be imported — and unit-tested — without torch, torch_geometric or
correctionlib. ``predictor.py`` re-exports everything here, so the public API is
unchanged from the single-model release.

IMPORTANT: this decode/transform contract must stay identical to the one the
members were trained with (``vbf_config.py`` holds the same rules). Changing it
silently corrupts predictions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Number of legacy targets (mjj/deta/eta_prod/ptsum). Hardcoded rather than
# imported from .pyg_vbf_dataset so this module stays free of torch.
NUM_TARGETS = 4


LEGACY_TARGETS = ["mjj", "deta", "eta_prod", "ptsum"]
LEGACY_TARGET_SPECS = [
    {"key": "mjj", "transform": "asinh"},
    {"key": "deta", "transform": "identity"},
    {"key": "eta_prod", "transform": "identity"},
    {"key": "ptsum", "transform": "log1p"},
]
DEFAULT_QUANTILES = [0.16, 0.50, 0.84]

#: The quark four-vector components the p4 set regresses.
P4_TARGET_KEYS = ["q1_E", "q1_px", "q1_py", "q1_pz", "q2_E", "q2_px", "q2_py", "q2_pz"]

#: Suffix of the di-quark observables DERIVED from the p4 heads. The plain names
#: (mjj, deta, eta_prod, ptsum) belong to the set that REGRESSES them (hl), so a
#: plain ``mjj`` never changes meaning with which sets are loaded.
P4_DERIVED_SUFFIX = "_p4"

#: Everything :func:`_derive_vbf_observables_from_p4` adds, in output order.
P4_DERIVED_KEYS = [
    "q1_pt", "q1_eta", "q1_phi", "q1_mass",
    "q2_pt", "q2_eta", "q2_phi", "q2_mass",
] + [f"{k}{P4_DERIVED_SUFFIX}" for k in LEGACY_TARGETS]


def _quantile_name(q: float) -> str:
    return f"q{int(round(100.0 * float(q)))}"


def _head_names_from_metadata(
    *,
    head_names: Sequence[str] | None = None,
    output_mode: str = "both",
    quantiles: Sequence[float] | None = None,
) -> list[str]:
    if head_names:
        return [str(x) for x in head_names]

    qnames = [_quantile_name(q) for q in (quantiles or DEFAULT_QUANTILES)]
    mode = str(output_mode or "both").lower()

    names: list[str] = []
    if mode in {"both", "quantile"}:
        names.extend(qnames)
    if mode in {"both", "point"}:
        names.append("point")

    if not names:
        raise ValueError(f"Invalid output mode '{output_mode}'")
    return names


def _target_specs_from_ckpt(ckpt: dict) -> list[dict]:
    specs = ckpt.get("target_specs", None)
    if specs:
        return [dict(s) for s in specs]

    cfg = ckpt.get("config", None) or {}
    specs = cfg.get("targets", None)
    if specs:
        return [dict(s) for s in specs]

    keys = ckpt.get("target_keys", None)
    if keys:
        return [{"key": str(k), "transform": "identity"} for k in keys]

    return LEGACY_TARGET_SPECS[: int(ckpt.get("num_targets", NUM_TARGETS))]


def _target_keys_from_specs(specs: Sequence[dict]) -> list[str]:
    return [str(s.get("key", s.get("name"))) for s in specs]


def _inverse_transform_array(z: np.ndarray, transform: str | None) -> np.ndarray:
    """Invert the scalar transform used in the YAML target specification.

    Kept only as the fallback when a target spec has no explicit ``decode``
    field. New code paths go through :func:`_decode_array`, which honours
    ``decode`` first (see :func:`_decode_rule_for_spec`).
    """
    z = np.asarray(z, dtype=np.float64)
    t = str(transform or "identity").strip().lower()

    if t in {"identity", "none", "null", "raw", ""}:
        return z.copy()
    if t in {"asinh", "arcsinh"}:
        return np.sinh(z)
    if t in {"log1p", "log"}:
        return np.expm1(z)
    if t in {"signed_log1p", "sl1p", "signedlog1p"}:
        return np.sign(z) * np.expm1(np.abs(z))

    raise ValueError(
        f"Unsupported target transform '{transform}'. "
        "Add its inverse in vbfnet_ensemble/transforms.py."
    )


def _normalise_rule(rule: object) -> str:
    """Normalise a decode/transform rule string. Mirrors eval/config._normalise_rule."""
    if rule is None:
        return "identity"
    r = str(rule).strip().lower()
    aliases = {
        "none": "identity",
        "null": "identity",
        "raw": "identity",
        "linear": "identity",
        "asinh_inverse": "sinh",
        "inverse_asinh": "sinh",
        "log1p_inverse": "expm1",
        "inverse_log1p": "expm1",
        "signed_expm1m": "signed_expm1",
        "sl1p_inv": "signed_expm1",
    }
    return aliases.get(r, r)


# Maps a forward transform to its inverse (decode), used only when a spec
# does not set an explicit ``decode`` field. Mirrors eval/config.
_TRANSFORM_TO_DECODE = {
    "identity": "identity",
    "asinh": "sinh",
    "log1p": "expm1",
    "signed_log1p": "signed_expm1",
    "sl1p": "signed_expm1",
}


def _decode_rule_for_spec(spec: dict) -> str:
    """Resolve the authoritative decode rule for a target spec.

    The explicit ``decode`` field wins (matching vbf_config.decode_* and
    eval/config.decode_phys). Only when it is absent do we fall back to the
    inverse of ``transform``. This keeps the deployed predictor in agreement
    with training/eval even when ``decode`` is set independently of
    ``transform`` (both are permitted by the config schema).
    """
    if spec.get("decode") is not None:
        return _normalise_rule(spec.get("decode"))
    tr = _normalise_rule(spec.get("transform", "identity"))
    return _TRANSFORM_TO_DECODE.get(tr, "identity")


def _decode_array(z: np.ndarray, rule: str) -> np.ndarray:
    """Apply a (normalised) decode rule to an array. Decode direction only."""
    z = np.asarray(z, dtype=np.float64)
    rule = _normalise_rule(rule)

    if rule == "identity":
        return z.copy()
    if rule == "sinh":
        return np.sinh(z)
    if rule == "expm1":
        return np.expm1(z)
    if rule == "signed_expm1":
        return np.sign(z) * np.expm1(np.abs(z))
    if rule == "exp":
        return np.exp(z)

    raise ValueError(
        f"Unsupported decode rule '{rule}'. "
        "Add it in vbfnet_ensemble/transforms.py:_decode_array."
    )


def _encode_array(x: np.ndarray, decode_rule: str) -> np.ndarray:
    """Forward transform (physical -> model space) = inverse of ``decode_rule``.

    Used to re-encode physical-space calibrated quantiles back to model space so
    the returned model-space array stays consistent with the physical one.
    """
    x = np.asarray(x, dtype=np.float64)
    rule = _normalise_rule(decode_rule)

    if rule == "identity":
        return x.copy()
    if rule == "sinh":            # inverse of sinh
        return np.arcsinh(x)
    if rule == "expm1":           # inverse of expm1
        return np.log1p(x)
    if rule == "signed_expm1":    # inverse of signed_expm1
        return np.sign(x) * np.log1p(np.abs(x))
    if rule == "exp":             # inverse of exp
        return np.log(x)

    raise ValueError(
        f"Unsupported encode for decode rule '{rule}'. "
        "Add it in vbfnet_ensemble/transforms.py:_encode_array."
    )


def decode_predictions_array(
    pred_log_full: np.ndarray,
    target_specs: Sequence[dict] | None = None,
) -> np.ndarray:
    """
    Decode full prediction array from model space to physical target space.

    Parameters
    ----------
    pred_log_full:
        Array with shape (N, nt, n_heads).
    target_specs:
        YAML target specs. If omitted, the legacy 4-target setup is assumed.
    """
    pred_log_full = np.asarray(pred_log_full, dtype=np.float64)
    pred_phys_full = pred_log_full.copy()

    if target_specs is None:
        target_specs = LEGACY_TARGET_SPECS[: pred_log_full.shape[1]]

    if len(target_specs) != pred_log_full.shape[1]:
        raise ValueError(
            f"Number of target specs ({len(target_specs)}) does not match "
            f"prediction target dimension ({pred_log_full.shape[1]})."
        )

    for ti, spec in enumerate(target_specs):
        pred_phys_full[:, ti, :] = _decode_array(
            pred_log_full[:, ti, :],
            _decode_rule_for_spec(spec),
        )

    return pred_phys_full


def _eta_from_px_py_pz(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    pt = np.sqrt(px * px + py * py)
    return np.arcsinh(pz / np.maximum(pt, 1e-9))


def _pt_from_px_py(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.sqrt(px * px + py * py)


def _phi_from_px_py(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.arctan2(py, px)


def _mass_from_p4(E: np.ndarray, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Invariant mass sqrt(max(E^2 - |p|^2, 0)); clamped to avoid NaNs from
    tiny negative m^2 caused by per-component regression noise."""
    m2 = E * E - px * px - py * py - pz * pz
    return np.sqrt(np.maximum(m2, 0.0))


def _derive_vbf_observables_from_p4(out: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Add derived observables when q1/q2 cartesian four-vector targets exist.

    Two families are produced from the (E, px, py, pz) heads of each jet:

      * per-jet cylindrical:  q1_pt/q1_eta/q1_phi/q1_mass (and q2_*),
      * di-jet high-level:    mjj_p4, deta_p4, eta_prod_p4, ptsum_p4.

    The di-jet names carry :data:`P4_DERIVED_SUFFIX` because the plain names are
    the hl set's REGRESSED observables.

    These derived q16/q50/q84 values are transforms of component-wise p4 heads.
    They are useful BDT inputs, but they are not guaranteed calibrated quantiles of
    the derived observable unless calibrated separately at observable level.
    """
    p4_keys = P4_TARGET_KEYS
    if not all(k in out for k in p4_keys):
        return

    common_heads = set(out["q1_E"].keys())
    for key in p4_keys[1:]:
        common_heads &= set(out[key].keys())

    if not common_heads:
        return

    for target in P4_DERIVED_KEYS:
        out.setdefault(target, {})
    s = P4_DERIVED_SUFFIX

    for head in sorted(common_heads):
        E1 = np.asarray(out["q1_E"][head], dtype=np.float64)
        px1 = np.asarray(out["q1_px"][head], dtype=np.float64)
        py1 = np.asarray(out["q1_py"][head], dtype=np.float64)
        pz1 = np.asarray(out["q1_pz"][head], dtype=np.float64)

        E2 = np.asarray(out["q2_E"][head], dtype=np.float64)
        px2 = np.asarray(out["q2_px"][head], dtype=np.float64)
        py2 = np.asarray(out["q2_py"][head], dtype=np.float64)
        pz2 = np.asarray(out["q2_pz"][head], dtype=np.float64)

        # ── per-jet cylindrical (pt, eta, phi, mass) ──────────────────────────
        eta1 = _eta_from_px_py_pz(px1, py1, pz1)
        eta2 = _eta_from_px_py_pz(px2, py2, pz2)
        pt1 = _pt_from_px_py(px1, py1)
        pt2 = _pt_from_px_py(px2, py2)

        out["q1_pt"][head] = pt1
        out["q1_eta"][head] = eta1
        out["q1_phi"][head] = _phi_from_px_py(px1, py1)
        out["q1_mass"][head] = _mass_from_p4(E1, px1, py1, pz1)

        out["q2_pt"][head] = pt2
        out["q2_eta"][head] = eta2
        out["q2_phi"][head] = _phi_from_px_py(px2, py2)
        out["q2_mass"][head] = _mass_from_p4(E2, px2, py2, pz2)

        # ── di-jet high-level (mjj_p4, deta_p4, eta_prod_p4, ptsum_p4) ────────
        E = E1 + E2
        px = px1 + px2
        py = py1 + py2
        pz = pz1 + pz2

        out[f"mjj{s}"][head] = _mass_from_p4(E, px, py, pz)
        out[f"deta{s}"][head] = np.abs(eta1 - eta2)
        out[f"eta_prod{s}"][head] = eta1 * eta2
        out[f"ptsum{s}"][head] = pt1 + pt2


def predictions_array_to_dict(
    pred_phys_full: np.ndarray,
    target_keys: Sequence[str] | None = None,
    head_names: Sequence[str] | None = None,
    *,
    include_p4_derived: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Convert array predictions to nested dictionary format.

    Output:
        out[target_key][head_name] -> array with length N
    """
    pred_phys_full = np.asarray(pred_phys_full, dtype=np.float64)
    nt = pred_phys_full.shape[1]
    nh = pred_phys_full.shape[2]

    if target_keys is None:
        target_keys = LEGACY_TARGETS[:nt]
    if head_names is None:
        head_names = _head_names_from_metadata(output_mode="both")[:nh]

    if len(target_keys) != nt:
        raise ValueError(f"target_keys length {len(target_keys)} != prediction nt {nt}")
    if len(head_names) != nh:
        raise ValueError(f"head_names length {len(head_names)} != prediction n_heads {nh}")

    out: dict[str, dict[str, np.ndarray]] = {}
    for ti, target in enumerate(target_keys):
        out[str(target)] = {
            str(head): pred_phys_full[:, ti, hi]
            for hi, head in enumerate(head_names)
        }

    if include_p4_derived:
        _derive_vbf_observables_from_p4(out)

    return out


def decode_predictions(
    pred_log_full: np.ndarray,
    target_specs: Sequence[dict] | None = None,
    head_names: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    pred_phys_full = decode_predictions_array(pred_log_full, target_specs=target_specs)
    keys = _target_keys_from_specs(target_specs or LEGACY_TARGET_SPECS[: pred_phys_full.shape[1]])
    return predictions_array_to_dict(
        pred_phys_full,
        target_keys=keys,
        head_names=head_names,
        include_p4_derived=True,
    )

