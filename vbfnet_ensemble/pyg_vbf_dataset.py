from __future__ import annotations

"""Build PyG graphs for VBF-jet regression from CMS ROOT files.

Each accepted event becomes one fully connected directed graph:
- nodes: reconstructed VBF jets
- edges: all ordered jet pairs without self-loops
- globals: Hbb, Htt, MET, and event activity
- targets: truth-level asinh(m_jj), |Δη_jj|, η_i*η_j and log1p(pT_sum)
"""

import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .vbf_config import (
    config_hash,
    load_vbf_config,
    target_specs,
    target_keys,
    num_targets,
    transform_scalar,
)

try:
    import uproot
except ImportError as e:
    raise ImportError("uproot and awkward are required: pip install uproot awkward") from e

_ROOT = None


def _get_root():
    """Lazy-load PyROOT only when ROOT four-vector operations are needed."""
    global _ROOT
    if _ROOT is None:
        try:
            import ROOT as _root
        except ImportError as e:
            raise ImportError(
                "PyROOT (ROOT) is required to build datasets from ROOT TTrees. "
                "Cached .pt datasets can still be loaded without PyROOT."
            ) from e
        _ROOT = _root
    return _ROOT

ALL_NODE_FEATURE_NAMES: List[str] = [
    "log(pt+1)",
    "eta",
    "sin(phi)",
    "cos(phi)",
    "log(mass+1)",
    "log(E+1)",
    "sl1p(px)",
    "sl1p(py)",
    "sl1p(pz)",
    "centrality",
    "dEta(hbb)",
    "sin(dPhi(hbb))",
    "cos(dPhi(hbb))",
    "dR(hbb)",
    "dEta(htt)",
    "sin(dPhi(htt))",
    "cos(dPhi(htt))",
    "dR(htt)",
    "btagDeepFlavB",
    "btagDeepFlavCvB",
    "btagDeepFlavCvL",
    "btagDeepFlavQG",
    "n_constituents",
]

ALL_GLOBAL_FEATURE_NAMES: List[str] = [
    "hbb_logpt",
    "hbb_eta",
    "hbb_sinphi",
    "hbb_cosphi",
    "hbb_logmass",
    "hbb_logE",
    "hbb_sl1p_px",
    "hbb_sl1p_py",
    "hbb_sl1p_pz",
    "htt_logpt",
    "htt_eta",
    "htt_sinphi",
    "htt_cosphi",
    "htt_logmass",
    "htt_logE",
    "htt_sl1p_px",
    "htt_sl1p_py",
    "htt_sl1p_pz",
    "met_logpt",
    "met_sinphi",
    "met_cosphi",
    "nVBFJet_raw",
    "log(HT_raw)",
]

ALL_EDGE_FEATURE_NAMES: List[str] = [
    "delta_eta",
    "eta_product",
    "sin_dphi",
    "cos_dphi",
    "delta_r",
    "asinh_m_ij",
    "log_pt_ratio",
    "kt_dist",
]

# Backward-compatible defaults for old plotting/debug scripts.
# These legacy constants describe the original 4-target setup. New training and
# inference code should use the YAML config and dataset instance properties.
NODE_FEATURE_NAMES = ALL_NODE_FEATURE_NAMES.copy()
GLOBAL_FEATURE_NAMES = ALL_GLOBAL_FEATURE_NAMES.copy()
EDGE_FEATURE_NAMES = ALL_EDGE_FEATURE_NAMES.copy()

TARGET_NAMES: List[str] = [
    "asinh(m_jj)",
    "|Δη_jj|",
    "η_i*η_j",
    "log1p(pt_q_lead + pt_q_sub)",
]
NUM_TARGETS = len(TARGET_NAMES)

NUM_NODE_FEATURES = len(NODE_FEATURE_NAMES)
NUM_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)
NUM_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)


def _cfg_feature_names(cfg: dict, group: str) -> list[str]:
    names = list(cfg["features"][group])
    allowed = {
        "node": ALL_NODE_FEATURE_NAMES,
        "edge": ALL_EDGE_FEATURE_NAMES,
        "global": ALL_GLOBAL_FEATURE_NAMES,
    }[group]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise KeyError(
            f"Unknown {group} features in YAML: {unknown}\n"
            f"Allowed: {allowed}"
        )
    return names

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Return sign(x) * log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _delta_phi_np(a: float, b: float) -> float:
    """Wrapped Δφ in (-π, π]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def make_lorentzvector_ptetaphim(pt, eta, phi, mass):
    """Build a ROOT TLorentzVector from (pt, eta, phi, mass)."""
    ROOT = _get_root()
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(float(pt), float(eta), float(phi), float(mass))
    return vec


def _p4_components(pt: float, eta: float, phi: float, mass: float):
    """Return (E, px, py, pz) from (pt, eta, phi, mass)."""
    ROOT = _get_root()
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(float(pt), float(eta), float(phi), float(mass))
    return vec.E(), vec.Px(), vec.Py(), vec.Pz()


def _delta_phi_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized wrapped Δφ in (-π, π]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def _inv_mass_pair_torch(
    pt_i: torch.Tensor,
    eta_i: torch.Tensor,
    phi_i: torch.Tensor,
    m_i: torch.Tensor,
    pt_j: torch.Tensor,
    eta_j: torch.Tensor,
    phi_j: torch.Tensor,
    m_j: torch.Tensor,
) -> torch.Tensor:
    """Vectorized invariant mass for jet pairs."""
    px_i = pt_i * torch.cos(phi_i)
    py_i = pt_i * torch.sin(phi_i)
    pz_i = pt_i * torch.sinh(eta_i)
    E_i = torch.sqrt(px_i**2 + py_i**2 + pz_i**2 + m_i.clamp_min(0.0) ** 2)

    px_j = pt_j * torch.cos(phi_j)
    py_j = pt_j * torch.sin(phi_j)
    pz_j = pt_j * torch.sinh(eta_j)
    E_j = torch.sqrt(px_j**2 + py_j**2 + pz_j**2 + m_j.clamp_min(0.0) ** 2)

    m2 = (E_i + E_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2
    return torch.sqrt(m2.clamp_min(0.0))


def _build_edges(
    pt: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    mass: torch.Tensor,
    edge_feature_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fully connected directed edges and selected edge features."""
    if edge_feature_names is None:
        edge_feature_names = EDGE_FEATURE_NAMES

    p = pt.shape[0]
    eps = 1e-6

    rows = torch.arange(p).repeat_interleave(p - 1)
    cols = torch.cat([
        torch.cat([torch.arange(p)[:i], torch.arange(p)[i + 1:]])
        for i in range(p)
    ])
    edge_index = torch.stack([rows, cols], dim=0)

    src, dst = rows, cols
    deta = eta[src] - eta[dst]
    eta_product = eta[src] * eta[dst]
    dphi = _delta_phi_torch(phi[src], phi[dst])
    dR = torch.sqrt(deta**2 + dphi**2)

    m_ij = _inv_mass_pair_torch(
        pt[src], eta[src], phi[src], mass[src],
        pt[dst], eta[dst], phi[dst], mass[dst],
    )

    values = {
        "delta_eta": deta,
        "eta_product": eta_product,
        "sin_dphi": torch.sin(dphi),
        "cos_dphi": torch.cos(dphi),
        "delta_r": dR,
        "asinh_m_ij": torch.asinh(m_ij),
        "log_pt_ratio": torch.log(pt[src].clamp_min(eps) / pt[dst].clamp_min(eps)),
        "kt_dist": torch.log((torch.minimum(pt[src], pt[dst]) ** 2 * dR**2).clamp_min(eps)),
    }

    edge_attr = torch.stack([values[name] for name in edge_feature_names], dim=1)
    return edge_index, edge_attr


def _four_vector_dict(v) -> dict:
    """Cache the cylindrical + cartesian components of a ROOT TLorentzVector."""
    return {
        "pt": v.Pt(), "eta": v.Eta(), "phi": v.Phi(), "mass": v.M(),
        "E": v.E(), "px": v.Px(), "py": v.Py(), "pz": v.Pz(),
    }


def _assemble_graph(jets, hbb, htt, met_pt, met_phi, n_jets, cfg, y):
    """Build a PyG ``Data`` graph from already-parsed reco objects.

    Shared by the ROOT path (:func:`_build_reco_graph`) and the array/dict path
    (:func:`build_data_from_arrays`) so both produce byte-identical features.

    Parameters
    ----------
    jets : list of dict
        Each jet needs ``pt, eta, phi, mass, E, px, py, pz`` plus the four
        ``btagDeepFlav*`` scores and ``n_constituents``.
    hbb, htt : dict
        The summed Hbb / Htt four-vectors, each with the 8 keys of
        :func:`_four_vector_dict`.
    y : torch.Tensor or None
        Target tensor ``(1, nt)``; ``None`` for inference (no truth).
    """
    node_feature_names = _cfg_feature_names(cfg, "node")
    global_feature_names = _cfg_feature_names(cfg, "global")
    edge_feature_names = _cfg_feature_names(cfg, "edge")

    # Keep all reconstructed jets, but sort by pT for deterministic tensor order.
    jets = sorted(jets, key=lambda z: z["pt"], reverse=True)

    eta_vals = [j["eta"] for j in jets]
    eta_center = 0.5 * (max(eta_vals) + min(eta_vals))
    eta_width = max(max(eta_vals) - min(eta_vals), 1e-3)
    HT = sum(j["pt"] for j in jets)

    node_rows = []
    pt_arr, eta_arr, phi_arr, mass_arr = [], [], [], []

    for jet in jets:
        phi = jet["phi"]

        deta_hbb = jet["eta"] - hbb["eta"]
        dphi_hbb = _delta_phi_np(phi, hbb["phi"])
        dr_hbb = math.sqrt(deta_hbb**2 + dphi_hbb**2)

        deta_htt = jet["eta"] - htt["eta"]
        dphi_htt = _delta_phi_np(phi, htt["phi"])
        dr_htt = math.sqrt(deta_htt**2 + dphi_htt**2)

        centrality = 1.0 - 2.0 * abs((jet["eta"] - eta_center) / eta_width)

        node_values = {
            "log(pt+1)": math.log1p(jet["pt"]),
            "eta": jet["eta"],
            "sin(phi)": math.sin(phi),
            "cos(phi)": math.cos(phi),
            "log(mass+1)": math.log1p(max(jet["mass"], 0.0)),
            "log(E+1)": math.log1p(max(jet["E"], 0.0)),
            "sl1p(px)": float(_signed_log1p(np.float32(jet["px"]))),
            "sl1p(py)": float(_signed_log1p(np.float32(jet["py"]))),
            "sl1p(pz)": float(_signed_log1p(np.float32(jet["pz"]))),
            "centrality": centrality,
            "dEta(hbb)": deta_hbb,
            "sin(dPhi(hbb))": math.sin(dphi_hbb),
            "cos(dPhi(hbb))": math.cos(dphi_hbb),
            "dR(hbb)": dr_hbb,
            "dEta(htt)": deta_htt,
            "sin(dPhi(htt))": math.sin(dphi_htt),
            "cos(dPhi(htt))": math.cos(dphi_htt),
            "dR(htt)": dr_htt,
            "btagDeepFlavB": jet["btagDeepFlavB"],
            "btagDeepFlavCvB": jet["btagDeepFlavCvB"],
            "btagDeepFlavCvL": jet["btagDeepFlavCvL"],
            "btagDeepFlavQG": jet["btagDeepFlavQG"],
            "n_constituents": jet["n_constituents"],
        }

        node_rows.append(np.asarray([node_values[name] for name in node_feature_names], dtype=np.float32))

        pt_arr.append(jet["pt"])
        eta_arr.append(jet["eta"])
        phi_arr.append(phi)
        mass_arr.append(jet["mass"])

    x = torch.tensor(np.stack(node_rows, axis=0), dtype=torch.float32)

    global_values = {
        "hbb_logpt": math.log1p(hbb["pt"]),
        "hbb_eta": hbb["eta"],
        "hbb_sinphi": math.sin(hbb["phi"]),
        "hbb_cosphi": math.cos(hbb["phi"]),
        "hbb_logmass": math.log1p(max(hbb["mass"], 0.0)),
        "hbb_logE": math.log1p(max(hbb["E"], 0.0)),
        "hbb_sl1p_px": float(_signed_log1p(np.float32(hbb["px"]))),
        "hbb_sl1p_py": float(_signed_log1p(np.float32(hbb["py"]))),
        "hbb_sl1p_pz": float(_signed_log1p(np.float32(hbb["pz"]))),
        "htt_logpt": math.log1p(htt["pt"]),
        "htt_eta": htt["eta"],
        "htt_sinphi": math.sin(htt["phi"]),
        "htt_cosphi": math.cos(htt["phi"]),
        "htt_logmass": math.log1p(max(htt["mass"], 0.0)),
        "htt_logE": math.log1p(max(htt["E"], 0.0)),
        "htt_sl1p_px": float(_signed_log1p(np.float32(htt["px"]))),
        "htt_sl1p_py": float(_signed_log1p(np.float32(htt["py"]))),
        "htt_sl1p_pz": float(_signed_log1p(np.float32(htt["pz"]))),
        "met_logpt": math.log1p(met_pt),
        "met_sinphi": math.sin(met_phi),
        "met_cosphi": math.cos(met_phi),
        "nVBFJet_raw": float(n_jets),
        "log(HT_raw)": math.log1p(HT),
    }

    u = torch.tensor(
        [[global_values[name] for name in global_feature_names]],
        dtype=torch.float32,
    )

    edge_index, edge_attr = _build_edges(
        torch.tensor(pt_arr, dtype=torch.float32),
        torch.tensor(eta_arr, dtype=torch.float32),
        torch.tensor(phi_arr, dtype=torch.float32),
        torch.tensor(mass_arr, dtype=torch.float32),
        edge_feature_names=edge_feature_names,
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)


def _build_reco_graph(ev, cfg, n_jets, targets):
    """Parse the reco branches of one ROOT event record and build its graph.

    ``ev`` is indexed by the *logical* branch names (``VBFJet_pt``, ``b1_pt``,
    ``met_phi``, …); a ``branch_map`` (see :class:`VBFJetRootDataset`) aliases a
    collaborator's actual branch names onto these before this point.
    """
    hbb = _four_vector_dict(
        make_lorentzvector_ptetaphim(float(ev["b1_pt"]), float(ev["b1_eta"]), float(ev["b1_phi"]), float(ev["b1_mass"]))
        + make_lorentzvector_ptetaphim(float(ev["b2_pt"]), float(ev["b2_eta"]), float(ev["b2_phi"]), float(ev["b2_mass"]))
    )
    htt = _four_vector_dict(
        make_lorentzvector_ptetaphim(float(ev["tau1_pt"]), float(ev["tau1_eta"]), float(ev["tau1_phi"]), float(ev["tau1_mass"]))
        + make_lorentzvector_ptetaphim(float(ev["tau2_pt"]), float(ev["tau2_eta"]), float(ev["tau2_phi"]), float(ev["tau2_mass"]))
    )
    met_pt = float(ev["met_pt"])
    met_phi = float(ev["met_phi"])

    jets = []
    for j in range(n_jets):
        pt_j = float(ev["VBFJet_pt"][j])
        eta_j = float(ev["VBFJet_eta"][j])
        phi_j = float(ev["VBFJet_phi"][j])
        mass_j = float(ev["VBFJet_mass"][j])
        E_j, px_j, py_j, pz_j = _p4_components(pt_j, eta_j, phi_j, mass_j)

        jets.append({
            "pt": pt_j, "eta": eta_j, "phi": phi_j, "mass": mass_j,
            "E": E_j, "px": px_j, "py": py_j, "pz": pz_j,
            "btagDeepFlavB": float(ev["VBFJet_btagDeepFlavB"][j]),
            "btagDeepFlavCvB": float(ev["VBFJet_btagDeepFlavCvB"][j]),
            "btagDeepFlavCvL": float(ev["VBFJet_btagDeepFlavCvL"][j]),
            "btagDeepFlavQG": float(ev["VBFJet_btagDeepFlavQG"][j]),
            "n_constituents": float(ev["VBFJet_nConstituents"][j]),
        })

    return _assemble_graph(jets, hbb, htt, met_pt, met_phi, n_jets, cfg, targets)


# Node-feature keys a jet dict must provide in the array/dict API (see
# build_data_from_arrays). pt/eta/phi/mass are mandatory; the b-tags and
# n_constituents default to 0.0 when the user omits them.
_JET_BTAG_KEYS = ("btagDeepFlavB", "btagDeepFlavCvB", "btagDeepFlavCvL", "btagDeepFlavQG", "n_constituents")


def _as_ptetaphim(obj, what):
    """Coerce a (pt, eta, phi, mass) tuple/list/dict into a 4-float tuple."""
    if isinstance(obj, dict):
        try:
            return (float(obj["pt"]), float(obj["eta"]), float(obj["phi"]), float(obj["mass"]))
        except KeyError as exc:
            raise KeyError(f"{what} dict is missing key {exc}") from exc
    seq = list(obj)
    if len(seq) != 4:
        raise ValueError(f"{what} must be (pt, eta, phi, mass); got {obj!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def build_data_from_arrays(vbf_jets, hbb, htt, met, cfg, acceptance=None) -> Optional[Data]:
    """Build one inference graph directly from user-supplied physics objects.

    No ROOT file and no branch-name assumptions — the collaborator passes the
    quantities the model needs. Returns ``None`` when ``< 2`` VBF jets are given
    (same reco acceptance as the ROOT path). ``y`` is always unset (inference).

    Parameters
    ----------
    vbf_jets : sequence of dict
        One dict per reco VBF jet with ``pt, eta, phi, mass`` (required) and
        optionally the four ``btagDeepFlav*`` scores + ``n_constituents``
        (default 0.0 when omitted).
    hbb, htt : (pt, eta, phi, mass) tuple or dict
        The Hbb (b1+b2) and Htt (tau1+tau2) **summed** four-vectors.
    met : (pt, phi) tuple, or dict with keys ``pt``/``phi``.
    cfg : dict
        Resolved VBF config (for feature/target layout).
    acceptance : dict or None
        Event-level jet gate (see :func:`passes_acceptance`); ``None`` = off.
    """
    if vbf_jets is None or len(vbf_jets) < 2:
        return None
    if acceptance and not passes_acceptance(
        [jd["pt"] for jd in vbf_jets], [jd["eta"] for jd in vbf_jets], acceptance
    ):
        return None

    n_jets = len(vbf_jets)
    jets = []
    for jd in vbf_jets:
        pt_j = float(jd["pt"]); eta_j = float(jd["eta"])
        phi_j = float(jd["phi"]); mass_j = float(jd["mass"])
        E_j, px_j, py_j, pz_j = _p4_components(pt_j, eta_j, phi_j, mass_j)
        jet = {
            "pt": pt_j, "eta": eta_j, "phi": phi_j, "mass": mass_j,
            "E": E_j, "px": px_j, "py": py_j, "pz": pz_j,
        }
        for k in _JET_BTAG_KEYS:
            jet[k] = float(jd.get(k, 0.0))
        jets.append(jet)

    hbb_p4 = make_lorentzvector_ptetaphim(*_as_ptetaphim(hbb, "hbb"))
    htt_p4 = make_lorentzvector_ptetaphim(*_as_ptetaphim(htt, "htt"))

    if isinstance(met, dict):
        met_pt, met_phi = float(met["pt"]), float(met["phi"])
    else:
        met_pt, met_phi = float(met[0]), float(met[1])

    return _assemble_graph(
        jets, _four_vector_dict(hbb_p4), _four_vector_dict(htt_p4),
        met_pt, met_phi, n_jets, cfg, None,
    )


def _extract_event(
    ev,
    cfg: dict,
    eta_acceptance: float | None = None,
    build_targets: bool = True,
    acceptance: dict | None = None,
) -> Optional[Data]:
    """
    Convert one event record into a PyG Data object.

    Physics convention:
      - LHE VBF quark indices remain fixed: [4, 5]
      - no reco-jet pT threshold
      - no max-jet truncation
      - graph remains fully_connected_directed
    """
    cuts = cfg.get("dataset", {}).get("event_cuts", {})
    eta_acc = float(eta_acceptance if eta_acceptance is not None else cuts.get("eta_acceptance", 4.7))
    truth_min_pt = float(cuts.get("truth_min_pt", 50.0))
    require_hbb_valid = bool(cuts.get("require_hbb_valid", True))
    require_n_lhe = int(cuts.get("require_n_lhe", 6))

    n_jets = int(ev["nVBFJet"])
    if n_jets < 2:
        return None
    if acceptance and not passes_acceptance(
        ev["VBFJet_pt"][:n_jets], ev["VBFJet_eta"][:n_jets], acceptance
    ):
        return None

    if not build_targets:
        # Inference on data without generator-level truth: skip the truth cuts
        # and target construction, and build the graph with y unset (None).
        return _build_reco_graph(ev, cfg, n_jets, None)

    n_lhe = int(ev["nLHEPart"])
    if n_lhe != require_n_lhe:
        return None
    if require_hbb_valid and ev["Hbb_isValid"] != 1:
        return None

    # Fixed by analysis convention: VBF LHE quarks at indices 4 and 5.
    q = []
    for k in [4, 5]:
        pt = float(ev["LHEPart_pt"][k])
        eta = float(ev["LHEPart_eta"][k])
        phi = float(ev["LHEPart_phi"][k])
        mass = float(ev["LHEPart_mass"][k])

        if abs(eta) >= eta_acc or pt < truth_min_pt:
            return None

        q.append({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
        
    # Deterministic truth-quark ordering for individual p4 regression.
    # q[0] = forward / higher-eta VBF quark
    # q[1] = backward / lower-eta VBF quark
    #
    # This avoids a label-swap ambiguity between LHE indices [4, 5].
    q = sorted(q, key=lambda z: z["eta"], reverse=True)
    
    lhe_q1 = make_lorentzvector_ptetaphim(q[0]["pt"], q[0]["eta"], q[0]["phi"], q[0]["mass"])
    lhe_q2 = make_lorentzvector_ptetaphim(q[1]["pt"], q[1]["eta"], q[1]["phi"], q[1]["mass"])

    # raw_target_values = {
    #     "mjj": float((lhe_q1 + lhe_q2).M()),
    #     "deta": float(abs(q[0]["eta"] - q[1]["eta"])),
    #     "eta_prod": float(q[0]["eta"] * q[1]["eta"]),
    #     "ptsum": float(q[0]["pt"] + q[1]["pt"]),
    # }

    # y_values = []
    # for spec in target_specs(cfg):
    #     raw_key = spec.get("raw", spec["key"])
    #     if raw_key not in raw_target_values:
    #         raise KeyError(
    #             f"Target raw source '{raw_key}' is not implemented. "
    #             f"Available raw targets: {sorted(raw_target_values)}"
    #         )
    #     y_values.append(transform_scalar(raw_target_values[raw_key], spec.get("transform", "identity")))

    # targets = torch.tensor([y_values], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Generator-level VBF quark truth targets
    # ------------------------------------------------------------------
    # q[0], q[1] are the two LHE VBF quarks, fixed by current convention
    # to LHEPart indices [4, 5]. These are generator-level four-vectors,
    # not reconstructed jets.


    q1_E, q1_px, q1_py, q1_pz = _p4_components(
        q[0]["pt"],
        q[0]["eta"],
        q[0]["phi"],
        q[0]["mass"],
    )

    q2_E, q2_px, q2_py, q2_pz = _p4_components(
        q[1]["pt"],
        q[1]["eta"],
        q[1]["phi"],
        q[1]["mass"],
    )

    # Keep the old global VBF quantities available too, so YAML can still choose
    # mjj/deta/eta_prod/ptsum or any q1/q2 p4 component without editing code.

    raw_target_values = {
        # Old/global VBF targets
        "mjj": float((lhe_q1 + lhe_q2).M()),
        "deta": float(abs(q[0]["eta"] - q[1]["eta"])),
        "eta_prod": float(q[0]["eta"] * q[1]["eta"]),
        "ptsum": float(q[0]["pt"] + q[1]["pt"]),

        # New q1 p4 targets
        "q1_E": float(q1_E),
        "q1_px": float(q1_px),
        "q1_py": float(q1_py),
        "q1_pz": float(q1_pz),

        # New q2 p4 targets
        "q2_E": float(q2_E),
        "q2_px": float(q2_px),
        "q2_py": float(q2_py),
        "q2_pz": float(q2_pz),
    }

    y_values = []
    for spec in target_specs(cfg):
        raw_key = spec.get("raw", spec["key"])

        if raw_key not in raw_target_values:
            raise KeyError(
                f"Target raw source '{raw_key}' is not implemented. "
                f"Available raw targets: {sorted(raw_target_values)}"
            )

        y_values.append(
            transform_scalar(
                raw_target_values[raw_key],
                spec.get("transform", "identity"),
            )
        )

    targets = torch.tensor([y_values], dtype=torch.float32)


    return _build_reco_graph(ev, cfg, n_jets, targets)


# Branch names are organised into two groups by *logical* name. Collaborators
# whose ROOT files use different names supply a ``branch_map`` (logical -> actual)
# so we read their branches under these logical names (see VBFJetRootDataset).
#
#   _FEATURE_BRANCHES : the reco inputs the model actually needs (always required).
#   _TRUTH_BRANCHES   : generator-level branches used ONLY to build the regression
#                       target y and apply truth cuts. Required for TRAINING data
#                       but optional for INFERENCE (require_truth=False).
_FEATURE_BRANCHES = [
    "nVBFJet",
    "VBFJet_pt", "VBFJet_eta", "VBFJet_phi", "VBFJet_mass",
    "VBFJet_btagDeepFlavB", "VBFJet_btagDeepFlavCvB",
    "VBFJet_btagDeepFlavCvL", "VBFJet_btagDeepFlavQG",
    "VBFJet_nConstituents",
    "b1_pt", "b1_eta", "b1_phi", "b1_mass",
    "b2_pt", "b2_eta", "b2_phi", "b2_mass",
    "tau1_pt", "tau1_eta", "tau1_phi", "tau1_mass",
    "tau2_pt", "tau2_eta", "tau2_phi", "tau2_mass",
    "met_pt", "met_phi",
]

_TRUTH_BRANCHES = [
    "nLHEPart",
    "LHEPart_pt", "LHEPart_eta", "LHEPart_phi", "LHEPart_mass",
    "Hbb_isValid",
]

# Back-compat: the full set (used when truth is present, e.g. training data).
_REQUIRED_BRANCHES = _FEATURE_BRANCHES + _TRUTH_BRANCHES

# CMS event identity. REQUIRED here (unlike in the training builder, where it
# is optional): the fold-routed predictor picks the member for every event from
# ``event % n_folds`` -- the rule the training split used -- so an event without
# a number cannot be predicted. Never a model input.
_EVENT_ID_BRANCHES = ["run", "luminosityBlock", "event"]


def passes_acceptance(jet_pt, jet_eta, acceptance: dict | None) -> bool:
    """Event-level reco gate: at least ``min_jets`` jets with
    ``pt >= jet_min_pt`` and ``|eta| <= jet_max_abs_eta``.

    It only decides whether the event gets a prediction. It never removes a jet:
    the graph is built from ALL of the event's VBF jets, exactly as in training.
    ``None`` or an empty dict disables it.
    """
    if not acceptance:
        return True
    pt = np.asarray(jet_pt, dtype=np.float64)
    eta = np.asarray(jet_eta, dtype=np.float64)
    good = (pt >= float(acceptance["jet_min_pt"])) & (
        np.abs(eta) <= float(acceptance["jet_max_abs_eta"])
    )
    return int(np.count_nonzero(good)) >= int(acceptance["min_jets"])

class DataListDataset(Dataset):
    """Small Dataset wrapper around a pre-built list of PyG graphs."""

    def __init__(self, data_list, accepted_indices=None, parent=None):
        super().__init__()
        self._data = data_list

        if accepted_indices is None:
            self.accepted_indices = np.arange(len(self._data), dtype=np.int64)
        else:
            self.accepted_indices = np.asarray(accepted_indices, dtype=np.int64)

        # Preserve metadata from the parent full dataset.
        if parent is not None:
            self.config = getattr(parent, "config", None)
            self.config_hash = getattr(parent, "config_hash", None)
            self.node_feature_names = getattr(parent, "node_feature_names", NODE_FEATURE_NAMES)
            self.edge_feature_names = getattr(parent, "edge_feature_names", EDGE_FEATURE_NAMES)
            self.global_feature_names = getattr(parent, "global_feature_names", GLOBAL_FEATURE_NAMES)
            self.target_specs = getattr(parent, "target_specs", [])
            self.target_keys = getattr(parent, "target_keys", [])
            self.num_targets_cfg = getattr(parent, "num_targets_cfg", len(self.target_keys))
        else:
            self.config = None
            self.config_hash = None
            self.node_feature_names = NODE_FEATURE_NAMES
            self.edge_feature_names = EDGE_FEATURE_NAMES
            self.global_feature_names = GLOBAL_FEATURE_NAMES
            self.target_specs = []
            self.target_keys = []
            self.num_targets_cfg = NUM_TARGETS

    def len(self) -> int:
        return len(self._data)

    def get(self, idx: int) -> Data:
        return self._data[idx]

    @property
    def num_node_features(self) -> int:
        return int(self._data[0].x.shape[1])

    @property
    def num_edge_features(self) -> int:
        return int(self._data[0].edge_attr.shape[1])

    @property
    def num_global_features(self) -> int:
        return int(self._data[0].u.shape[1])


def subset_dataset(ds, indices):
    """Return a lightweight subset while preserving original accepted indices and metadata."""
    indices = np.asarray(indices, dtype=np.int64)
    parent_acc = getattr(ds, "accepted_indices", np.arange(len(ds._data), dtype=np.int64))

    return DataListDataset(
        [ds._data[i] for i in indices],
        accepted_indices=parent_acc[indices],
        parent=ds,
    )


class VBFJetRootDataset(Dataset):
    """Read ROOT events and keep accepted events as in-memory PyG graphs."""

    def __init__(
    self,
    root_files: Union[str, List[str]],
    tree_name: str | None = None,
    eta_acceptance: float = 4.7,
    max_events: Optional[int] = None,
    event_start: Optional[int] = None,
    event_stop: Optional[int] = None,
    verbose: bool = True,
    config: dict | str | None = None,
    branch_map: dict | None = None,
    require_truth: bool | None = None,
    acceptance: dict | None = None,
    ):
        super().__init__()

        if isinstance(config, str) or config is None:
            self.config = load_vbf_config(config)
        else:
            self.config = config

        # branch_map: logical name -> the collaborator's actual branch name. Only
        # the entries that differ are used as uproot aliases; unmapped logical
        # names must exist verbatim in the tree.
        self.branch_map = dict(branch_map or {})
        self.require_truth = require_truth
        # Event-level reco gate, applied in both truth and inference mode.
        self.acceptance = dict(acceptance) if acceptance else None

        self.config_hash = config_hash(self.config)
        self.node_feature_names = _cfg_feature_names(self.config, "node")
        self.edge_feature_names = _cfg_feature_names(self.config, "edge")
        self.global_feature_names = _cfg_feature_names(self.config, "global")
        self.target_specs = target_specs(self.config)
        self.target_keys = target_keys(self.config)
        self.num_targets_cfg = num_targets(self.config)

        # Precedence: an explicitly passed tree_name wins (so collaborators can
        # point at their own tree), else the config's dataset.tree_name, else "Events".
        tree_name = tree_name or self.config.get("dataset", {}).get("tree_name", None) or "Events"

        if isinstance(root_files, str):
            root_files = [root_files]

        resolved: List[str] = []
        for f in root_files:
            if str(f).startswith("root://"):
                resolved.append(str(f))
                continue

            p = Path(f)
            if "*" in str(p) or "?" in str(p):
                resolved.extend(sorted(str(x) for x in Path(p.parent).glob(p.name)))
            else:
                resolved.append(str(f))

        if not resolved:
            raise FileNotFoundError(f"No ROOT files found matching: {root_files}")

        raw_start = event_start if event_start is not None else 0
        raw_stop = event_stop

        def _actual(logical: str) -> str:
            """Map a logical branch name to the collaborator's actual name."""
            return self.branch_map.get(logical, logical)

        self._data: List[Data] = []
        # Original raw event index (0-based, sequential over the whole read) for
        # each ACCEPTED event. Lets callers realign per-raw-event arrays (e.g. the
        # old-selection in make_bdt_inputs) to the kept GNN rows after the dataset
        # drops events. For the single-file, event_start=0 case this equals the
        # entry index in the ROOT tree.
        raw_event_indices: List[int] = []
        # (run, luminosityBlock, event) of each ACCEPTED event, aligned with
        # raw_event_indices. event also rides on each graph as ``data.event``.
        event_ids: List[tuple] = []
        n_read = 0
        n_accepted = 0
        n_rejected = 0
        build_targets = True  # decided per-file from availability / require_truth

        for file_path in resolved:
            if max_events is not None and n_accepted >= max_events:
                break
            if verbose:
                print(f"  Reading {file_path} …", flush=True)

            with uproot.open(file_path) as f:
                if tree_name not in f:
                    raise KeyError(
                        f"Tree '{tree_name}' not found in {file_path}. "
                        f"Available keys: {list(f.keys())}"
                    )
                tree = f[tree_name]

                available = set(tree.keys())

                # Feature branches are always required (after mapping).
                missing_feat = [b for b in _FEATURE_BRANCHES if _actual(b) not in available]
                missing_ids = [b for b in _EVENT_ID_BRANCHES if _actual(b) not in available]
                if missing_ids:
                    raise KeyError(
                        f"Missing event-id branches in {file_path}: "
                        f"{[_actual(b) for b in missing_ids]}. The fold-routed "
                        f"predictor needs the CMS event number of every event "
                        f"(member = event % n_folds)."
                    )
                if missing_feat:
                    raise KeyError(
                        f"Missing required feature branches in {file_path}: "
                        f"{[ _actual(b) for b in missing_feat ]}\n"
                        f"(logical names: {missing_feat})\n"
                        f"Supply a branch_map={{logical: actual}} or check the tree. "
                        f"Available: {sorted(available)}"
                    )

                # Truth branches are optional (inference). Auto-detect unless the
                # caller forced require_truth.
                truth_present = all(_actual(b) in available for b in _TRUTH_BRANCHES)
                if self.require_truth is True and not truth_present:
                    missing_truth = [_actual(b) for b in _TRUTH_BRANCHES if _actual(b) not in available]
                    raise KeyError(
                        f"require_truth=True but truth branches are missing in "
                        f"{file_path}: {missing_truth}"
                    )
                build_targets = truth_present if self.require_truth is None else bool(self.require_truth)

                needed = (
                    list(_FEATURE_BRANCHES)
                    + list(_EVENT_ID_BRANCHES)
                    + (list(_TRUTH_BRANCHES) if build_targets else [])
                )
                aliases = {b: _actual(b) for b in needed if _actual(b) != b}

                arrays = tree.arrays(
                    needed,
                    aliases=aliases or None,
                    library="ak",
                    entry_start=raw_start,
                    entry_stop=raw_stop,
                )

            for i in range(len(arrays)):
                if max_events is not None and n_accepted >= max_events:
                    break

                n_read += 1
                data = _extract_event(
                    arrays[i],
                    cfg=self.config,
                    eta_acceptance=eta_acceptance,
                    build_targets=build_targets,
                    acceptance=self.acceptance,
                )

                if data is None:
                    n_rejected += 1
                else:
                    n_accepted += 1
                    ids = (
                        int(arrays[i]["run"]),
                        int(arrays[i]["luminosityBlock"]),
                        int(arrays[i]["event"]),
                    )
                    data.event = torch.tensor([ids[2]], dtype=torch.long)
                    self._data.append(data)
                    raw_event_indices.append(n_read - 1)
                    event_ids.append(ids)

        self.raw_event_indices = np.asarray(raw_event_indices, dtype=np.int64)
        self.event_ids = np.asarray(event_ids, dtype=np.int64).reshape(-1, 3)

        if verbose:
            window = f"[{raw_start}, {raw_stop if raw_stop is not None else 'EOF'})"
            print(
                f"  Events raw_window={window}  read={n_read:,}  "
                f"accepted={n_accepted:,}  rejected={n_rejected:,}  "
                f"({100 * n_rejected / max(n_read, 1):.1f}% cut)"
            )
            print(f"  Config hash: {self.config_hash}")
            print(f"  Node features  : {len(self.node_feature_names)}")
            print(f"  Edge features  : {len(self.edge_feature_names)}")
            print(f"  Global features: {len(self.global_feature_names)}")
            print(f"  Targets        : {self.target_keys}")

        if n_accepted == 0:
            raise RuntimeError("All events were rejected. Check ROOT branches and cuts.")

        self.accepted_indices = np.arange(len(self._data), dtype=np.int64)

    def len(self) -> int:
        return len(self._data)

    def get(self, idx: int) -> Data:
        return self._data[idx]

    @property
    def num_node_features(self) -> int:
        return int(self._data[0].x.shape[1])

    @property
    def num_edge_features(self) -> int:
        return int(self._data[0].edge_attr.shape[1])

    @property
    def num_global_features(self) -> int:
        return int(self._data[0].u.shape[1])

    @property
    def num_targets(self) -> int:
        return int(self._data[0].y.shape[1])

    def summary(self) -> str:
        return (
            f"VBFJetRootDataset | events={len(self):,} | "
            f"node={self.num_node_features} edge={self.num_edge_features} "
            f"global={self.num_global_features}"
        )
    
    def save(self, path: str) -> None:
        """Save the in-memory graph list and full metadata."""
        payload = {
            "data": self._data,
            "accepted_indices": getattr(
                self, "accepted_indices", np.arange(len(self._data), dtype=np.int64)
            ),
            "config": self.config,
            "config_hash": self.config_hash,
            "node_feature_names": self.node_feature_names,
            "edge_feature_names": self.edge_feature_names,
            "global_feature_names": self.global_feature_names,
            "target_specs": self.target_specs,
            "target_keys": self.target_keys,
            "num_targets": self.num_targets_cfg,
        }
        torch.save(payload, path)
        print(f"Dataset cached to {path}")

    @classmethod
    def load_cached(
        cls,
        path: str,
        config: dict | str | None = None,
        strict_config: bool = True,
    ) -> "VBFJetRootDataset":
        """Load a dataset previously written with save()."""
        obj = cls.__new__(cls)
        Dataset.__init__(obj)

        payload = torch.load(path, weights_only=False)

        if isinstance(payload, dict) and "data" in payload:
            obj._data = payload["data"]
            obj.accepted_indices = np.asarray(
                payload.get("accepted_indices", np.arange(len(obj._data), dtype=np.int64)),
                dtype=np.int64,
            )

            obj.config = payload.get("config", None)
            obj.config_hash = payload.get("config_hash", None)
            obj.node_feature_names = payload.get("node_feature_names", NODE_FEATURE_NAMES)
            obj.edge_feature_names = payload.get("edge_feature_names", EDGE_FEATURE_NAMES)
            obj.global_feature_names = payload.get("global_feature_names", GLOBAL_FEATURE_NAMES)
            obj.target_specs = payload.get("target_specs", [])
            obj.target_keys = payload.get("target_keys", [])
            obj.num_targets_cfg = int(payload.get("num_targets", len(obj.target_keys)))

            if config is not None:
                cfg = load_vbf_config(config) if isinstance(config, str) else config
                expected = config_hash(cfg)
                if obj.config_hash is not None and expected != obj.config_hash:
                    msg = (
                        f"Cached dataset config hash mismatch:\n"
                        f"  cache : {obj.config_hash}\n"
                        f"  current: {expected}\n"
                        f"Delete the cache or use the matching config."
                    )
                    if strict_config:
                        raise RuntimeError(msg)
                    print(f"[warn] {msg}")
        else:
            obj._data = payload
            obj.accepted_indices = np.arange(len(obj._data), dtype=np.int64)
            obj.config = load_vbf_config(config)
            obj.config_hash = config_hash(obj.config)
            obj.node_feature_names = _cfg_feature_names(obj.config, "node")
            obj.edge_feature_names = _cfg_feature_names(obj.config, "edge")
            obj.global_feature_names = _cfg_feature_names(obj.config, "global")
            obj.target_specs = target_specs(obj.config)
            obj.target_keys = target_keys(obj.config)
            obj.num_targets_cfg = num_targets(obj.config)

        print(f"Loaded cached dataset: {len(obj._data):,} events from {path}")
        return obj


if __name__ == "__main__":
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Smoke-test VBFJetRootDataset graph construction.",
    )
    parser.add_argument("root_file", help="Input ROOT file, for example anaTuple_0.root")
    parser.add_argument(
        "tree_name",
        nargs="?",
        default="Events",
        help="TTree name inside the ROOT file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "YAML config to use. If omitted, load_vbf_config(None) is used, "
            "which usually resolves to configs/vbfnet_config.yaml."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum number of accepted events to build for the smoke test.",
    )
    args = parser.parse_args()

    ds = VBFJetRootDataset(
        root_files=[args.root_file],
        tree_name=args.tree_name,
        max_events=args.max_events,
        verbose=True,
        config=args.config,
    )

    print(ds.summary())
    print(f"num_node_features   = {ds.num_node_features}")
    print(f"num_edge_features   = {ds.num_edge_features}")
    print(f"num_global_features = {ds.num_global_features}")
    print(f"num_targets         = {ds.num_targets}")
    print("target_keys         =", ds.target_keys)

    for i in range(len(ds)):
        d = ds[i]
        n_nodes = d.x.shape[0]
        n_edges_expected = n_nodes * (n_nodes - 1)

        assert d.x.shape == (n_nodes, ds.num_node_features)
        assert d.edge_index.shape == (2, n_edges_expected)
        assert d.edge_attr.shape == (n_edges_expected, ds.num_edge_features)
        assert d.u.shape == (1, ds.num_global_features)
        assert d.y.shape == (1, ds.num_targets)
        assert torch.isfinite(d.x).all()
        assert torch.isfinite(d.edge_attr).all()
        assert torch.isfinite(d.u).all()
        assert torch.isfinite(d.y).all()

        print(
            f"event {i:02d}: "
            f"nodes={n_nodes}, edges={d.edge_attr.shape[0]}, "
            f"x={tuple(d.x.shape)}, edge_attr={tuple(d.edge_attr.shape)}, "
            f"u={tuple(d.u.shape)}, y={tuple(d.y.shape)}"
        )
        print("  y =", d.y.numpy().round(4).tolist()[0])

    loader = DataLoader(ds, batch_size=min(10, len(ds)), shuffle=False)
    batch = next(iter(loader))

    assert batch.y.shape[1] == ds.num_targets
    assert batch.edge_attr.shape[1] == ds.num_edge_features
    assert batch.u.shape[1] == ds.num_global_features

    print("\nBatched test:")
    print(f"  batch.num_graphs  = {batch.num_graphs}")
    print(f"  batch.x           = {tuple(batch.x.shape)}")
    print(f"  batch.edge_index  = {tuple(batch.edge_index.shape)}")
    print(f"  batch.edge_attr   = {tuple(batch.edge_attr.shape)}")
    print(f"  batch.u           = {tuple(batch.u.shape)}")
    print(f"  batch.y           = {tuple(batch.y.shape)}")
    print("\nGraph construction test passed.")