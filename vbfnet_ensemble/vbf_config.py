from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# SUPPORTED_TRANSFORMS = {"identity", "asinh", "log1p"}
# SUPPORTED_DECODES = {"identity", "sinh", "expm1"}

SUPPORTED_TRANSFORMS = {
    "identity",
    "asinh",
    "log1p",
    "signed_log1p",
    "sl1p",
}

SUPPORTED_DECODES = {
    "identity",
    "sinh",
    "expm1",
    "signed_expm1",
    "signed_expm1m",
    "sl1p_inv",
}

# Legal architecture values, kept in sync with pyg_vbf_gnn._activation /
# _norm_layer / VBFNet pool handling. Validated at config-load time so a typo
# fails fast with an actionable message instead of deep inside model build.
SUPPORTED_ACTIVATIONS = {"relu", "gelu", "silu", "elu"}
SUPPORTED_NORMS = {"layernorm", "batchnorm", "none", "identity", ""}
SUPPORTED_POOLS = {"mean", "max", "mean+max"}
SUPPORTED_AGGREGATIONS = {"sum", "mean", "max", "min", "mul", "std", "var"}

# Loss sections that accept an enabled_targets selector.
_LOSS_SECTIONS = ("pinball", "point", "variance", "coverage", "conditional_bias")


def load_vbf_config(path: str | None) -> dict[str, Any]:
    if path is None:
        path = "configs/vbfnet_config.yaml"

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML configuration. Install with: pip install pyyaml"
        ) from exc

    with open(Path(path).expanduser(), "r") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Empty config file: {path}")

    validate_vbf_config(cfg)
    return cfg


def validate_vbf_config(cfg: dict[str, Any]) -> None:
    targets = cfg.get("targets", [])
    if not targets:
        raise ValueError("Config must contain a non-empty 'targets' list.")

    keys = [t["key"] for t in targets]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate target keys in config: {keys}")

    for t in targets:
        tr = t.get("transform", "identity")
        de = t.get("decode", "identity")
        if tr not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Unsupported transform for target {t['key']}: {tr}")
        if de not in SUPPORTED_DECODES:
            raise ValueError(f"Unsupported decode for target {t['key']}: {de}")

    output = cfg.get("output", {})
    mode = output.get("mode", "both")
    if mode not in {"both", "quantile", "point"}:
        raise ValueError("output.mode must be one of: both, quantile, point")

    qs = output.get("quantiles", [])
    if mode in {"both", "quantile"}:
        if not qs:
            raise ValueError("Quantile output requested but output.quantiles is empty.")
        if sorted(qs) != list(qs):
            raise ValueError(f"Quantiles must be sorted increasingly: {qs}")
        if min(qs) <= 0.0 or max(qs) >= 1.0:
            raise ValueError(f"Quantiles must lie inside (0,1): {qs}")

    # ── features: each declared feature group must be a non-empty list ──────────
    features = cfg.get("features", {})
    if features:
        for group in ("node", "edge", "global"):
            if group in features:
                feats = features[group]
                if not isinstance(feats, (list, tuple)) or len(feats) == 0:
                    raise ValueError(
                        f"features.{group} must be a non-empty list; got {feats!r}."
                    )

    # ── model: pool / activation / norm / aggregation must be legal ─────────────
    model = cfg.get("model", {})
    if model:
        pool = str(model.get("pool", "mean+max")).lower().strip()
        if pool not in SUPPORTED_POOLS:
            raise ValueError(
                f"model.pool must be one of {sorted(SUPPORTED_POOLS)}; got {model.get('pool')!r}."
            )

        act = str(model.get("activation", "GELU")).lower().strip()
        if act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"model.activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}; "
                f"got {model.get('activation')!r}."
            )

        norm = model.get("norm", "LayerNorm")
        norm_key = "" if norm is None else str(norm).lower().strip()
        if norm_key not in SUPPORTED_NORMS:
            raise ValueError(
                f"model.norm must be one of LayerNorm, BatchNorm, none; got {norm!r}."
            )

        agg = model.get("aggregation", ["sum", "mean", "max"])
        if isinstance(agg, str):
            agg = [agg]
        if not isinstance(agg, (list, tuple)) or len(agg) == 0:
            raise ValueError(
                f"model.aggregation must be a non-empty list; got {model.get('aggregation')!r}."
            )
        unknown_agg = sorted({str(a).lower().strip() for a in agg} - SUPPORTED_AGGREGATIONS)
        if unknown_agg:
            raise ValueError(
                f"model.aggregation has unsupported entries {unknown_agg}; "
                f"allowed: {sorted(SUPPORTED_AGGREGATIONS)}."
            )

    # ── reweighting.target (when active) must be a configured target key ────────
    rw = cfg.get("reweighting", {})
    if rw and str(rw.get("mode", "none")).lower() != "none":
        rw_target = rw.get("target")
        if rw_target is not None and rw_target not in keys:
            raise ValueError(
                f"reweighting.target={rw_target!r} is not a configured target. "
                f"Available targets: {keys}."
            )

    # ── loss.*.enabled_targets must reference real target keys ──────────────────
    loss = cfg.get("loss", {})
    key_set = set(keys)
    for section in _LOSS_SECTIONS:
        sec = loss.get(section)
        if not isinstance(sec, dict):
            continue
        sel = sec.get("enabled_targets", "all")
        if sel is None or sel == "all":
            continue
        if isinstance(sel, str):
            sel = [sel]
        unknown = sorted(set(sel) - key_set)
        if unknown:
            raise ValueError(
                f"loss.{section}.enabled_targets references unknown targets {unknown}; "
                f"available: {keys}."
            )


def config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Config sections that actually change the *built dataset* (the cached Data
# objects). Everything else — model, output, loss, checkpoint, reweighting,
# training — affects only the network or the training loop, not the stored
# graphs/targets, so it must NOT invalidate the dataset cache. (Reweighting
# weights are attached after the cache is loaded, never baked into it.)
DATASET_HASH_SECTIONS = ("features", "targets", "dataset")


def dataset_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of only the config parts that determine dataset content.

    Used to key/validate the dataset cache so that changing training-only knobs
    (num_workers, epochs, batch_size, dropout, checkpoint weights, reweighting,
    …) does not force an expensive rebuild from ROOT. See
    :data:`DATASET_HASH_SECTIONS`.
    """
    subset = {k: cfg.get(k) for k in DATASET_HASH_SECTIONS}
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_resolved_config(cfg: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml
    except ImportError:
        with open(out_path.with_suffix(".json"), "w") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True)
        return

    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def target_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg["targets"])


def target_keys(cfg: dict[str, Any]) -> list[str]:
    return [t["key"] for t in target_specs(cfg)]


def target_index(cfg: dict[str, Any], key: str) -> int:
    keys = target_keys(cfg)
    if key not in keys:
        raise KeyError(f"Target '{key}' not in configured targets: {keys}")
    return keys.index(key)


def num_targets(cfg: dict[str, Any]) -> int:
    return len(target_specs(cfg))


def output_mode(cfg: dict[str, Any]) -> str:
    return cfg.get("output", {}).get("mode", "both")


def quantiles(cfg: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(q) for q in cfg.get("output", {}).get("quantiles", []))


def use_quantiles(cfg: dict[str, Any]) -> bool:
    return output_mode(cfg) in {"both", "quantile"}


def use_point(cfg: dict[str, Any]) -> bool:
    return output_mode(cfg) in {"both", "point"}


def num_heads(cfg: dict[str, Any]) -> int:
    n = 0
    if use_quantiles(cfg):
        n += len(quantiles(cfg))
    if use_point(cfg):
        n += 1
    if n <= 0:
        raise ValueError("No output heads are enabled.")
    return n


def head_names(cfg: dict[str, Any]) -> list[str]:
    names = []
    if use_quantiles(cfg):
        names.extend([f"q{int(round(q * 100)):02d}" for q in quantiles(cfg)])
    if use_point(cfg):
        names.append("point")
    return names


def target_weight_vector(
    cfg: dict[str, Any],
    field: str,
    default: float = 1.0,
    normalize: bool = True,
) -> list[float]:
    vals = [float(t.get(field, default)) for t in target_specs(cfg)]
    arr = np.asarray(vals, dtype=np.float64)
    if normalize:
        s = float(arr.sum())
        if s <= 0:
            raise ValueError(f"All {field} weights are zero or negative.")
        arr = arr / s
    return arr.astype(np.float32).tolist()


def enabled_target_mask(
    cfg: dict[str, Any],
    enabled_targets: str | list[str] | None,
) -> list[bool]:
    keys = target_keys(cfg)
    if enabled_targets is None or enabled_targets == "all":
        return [True] * len(keys)
    enabled = set(enabled_targets)
    unknown = sorted(enabled - set(keys))
    if unknown:
        raise KeyError(f"Unknown enabled targets {unknown}; available: {keys}")
    return [k in enabled for k in keys]


def signed_log1p_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1_np(z):
    z = np.asarray(z, dtype=np.float64)
    return np.sign(z) * np.expm1(np.abs(z))


def signed_log1p_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1_torch(z: torch.Tensor) -> torch.Tensor:
    return torch.sign(z) * torch.expm1(torch.abs(z))


def transform_scalar(x: float, rule: str) -> float:
    rule = str(rule)

    if rule == "identity":
        return float(x)

    if rule == "asinh":
        return float(np.arcsinh(max(float(x), 0.0)))

    if rule == "log1p":
        return float(np.log1p(max(float(x), 0.0)))

    if rule in {"signed_log1p", "sl1p"}:
        x = float(x)
        return float(np.sign(x) * np.log1p(abs(x)))

    raise ValueError(f"Unknown transform: {rule}")


def decode_np(arr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.array(arr, copy=True, dtype=np.float64)

    if arr.ndim not in (2, 3):
        raise ValueError(f"decode_np expects rank 2 or 3, got {arr.shape}")

    for ti, spec in enumerate(target_specs(cfg)[: arr.shape[1]]):
        rule = spec.get("decode", "identity")

        if arr.ndim == 2:
            z = arr[:, ti]

            if rule == "identity":
                out[:, ti] = z
            elif rule == "sinh":
                out[:, ti] = np.sinh(z)
            elif rule == "expm1":
                out[:, ti] = np.expm1(z)
            elif rule in {"signed_expm1", "signed_expm1m", "sl1p_inv"}:
                out[:, ti] = signed_expm1_np(z)
            else:
                raise ValueError(f"Unknown decode rule: {rule}")

        else:
            z = arr[:, ti, :]

            if rule == "identity":
                out[:, ti, :] = z
            elif rule == "sinh":
                out[:, ti, :] = np.sinh(z)
            elif rule == "expm1":
                out[:, ti, :] = np.expm1(z)
            elif rule in {"signed_expm1", "signed_expm1m", "sl1p_inv"}:
                out[:, ti, :] = signed_expm1_np(z)
            else:
                raise ValueError(f"Unknown decode rule: {rule}")

    return out


def decode_torch(z: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    rule = spec.get("decode", "identity")

    if rule == "identity":
        return z

    if rule == "sinh":
        return torch.sinh(z)

    if rule == "expm1":
        return torch.expm1(z)

    if rule in {"signed_expm1", "signed_expm1m", "sl1p_inv"}:
        return signed_expm1_torch(z)

    raise ValueError(f"Unknown decode rule: {rule}")


def apply_cli_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    """
    Keep CLI lightweight. Only override values that are already standard
    command-line knobs in the existing training script.
    """
    cfg = copy.deepcopy(cfg)

    # Data
    if getattr(args, "tree_name", None) is not None:
        cfg.setdefault("dataset", {})["tree_name"] = args.tree_name

    # Model overrides
    m = cfg.setdefault("model", {})
    for key in [
        "node_dim",
        "edge_dim",
        "global_dim",
        "n_layers",
        "dropout",
        "pool",
    ]:
        if hasattr(args, key) and getattr(args, key) is not None:
            m[key] = getattr(args, key)

    if hasattr(args, "mlp_hidden") and args.mlp_hidden is not None:
        m["head_hidden"] = list(args.mlp_hidden)

    # Training overrides. CLI arguments use default=None for these knobs, so
    # YAML values are preserved unless the user explicitly provides a CLI value.
    tr = cfg.setdefault("training", {})
    for key in [
        "lr",
        "weight_decay",
        "lr_schedule",
        "lr_min",
        "lr_warmup_epochs",
        "clip_grad",
        "epochs",
        "batch_size",
        "num_workers",
    ]:
        if hasattr(args, key) and getattr(args, key) is not None:
            tr[key] = getattr(args, key)

    # Optional target-weight overrides. These must match the configured target
    # count, avoiding the old hard-coded 4-target assumption.
    for attr, field in [
        ("target_weights", "loss_weight"),
        ("point_loss_weights", "point_loss_weight"),
    ]:
        vals = getattr(args, attr, None)
        if vals is None:
            continue
        targets = cfg.get("targets", [])
        if len(vals) != len(targets):
            keys = [t.get("key", f"target_{i}") for i, t in enumerate(targets)]
            raise ValueError(
                f"--{attr} expects {len(targets)} values, one per configured target "
                f"{keys}; got {len(vals)}"
            )
        for spec, value in zip(targets, vals):
            spec[field] = float(value)

    # Checkpoint-selection overrides. As above, YAML stays authoritative unless
    # the corresponding CLI flag is explicitly supplied.
    ckpt = cfg.setdefault("checkpoint", {})
    for attr, key in [
        ("ckpt_epsilon", "epsilon"),
        ("ckpt_alpha", "alpha"),
        ("ckpt_beta", "beta"),
        ("ckpt_gamma", "gamma"),
        ("ckpt_delta", "delta"),
    ]:
        if hasattr(args, attr) and getattr(args, attr) is not None:
            ckpt[key] = float(getattr(args, attr))

    # Reweighting overrides
    rw = cfg.setdefault("reweighting", {})
    if hasattr(args, "reweight_mode") and args.reweight_mode is not None:
        rw["mode"] = args.reweight_mode
    if hasattr(args, "weight_bins") and args.weight_bins is not None:
        rw["bins"] = args.weight_bins
    if hasattr(args, "weight_max_ratio") and args.weight_max_ratio is not None:
        rw["max_ratio"] = args.weight_max_ratio

    validate_vbf_config(cfg)
    return cfg
