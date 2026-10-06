"""Compatibility validation for a set of ensemble members.

The single most dangerous failure mode of an ensemble is that it *works* on
incompatible members: shapes line up, ``load_state_dict(strict=True)`` passes,
and the reduction silently launders garbage into a plausible-looking number. The
checks here exist to make that impossible, and they are a **pure function over
metadata dicts** so they can be exercised without torch, without checkpoints and
without fakery — the same design as the fold-aggregation check used in training,
which refuses to average folds that are not a clean partition.

The same function runs when a release is built and at load
time (``VBFNetEnsemble.__init__``), so the two can never disagree about what
"compatible" means.
"""

from __future__ import annotations

from typing import Sequence

from .transforms import _decode_rule_for_spec


class EnsembleCompatibilityError(ValueError):
    """Raised when a set of members cannot be ensembled."""


#: Keys that must be EXACTLY equal across every member. There is deliberately no
#: flag that downgrades any of these to a warning.
HARD_KEYS: tuple[str, ...] = (
    "config_hash",
    "target_keys",
    "head_names",
    "quantiles",
    "output_mode",
    "num_targets",
    "num_heads",
    "num_node_features",
    "num_edge_features",
    "num_global_features",
    "node_feature_names",
    "edge_feature_names",
    "global_feature_names",
)

#: Keys that must be DISTINCT across members.
DISTINCT_KEYS: tuple[str, ...] = ("fold_id", "state_dict_sha256")

#: Keys that legitimately differ and are only reported.
SOFT_KEYS: tuple[str, ...] = ("epoch", "val_loss", "selection_score")


def _member_label(meta: dict, index: int) -> str:
    fold = meta.get("fold_id", None)
    fold = f"fold{fold}" if fold is not None else f"member[{index}]"
    src = meta.get("file", meta.get("source_file", ""))
    return f"{fold} ({src})" if src else fold


def _normalise(value):
    """Make a metadata value comparable and hashable (lists -> tuples)."""
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalise(v)) for k, v in value.items()))
    return value


def _fold_id_of(meta: dict, index: int):
    fid = meta.get("fold_id", None)
    if fid is None:
        args = meta.get("args", {}) or {}
        fid = args.get("fold_id", None)
    return fid if fid is not None else index


def decode_rules(meta: dict) -> tuple[str, ...]:
    """The per-target decode rules a member will actually be decoded with.

    Compared instead of the raw ``target_specs`` because two specs can differ in
    cosmetic fields (``latex``, ``plain``) while decoding identically — and,
    far more importantly, can *agree* on ``transform`` while disagreeing on an
    explicit ``decode`` override, which is what actually gets applied.
    """
    specs = meta.get("target_specs", None) or []
    return tuple(_decode_rule_for_spec(dict(s)) for s in specs)


def validate_members(metas: Sequence[dict], *, strict: bool = True) -> list[str]:
    """Return a list of compatibility problems. An empty list means compatible.

    Parameters
    ----------
    metas:
        One metadata dict per member — a checkpoint with the heavy ``model`` /
        ``optimiser`` entries removed. ``fold_id`` and ``state_dict_sha256`` are
        expected to have been added by the caller.
    strict:
        When False, the SOFT keys are still reported but nothing else changes.
        No value of this flag can clear a HARD or DISTINCT problem.
    """
    problems: list[str] = []

    if not metas:
        return ["No members supplied."]
    if len(metas) == 1:
        return problems  # a 1-member "ensemble" is degenerate but not incompatible

    ref, ref_label = metas[0], _member_label(metas[0], 0)

    for key in HARD_KEYS:
        ref_val = _normalise(ref.get(key, None))
        for i, meta in enumerate(metas[1:], start=1):
            val = _normalise(meta.get(key, None))
            if val != ref_val:
                problems.append(
                    f"{key} differs: {ref_label} has {ref.get(key, None)!r}, "
                    f"{_member_label(meta, i)} has {meta.get(key, None)!r}"
                )

    ref_rules = decode_rules(ref)
    for i, meta in enumerate(metas[1:], start=1):
        rules = decode_rules(meta)
        if rules != ref_rules:
            problems.append(
                f"target decode rules differ: {ref_label} has {list(ref_rules)}, "
                f"{_member_label(meta, i)} has {list(rules)}"
            )

    for key in DISTINCT_KEYS:
        seen: dict = {}
        for i, meta in enumerate(metas):
            val = _normalise(meta.get(key, None))
            if val is None:
                continue
            if val in seen:
                problems.append(
                    f"{key} is duplicated: {_member_label(metas[seen[val]], seen[val])} "
                    f"and {_member_label(meta, i)} both have {meta.get(key)!r}"
                    + (
                        " — the same weights would get a double vote in the reduction."
                        if key == "state_dict_sha256"
                        else ""
                    )
                )
            else:
                seen[val] = i

    # Partition provenance: these live in `args` on a real training checkpoint.
    # split_mode is absent on event % K ("event_mod") runs, whose CLI no longer
    # has the flag; split_seed is absent there too.
    for key in ("n_folds", "split_seed", "split_mode"):
        vals = {
            i: (meta.get(key, None) or (meta.get("args", {}) or {}).get(key, None))
            for i, meta in enumerate(metas)
        }
        distinct = {v for v in vals.values() if v is not None}
        if len(distinct) > 1:
            problems.append(
                f"{key} differs across members ({sorted(distinct)}) — the members did "
                "not hold out the same events, so some of them trained on data the "
                "others treat as held out."
            )

    # Routing sends event % K == k to member k, which is only honest if member
    # k never trained on those events. A random split gives no such guarantee.
    for i, meta in enumerate(metas):
        mode = meta.get("split_mode", None) or (meta.get("args", {}) or {}).get("split_mode", None)
        if mode == "random":
            problems.append(
                f"{_member_label(meta, i)} was trained on a RANDOM split: its held-out "
                "events are not event % K, so fold routing would send it events it "
                "trained on."
            )

    return problems


def assert_members_compatible(metas: Sequence[dict], *, strict: bool = True) -> None:
    """Raise :class:`EnsembleCompatibilityError` listing *every* problem found."""
    problems = validate_members(metas, strict=strict)
    if problems:
        bullet = "\n  - ".join(problems)
        raise EnsembleCompatibilityError(
            f"Refusing to ensemble {len(metas)} incompatible members:\n  - {bullet}\n"
            "Routing events across these would produce plausible-looking but "
            "meaningless predictions. Rebuild the release from a single k-fold run."
        )


def order_members(metas: Sequence[dict]) -> list[int]:
    """Indices that order members by ``fold_id``.

    Routing sends ``event % K == k`` to ``models[k]``, so the order IS the
    routing table: ordering by ``fold_id`` rather than by filename or glob order
    makes it independent of how the checkpoints happen to be named.
    """
    return sorted(range(len(metas)), key=lambda i: _fold_id_of(metas[i], i))
