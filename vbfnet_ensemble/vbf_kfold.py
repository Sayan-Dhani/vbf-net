"""K-fold cross-validation splits and fold-aware cache naming.

Single source of truth shared by ``pyg_vbf_train.py`` (which builds the splits
and writes the per-split dataset caches) and ``eval/io.py`` (which has to find
those caches again). Both sides MUST go through these helpers: the per-split
cache filename is the one place where a fold can leak into another fold's
numbers, because the cache is preferred over re-subsetting the full dataset.

Deliberately numpy-only — no torch, no ROOT, no config import — so it is cheap
to import and cheap to test.

Fold convention
---------------
There is ONE split mode and it uses no randomness: every accepted event goes to
fold ``event % n_folds``, using the CMS ``event`` branch. Fold ``fold_id`` is
validation, the other ``n_folds - 1`` folds are train. There is no held-out
test set -- every event is in exactly one fold, so the K validation sets
together cover the whole dataset (the out-of-fold set, ``aggregate_oof.py``).

An event's fold is a property of the event itself, so it lands in the same fold
on every rebuild, whatever the file order, file set, cuts, ``--max_events`` or
``--build_workers``.

This replaces two older schemes, whose split files and per-split caches are NOT
interchangeable with these ones:

* ``random``        -- seeded permutation, 10 % test carved out first;
* ``deterministic`` -- test = first 10 % raw entries of each file, the rest
  ``event % K`` (per-split caches tagged ``_det``).

The per-split caches of this scheme carry an ``_evmod`` tag, so they can never
be confused with a ``_det`` cache of the same fold id sitting in a shared
``--cache_dir`` (same ``dataset_hash``, different events).

The K folds themselves (the events with ``event % K == k``) are exactly the K
validation sets, so the per-fold caches written once at cache-build time
(``VBFJetRootDataset.save_fold_caches``) use the ``val`` cache names:
``val_<hash>_evmod_fold<k>of<K>_dataset.pt``.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "SPLIT_MODE",
    "validate_fold_args",
    "assign_folds",
    "make_fold_indices",
    "fold_tag",
    "split_cache_filename",
    "parse_split_cache_name",
]

# Recorded as ``split_mode`` in every split .npz written by this code. Split
# files carrying any other value were made by a retired scheme (see above).
SPLIT_MODE = "event_mod"

# Filename tag of this scheme's per-split caches.
_MODE_TAG = "_evmod"

# ``full_<hash>_dataset.pt`` or ``val_<hash>_evmod_fold2of5_dataset.pt``
_CACHE_RE = re.compile(
    r"^(?P<split>full|train|val|test)_(?P<stem>.+?)(?:_evmod)?"
    r"(?:_fold(?P<fold>\d+)of(?P<nfolds>\d+))?_dataset\.pt$"
)


def validate_fold_args(n_folds, fold_id) -> tuple[int, int]:
    """Check a ``(--n_folds, --fold_id)`` pair, returning them as ints.

    Both must be given together: a fold id without a fold count has no meaning,
    and a fold count without an id does not say which fold to train.
    """
    if (n_folds is None) != (fold_id is None):
        raise ValueError(
            "--n_folds and --fold_id must be given together "
            f"(got n_folds={n_folds}, fold_id={fold_id})."
        )
    n_folds = int(n_folds)
    fold_id = int(fold_id)
    if n_folds < 2:
        raise ValueError(f"--n_folds must be >= 2, got {n_folds}.")
    if not (0 <= fold_id < n_folds):
        raise ValueError(
            f"--fold_id must be in [0, {n_folds}), got {fold_id}. Folds are 0-based."
        )
    return n_folds, fold_id


def assign_folds(event, n_folds: int) -> np.ndarray:
    """Fold number of every event: ``event % n_folds``, as an int64 array.

    ``event[i]`` is the CMS ``event`` number of accepted event ``i`` (aligned
    with the dataset). The single place the fold rule is written down; both
    :func:`make_fold_indices` and the per-fold caches go through it.
    """
    n_folds, _ = validate_fold_args(n_folds, 0)

    event = np.asarray(event, dtype=np.int64)
    if len(event) == 0:
        raise ValueError("No events to split.")
    if np.any(event < 0):
        raise ValueError(
            f"{int(np.sum(event < 0)):,} event(s) have no CMS event number "
            f"(read as -1: the file lacks run/luminosityBlock/event). The "
            f"fold split needs it for every event."
        )
    return event % n_folds


def make_fold_indices(
    event,
    n_folds: int,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx)`` for one fold, with no randomness at all.

    Event ``i`` is validation if ``event[i] % n_folds == fold_id``, train
    otherwise (see :func:`assign_folds`). Every event is in exactly one of the
    two, so the union over folds of ``val_idx`` is ``arange(len(event))``.
    Indices come back sorted.
    """
    n_folds, fold_id = validate_fold_args(n_folds, fold_id)

    is_val = assign_folds(event, n_folds) == fold_id
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)

    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError(
            f"Fold {fold_id}/{n_folds} is empty "
            f"(train={len(train_idx)}, val={len(val_idx)})."
        )

    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def fold_tag(fold_id, n_folds) -> str:
    """Filename fragment identifying a fold, e.g. ``_fold2of5``."""
    n_folds, fold_id = validate_fold_args(n_folds, fold_id)
    return f"_fold{fold_id}of{n_folds}"


def split_cache_filename(
    split_name: str,
    hash_stem: str,
    fold_id,
    n_folds,
) -> str:
    """Basename of a per-split dataset cache: ``<split>_<hash>_evmod_fold<k>of<K>_dataset.pt``.

    The fold is stamped into the name so that concurrent or sequential folds
    sharing one ``--cache_dir`` cannot overwrite or silently reuse each
    other's caches.
    """
    return f"{split_name}_{hash_stem}{_MODE_TAG}{fold_tag(fold_id, n_folds)}_dataset.pt"


def parse_split_cache_name(basename: str):
    """Inverse of :func:`split_cache_filename` (also accepts ``full_<hash>_dataset.pt``).

    Returns ``(split_name, hash_stem, fold_id, n_folds)`` with the fold entries
    None for an untagged cache name, or None if the name does not match.
    """
    m = _CACHE_RE.match(basename)
    if not m:
        return None
    fold = m.group("fold")
    nf = m.group("nfolds")
    return (
        m.group("split"),
        m.group("stem"),
        int(fold) if fold is not None else None,
        int(nf) if nf is not None else None,
    )
