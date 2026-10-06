"""VBFNet_Ensemble — K k-fold VBF-Net regressors with fold-routed inference.

Reconstructs the generator-level VBF quark four-vectors from reco jets in the
HH->bbtautau VBF analysis. Each event is predicted by exactly one of the K
k-fold members, ``k = event % K`` -- the rule the training split used -- so on
the training samples every prediction is the out-of-fold one. See
:mod:`vbfnet_ensemble.routing`.

Drop-in replacement for the single-model release::

    -from trained_model_VBFNet import VBFNet
    +from vbfnet_ensemble import VBFNet

``VBFNet`` is an explicit alias for :class:`VBFNetEnsemble`. See README.md for
the keys that are new (``fold_id``, ``run``, ``lumi``, ``event``, ``route``).
"""

from .branch_map import check_branch_map, load_branch_map
from .calibration import apply_quantile_calibration
from .predictor import VBFNetEnsemble, build_model_from_ckpt
from .routing import (
    DEFAULT_ACCEPTANCE,
    ROUTE_RULE,
    quantile_crossing_rate,
    resolve_acceptance,
    route_folds,
)
from .transforms import (
    decode_predictions,
    decode_predictions_array,
    predictions_array_to_dict,
)
from .validate import EnsembleCompatibilityError, validate_members

__version__ = "4.0.0+hl"

#: Drop-in alias. The migration is meant to be a one-line import change, so this
#: is a documented part of the API, not a compatibility shim.
VBFNet = VBFNetEnsemble

__all__ = [
    "VBFNet",
    "VBFNetEnsemble",
    "EnsembleCompatibilityError",
    "DEFAULT_ACCEPTANCE",
    "ROUTE_RULE",
    "apply_quantile_calibration",
    "build_model_from_ckpt",
    "check_branch_map",
    "decode_predictions",
    "decode_predictions_array",
    "load_branch_map",
    "predictions_array_to_dict",
    "quantile_crossing_rate",
    "resolve_acceptance",
    "route_folds",
    "validate_members",
    "__version__",
]
