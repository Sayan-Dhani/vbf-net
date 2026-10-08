"""VBFNet_Ensemble — fold-routed k-fold VBF-Net regressors for HH->bbtautau VBF.

Two model sets ship in ``ensembles/``, each a k-fold ensemble with its own
calibration: ``p4`` regresses the VBF-quark four-vectors ``q{1,2}_{E,px,py,pz}``,
``hl`` regresses ``mjj``, ``deta``, ``eta_prod`` and ``ptsum`` directly. Each
event is predicted by exactly one member of each set, ``k = event % K`` -- the
rule the training split used -- so on the training samples every prediction is
the out-of-fold one. See :mod:`vbfnet_ensemble.routing`.

Ask for targets; the right sets load by themselves::

    from vbfnet_ensemble import VBFNet
    net = VBFNet(targets=["q1_E", "mjj"])      # p4 + hl; default: every set
    out = net.predict_root("signal.root")

:class:`VBFNet` (:mod:`.unified`) routes targets to sets and merges their
predictions; :class:`VBFNetEnsemble` (:mod:`.predictor`) runs one set. A plain
``mjj`` is always the regressed (hl) value; the p4-derived one is ``mjj_p4``.
"""

from .branch_map import check_branch_map, load_branch_map
from .calibration import apply_quantile_calibration
from .manifest import DEFAULT_ENSEMBLE, available_ensembles
from .predictor import VBFNetEnsemble, build_model_from_ckpt
from .routing import (
    DEFAULT_ACCEPTANCE,
    ROUTE_RULE,
    quantile_crossing_rate,
    resolve_acceptance,
    route_folds,
)
from .transforms import (
    P4_DERIVED_KEYS,
    P4_DERIVED_SUFFIX,
    P4_TARGET_KEYS,
    decode_predictions,
    decode_predictions_array,
    predictions_array_to_dict,
)
from .unified import VBFNet, resolve_targets, target_catalogue
from .validate import EnsembleCompatibilityError, validate_members

__version__ = "5.0.0"

__all__ = [
    "VBFNet",
    "VBFNetEnsemble",
    "EnsembleCompatibilityError",
    "DEFAULT_ACCEPTANCE",
    "DEFAULT_ENSEMBLE",
    "P4_DERIVED_KEYS",
    "P4_DERIVED_SUFFIX",
    "P4_TARGET_KEYS",
    "ROUTE_RULE",
    "apply_quantile_calibration",
    "available_ensembles",
    "build_model_from_ckpt",
    "check_branch_map",
    "decode_predictions",
    "decode_predictions_array",
    "load_branch_map",
    "predictions_array_to_dict",
    "quantile_crossing_rate",
    "resolve_acceptance",
    "resolve_targets",
    "route_folds",
    "target_catalogue",
    "validate_members",
    "__version__",
]
