# Model Card — VBFNet_Ensemble 5.0.0

GitHub release v2.0.0 = package version 5.0.0. It ships two model sets in one package:

| set | regresses | from release | members built | training campaign |
|---|---|---|---|---|
| `p4` | the two quark four-vectors `q{1,2}_{E,px,py,pz}` | v1.0.0 (package 4.0.0) | 2026-09-28 | `pyg_vbf_kfold_evmod` |
| `hl` | `mjj`, `deta`, `eta_prod`, `ptsum` directly | hl-v1.0.0 (package 4.0.0+hl) | 2026-10-05 | `pyg_vbf_kfold_evmod_hl` |

The weights and calibrations of both sets are unchanged from those releases, and so are
their predictions. On 2,000 signal and 2,000 DY events (CPU), every target, head and
calibrated value is bit-identical to v1.0.0 and hl-v1.0.0. Only the names of the four
observables derived from the p4 four-vectors changed: they are now `mjj_p4`, `deta_p4`,
`eta_prod_p4` and `ptsum_p4`, because the plain names belong to the hl set's regressed
values.

## What it does

For each reconstructed event, the models regress generator-level (LHE) properties of the
two VBF quarks in HH→bbττ VBF production. q1 is the quark with the larger η.

| target | quantity | unit | set | trained in | decoded with |
|---|---|---|---|---|---|
| `q1_E`, `q1_px`, `q1_py`, `q1_pz` | four-vector of q1 | GeV | p4 | `signed_log1p` | `signed_expm1` |
| `q2_E`, `q2_px`, `q2_py`, `q2_pz` | four-vector of q2 | GeV | p4 | `signed_log1p` | `signed_expm1` |
| `mjj` | m_qq, invariant mass of the two quarks | GeV | hl | `asinh` | `sinh` |
| `deta` | \|Δη_qq\| | — | hl | identity | identity |
| `eta_prod` | η_q1 · η_q2 | — | hl | identity | identity |
| `ptsum` | pT_q1 + pT_q2 | GeV | hl | `log1p` | `expm1` |

Each target has four heads:
- a point estimate (`point`),
- the 16 %, 50 % and 84 % quantiles (`q16`, `q50`, `q84`).

The hl quantiles are **quantiles of the observables themselves**, so `[q16, q84]` is a
real 68 % interval for m_qq, |Δη|, η·η and the pT sum. The p4 set also gives observables
derived from its four-vectors (`q{1,2}_{pt,eta,phi,mass}`, `mjj_p4`, `deta_p4`,
`eta_prod_p4`, `ptsum_p4`); their q16/q84 are computed from component quantiles and are
not quantiles of the observable.

Both sets have the same inputs:
- **Graph nodes:** every VBF jet of the event, with 23 features each.
- **Edges:** jet pairs, with 8 features each.
- **Global features (23):** the H→bb and H→ττ candidates and MET.

Both use the network `PyGVBFGNN`: input encoders, 6 `EdgeConvWithAttr` layers (width 128;
sum/mean/max aggregation), mean+max pooling, and quantile + point heads. A p4 member has
6,401,538 parameters, an hl member 6,399,471 (only the output layer differs). The hl
training loss is the p4 one with per-target weights mjj 2, deta 2, eta_prod 1, ptsum 1.

## Members and routing

Each set has five members, one per fold. Each event is predicted by **exactly one** member
of each set, `k = event % 5` (the CMS `event` branch). Member k never saw events with
`event % 5 == k` during training, so its prediction for them is a true out-of-fold
prediction. There is no averaging across members. With both sets loaded, the event goes
to member k of p4 and member k of hl.

**p4 set**

| fold | epoch | train events | val events | selection score | val loss |
|---|---|---|---|---|---|
| 0 | 160 | 1,329,936 | 331,768 | 0.6109 | 0.3474 |
| 1 | 170 | 1,329,207 | 332,497 | 0.6114 | 0.3491 |
| 2 | 187 | 1,329,061 | 332,643 | 0.6093 | 0.3482 |
| 3 | 177 | 1,329,233 | 332,471 | 0.6128 | 0.3500 |
| 4 | 145 | 1,329,379 | 332,325 | 0.6079 | 0.3470 |

- **Source runs:** `pyg_vbf_kfold_evmod_fold{k}/best_model.pt`, selected on best selection
  score.
- **Hashes:** `config_hash 3fef998e66c87985`, `dataset_hash 78b7f45952655811`.

**hl set**

| fold | epoch | train events | val events | selection score | val loss |
|---|---|---|---|---|---|
| 0 | 86 | 1,329,936 | 331,768 | 0.4130 | 0.2166 |
| 1 | 90 | 1,329,207 | 332,497 | 0.4038 | 0.2043 |
| 2 | 103 | 1,329,061 | 332,643 | 0.4160 | 0.2117 |
| 3 | 91 | 1,329,233 | 332,471 | 0.4190 | 0.2143 |
| 4 | 87 | 1,329,379 | 332,325 | 0.4098 | 0.2161 |

- **Source runs:** `pyg_vbf_kfold_evmod_hl_fold{k}/best_model.pt`, selected on best
  selection score with the same checkpoint-selection weights as the p4 set. Selection
  scores are not comparable between the sets, because the targets differ.
- **Hashes:** `config_hash 9415f101ac7df702` (the members' resolved training
  configuration: the training YAML, hash `235d409f85466dfc`, plus the command-line
  settings for checkpoint selection, reweighting, batch size and epochs),
  `dataset_hash c0aa4eef9692074e`.

For both sets the optimiser state is stripped from the shipped copies. File, weight and
source-checkpoint sha256 values are in `ensembles/<set>/RELEASE_MANIFEST.json`.

## Split: `event % 5`, no test set

All 1,661,704 events that pass the training selection are divided into five folds by
`event % 5`. Fold k is member k's validation set, and member k trains on the other four
folds. **No events are held out from every member**, so there is no independent test
sample. The split is the same for both sets, event for event.

Two consequences:
- The out-of-fold (OOF) set is the only unbiased per-member evaluation.
- An event's fold depends only on its event number. It does not change with file order,
  added files or the number of events read.

## Performance (out-of-fold)

These are OOF results over all 1,661,704 training-selection events. For each event, the
prediction comes from the member that did not train on it, which is exactly what the
release returns for these events. MAE uses the point head. coverage68 is the fraction of
events with the truth inside [q16, q84], before calibration.

**p4 set** (GeV)

| target | MAE | median AE | q50 MAE | coverage68 |
|---|---|---|---|---|
| q1_E  | 129.07 | 65.95 | 130.61 | 0.680 |
| q1_px | 14.26 | 7.93 | 14.29 | 0.681 |
| q1_py | 14.16 | 7.84 | 14.27 | 0.677 |
| q1_pz | 131.25 | 66.41 | 131.72 | 0.680 |
| q2_E  | 127.73 | 65.33 | 128.86 | 0.683 |
| q2_px | 14.12 | 7.88 | 14.20 | 0.692 |
| q2_py | 14.07 | 7.84 | 14.15 | 0.682 |
| q2_pz | 129.79 | 65.76 | 130.33 | 0.684 |

Over the 8 targets, the mean MAE is **71.81 GeV**. By fold it ranges from 71.54 to 72.19
(std 0.29).

**hl set**

| target | MAE | median AE | q50 MAE | coverage68 | bias / spread of the point head |
|---|---|---|---|---|---|
| `mjj` [GeV] | 171.45 | 101.97 | 171.72 | 0.687 | median rel. −0.26 %, σ68 rel. 14.5 % |
| `deta` | 0.2085 | 0.0819 | 0.2075 | 0.674 | median −0.008, σ68 0.144 |
| `eta_prod` | 0.4963 | 0.1860 | 0.4853 | 0.684 | median +0.018, σ68 0.338 |
| `ptsum` [GeV] | 23.19 | 16.57 | 25.44 | 0.686 | median rel. +0.42 %, σ68 rel. 11.0 % |

Fold-to-fold spread of the MAE (std over the 5 members): mjj 0.96 GeV, deta 0.0025,
eta_prod 0.0048, ptsum 0.17 GeV.

**hl compared with the p4-derived observables** on the same 1,661,704 events and the same
split:

| observable | MAE, p4 (derived) | MAE, hl | 68 % coverage, p4 (component quantiles) | 68 % coverage, hl |
|---|---|---|---|---|
| m_qq [GeV] | 171.26 | 171.45 | 0.793 | 0.687 |
| \|Δη\| | 0.2171 | 0.2085 | 0.606 | 0.674 |
| pT sum [GeV] | 23.62 | 23.19 | 0.505 | 0.686 |

The point estimates are about equally accurate (|Δη| about 4 % and the pT sum about 2 %
better from hl, m_qq equal). The difference is the uncertainty: the hl interval is a
quantile of the observable itself (calibrated, when calibration is on), whereas the p4
set has no valid per-event interval for these observables. Use the hl values when you
need an uncertainty on m_qq, |Δη|, η·η or the pT sum; use p4 for anything that needs the
individual quarks.

**Release checks.** For each set, the first training file (CV = 1, C2V = 0, C3 = 1) was
rebuilt with its release and compared event by event with that set's OOF predictions:

| | events (all in the OOF file) | routed to a different member | truth | prediction difference (model space) |
|---|---|---|---|---|
| p4 | 34,012 | 0 | identical | median 2.4e-7, max 1.0e-4 |
| hl | 34,012 | 0 | identical | median 2.4e-7, max 6.8e-5 |

The differences are float32 rounding; two different members differ by ~0.08.

## Quantile calibration

- **What is shipped:** for each set, one set of shift tables per member,
  `ensembles/<set>/calibrations/fold{k}/`. Member k's tables are fitted on member k's OOF
  rows, which are exactly the events routed to it. They are applied only to those rows.
  Settings: shifts binned in the member's own q50, 15 linear bins, at least 20 events per
  bin. The shifts are in the target's own units (GeV for the p4 components, mjj and
  ptsum; none for deta and eta_prod). The point head is never calibrated.
- **Default:** calibration is off in the Python API
  (`VBFNet(use_quantile_calibration=False)`) and on in
  `example_bdt_input/bdt_inputs_config.yaml`. The switch acts on every loaded set.
- **How it was decided.** The decision rule was fixed in writing before any calibration
  numbers were looked at. It was first set on 2026-09-23 for an earlier training campaign
  and applied to both sets unchanged:
  - Each member's calibration was fitted on half of its OOF rows and scored on the other
    half.
  - Primary criterion: the median over members of the median over the set's targets of
    |coverage68 − 0.68|.
- **Outcome, p4:** **enable**.
  - The primary criterion went from 0.00320 raw to 0.00128 calibrated, and every member
    improved.
  - 7 of 8 targets improved. q2_pz got slightly worse (0.00140 → 0.00181). It was already
    at the statistical floor of the held-out half.
  - A post-hoc bin-count scan (5–30 bins) gave no reason to change from 15.
- **Outcome, hl:** **enable**.
  - The primary criterion went from 0.00678 raw to 0.00121 calibrated, and every member
    improved.
  - All 4 targets improved in the median over members (mjj 0.0106 → 0.0007,
    deta 0.0071 → 0.0012, eta_prod 0.0073 → 0.0015, ptsum 0.0042 → 0.0025); ptsum got
    worse in 1 of the 5 members.
- For both sets the pre-registered secondary criterion (pinball loss) was not evaluated,
  because the fitting script does not report it, and the shipped tables are the refit on
  all OOF rows.

## Validity domain

The two sets were trained on the same events with the same selection.

**Training selection** (generator level). An event was used only if all of these hold:
- ≥ 2 VBF jets,
- `nLHEPart == 6`,
- `Hbb_isValid == 1`,
- both LHE VBF quarks (LHE indices 4 and 5) have **pT ≥ 50 GeV** and |η| < 4.7.

**Training samples:**
- 53 anaTuple files, production v2608, Run3_2022EE only.
- VBF HH→bbττ signal only, at 10 (CV, C2V, C3) coupling points.
- 1,661,704 events pass this selection.

**Deployment gate** (reconstruction level). A prediction is made only for events with at
least 2 VBF jets that have pT ≥ 50 GeV and |η| ≤ 4.7. The gate decides which events get
a prediction. The graph is still built from all of the event's jets. No generator
information is used.

The gate is **not** the training selection. Predictions fall outside the validity domain
for:
- **Events whose LHE quarks have pT < 50 GeV.** The gate lets such events through, but
  the models never saw them. On a VBF HH signal file processed with the deployment gate,
  only about a third of events fall inside the training selection. Performance there is an
  extrapolation and has not been validated. Note that the truth pT sum of every training
  event is ≥ 100 GeV.
- **Events with `Hbb_isValid == 0`.** The gate does not require it, so these events reach
  the models, but none were in training.
- **Anything other than VBF HH signal**, for example DY or tt̄ background, or data. The
  models were trained only on signal. They are applied to everything as fixed functions,
  but their accuracy there is unknown.
- **Other eras, other ntuple versions and systematic-variation trees.** Only nominal
  Run3_2022EE v2608 was used in training.

## Requirements and caveats

- **Event numbers.** Input trees must have `run`, `luminosityBlock` and `event`. Files
  without them are refused, because routing needs `event`.
- **Joining output rows back to input events.** MC event numbers repeat across samples,
  since the coupling points share ranges. This does not affect routing, but any join of
  outputs back to inputs must also key on the file.
- **Cost of routing.** One member per event is less accurate than averaging all five. In
  exchange, every event gets an honest out-of-fold prediction, and signal and background
  are treated the same way.
- **The hl set gives no four-vectors**, so it cannot give η_q1, η_q2, single-quark pT or
  any observable other than its four; those come from the p4 set.

## Integrity

`python scripts/verify_release.py --strict` recomputes the sha256 of every shipped source
file and of each set manifest, and through the set manifests of every member and
calibration file. It compares them with `RELEASE_MANIFEST.json` and
`ensembles/<set>/RELEASE_MANIFEST.json`, and loads every member.
