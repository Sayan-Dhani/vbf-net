# Model Card — VBFNet_Ensemble 4.0.0+hl

GitHub release hl-v1.0.0 (branch `hl-regression`) = package version 4.0.0+hl, built
2026-10-05. Members from the `pyg_vbf_kfold_evmod_hl` training campaign.

This is the HL model. It shares the code, inputs, routing, acceptance gate and calibration
scheme of the p4 model (GitHub release v1.0.0, branch `p4-regression`), but its members
regress the high-level VBF observables directly instead of the two quark four-vectors.

## What it does

For each reconstructed event, the model regresses four generator-level (LHE) observables
of the two VBF quarks in HH→bbττ VBF production:

| target | quantity | unit | trained in | decoded with |
|---|---|---|---|---|
| `mjj` | m_qq, invariant mass of the two quarks | GeV | `asinh` | `sinh` |
| `deta` | \|Δη_qq\| | — | identity | identity |
| `eta_prod` | η_q1 · η_q2 | — | identity | identity |
| `ptsum` | pT_q1 + pT_q2 | GeV | `log1p` | `expm1` |

Each target has four heads:
- a point estimate (`point`),
- the 16 %, 50 % and 84 % quantiles (`q16`, `q50`, `q84`).

Unlike the p4 model, these quantiles are **quantiles of the observables themselves**, not
values computed from component-wise quantiles, so `[q16, q84]` is a real 68 % interval for
m_qq, |Δη|, η·η and the pT sum.

The inputs are the same as in the p4 model:
- **Graph nodes:** every VBF jet of the event, with 23 features each.
- **Edges:** jet pairs, with 8 features each.
- **Global features (23):** the H→bb and H→ττ candidates and MET.

The network is `PyGVBFGNN`: input encoders, 6 `EdgeConvWithAttr` layers (width 128;
sum/mean/max aggregation), mean+max pooling, and quantile + point heads. It has
6,399,471 parameters per member. The training loss is the p4 model's, with per-target
weights mjj 2, deta 2, eta_prod 1, ptsum 1.

## Members and routing

Five members, one per fold. Each event is predicted by **exactly one** member,
`k = event % 5` (the CMS `event` branch). Member k never saw events with `event % 5 == k`
during training, so its prediction for them is a true out-of-fold prediction. There is no
averaging across members.

| fold | epoch | train events | val events | selection score | val loss |
|---|---|---|---|---|---|
| 0 | 86 | 1,329,936 | 331,768 | 0.4130 | 0.2166 |
| 1 | 90 | 1,329,207 | 332,497 | 0.4038 | 0.2043 |
| 2 | 103 | 1,329,061 | 332,643 | 0.4160 | 0.2117 |
| 3 | 91 | 1,329,233 | 332,471 | 0.4190 | 0.2143 |
| 4 | 87 | 1,329,379 | 332,325 | 0.4098 | 0.2161 |

- **Source runs:** `pyg_vbf_kfold_evmod_hl_fold{k}/best_model.pt` of the training
  campaign, selected on best selection score with the same checkpoint-selection weights
  as the p4 model. The optimiser state is stripped from the shipped copies. Selection
  scores are not comparable with the p4 model's, because the targets differ.
- **Hashes:** `config_hash 9415f101ac7df702` (the members' resolved training
  configuration: the training YAML, hash `235d409f85466dfc`, plus the command-line
  settings for checkpoint selection, reweighting, batch size and epochs),
  `dataset_hash c0aa4eef9692074e`.
- **Where the full provenance lives:** file, weight and source-checkpoint sha256 values
  are in `RELEASE_MANIFEST.json`.

## Split: `event % 5`, no test set

All 1,661,704 events that pass the training selection are divided into five folds by
`event % 5`. Fold k is member k's validation set, and member k trains on the other four
folds. **No events are held out from every member**, so there is no independent test
sample.

Two consequences:
- The out-of-fold (OOF) set is the only unbiased per-member evaluation.
- An event's fold depends only on its event number. It does not change with file order,
  added files or the number of events read.

The split is the same as the p4 model's, event for event.

## Performance (out-of-fold)

These are OOF results over all 1,661,704 training-selection events. For each event, the
prediction comes from the member that did not train on it, which is exactly what the
release returns for these events. MAE uses the point head. coverage68 is the fraction of
events with the truth inside [q16, q84], before calibration.

| target | MAE | median AE | q50 MAE | coverage68 | bias / spread of the point head |
|---|---|---|---|---|---|
| `mjj` [GeV] | 171.45 | 101.97 | 171.72 | 0.687 | median rel. −0.26 %, σ68 rel. 14.5 % |
| `deta` | 0.2085 | 0.0819 | 0.2075 | 0.674 | median −0.008, σ68 0.144 |
| `eta_prod` | 0.4963 | 0.1860 | 0.4853 | 0.684 | median +0.018, σ68 0.338 |
| `ptsum` [GeV] | 23.19 | 16.57 | 25.44 | 0.686 | median rel. +0.42 %, σ68 rel. 11.0 % |

Fold-to-fold spread of the MAE (std over the 5 members): mjj 0.96 GeV, deta 0.0025,
eta_prod 0.0048, ptsum 0.17 GeV.

**Compared with the p4 model** on the same 1,661,704 events and the same split, where
these observables are computed from the regressed four-vectors:

| observable | MAE, p4 model (derived) | MAE, this model | 68 % coverage, p4 (component quantiles) | 68 % coverage, this model |
|---|---|---|---|---|
| m_qq [GeV] | 171.26 | 171.45 | 0.793 | 0.687 |
| \|Δη\| | 0.2171 | 0.2085 | 0.606 | 0.674 |
| pT sum [GeV] | 23.62 | 23.19 | 0.505 | 0.686 |

The point estimates are about equally accurate (|Δη| about 4 % and the pT sum about 2 %
better here, m_qq equal). The difference is the uncertainty: here the interval is a
quantile of the observable itself (calibrated, when calibration is on), whereas the p4
model has no valid per-event interval for these observables.

**Release check.** The first training file (CV = 1, C2V = 0, C3 = 1) was rebuilt with the
release and compared event by event with the OOF predictions:
- 34,012 events, all present in the OOF file,
- 0 events routed to a different member,
- identical truth,
- prediction difference: median 2.4e-7, max 6.8e-5 in model space (float32 rounding;
  no value above 1e-4).

## Quantile calibration

- **What is shipped:** one set of shift tables per member, `calibrations/fold{k}/`.
  Member k's tables are fitted on member k's OOF rows, which are exactly the events routed
  to it. They are applied only to those rows. Settings: shifts binned in the member's own
  q50, 15 linear bins, at least 20 events per bin. The shifts are in the target's own
  units (GeV for mjj and ptsum, none for deta and eta_prod). The point head is never
  calibrated.
- **Default:** calibration is off in the Python API
  (`VBFNetEnsemble(use_quantile_calibration=False)`) and on in
  `example_bdt_input/bdt_inputs_config.yaml`.
- **How it was decided.** The decision rule was fixed in writing before any calibration
  numbers were computed. It is the rule used for the p4 model, applied here unchanged:
  - Each member's calibration was fitted on half of its OOF rows and scored on the other
    half.
  - Primary criterion: the median over members of the median over the 4 targets of
    |coverage68 − 0.68|.
- **Outcome:** **enable**.
  - The primary criterion went from 0.00678 raw to 0.00121 calibrated, and every member
    improved.
  - All 4 targets improved in the median over members (mjj 0.0106 → 0.0007,
    deta 0.0071 → 0.0012, eta_prod 0.0073 → 0.0015, ptsum 0.0042 → 0.0025); ptsum got
    worse in 1 of the 5 members.
  - The pre-registered secondary criterion (pinball loss) was not evaluated, because the
    fitting script does not report it.
  - The shipped tables are the refit on all OOF rows.

## Validity domain

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
  the model never saw them. On a VBF HH signal file processed with the deployment gate,
  only about a third of events fall inside the training selection. Performance there is an
  extrapolation and has not been validated. Note that the truth pT sum of every training
  event is ≥ 100 GeV.
- **Events with `Hbb_isValid == 0`.** The gate does not require it, so these events reach
  the model, but none were in training.
- **Anything other than VBF HH signal**, for example DY or tt̄ background, or data. The
  model was trained only on signal. It is applied to everything as a fixed function, but
  its accuracy there is unknown.
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
- **No four-vectors.** This model does not predict the quark four-vectors, so it cannot
  give η_q1, η_q2, single-quark pT or any observable other than the four above.

## Integrity

`python scripts/verify_release.py --strict` recomputes the sha256 of every member,
calibration file and source file, compares them with `RELEASE_MANIFEST.json`, and
loads every member.
