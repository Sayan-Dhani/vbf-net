# Model Card — VBFNet_Ensemble 4.0.0

GitHub release v1.0.0 (branch `p4-regression`) = package version 4.0.0, built 2026-09-28.
Members from the `pyg_vbf_kfold_evmod` training campaign.

## What it does

For each reconstructed event, the model regresses the generator-level (LHE) four-vectors
of the two VBF quarks in HH→bbττ VBF production. It outputs 8 targets,
`q{1,2}_{E,px,py,pz}` in GeV, where q1 is the quark with the larger η. Each target has
four heads:
- a point estimate (`point`),
- the 16 %, 50 % and 84 % quantiles (`q16`, `q50`, `q84`).

The inputs are:
- **Graph nodes:** every VBF jet of the event, with 23 features each.
- **Edges:** jet pairs, with 8 features each.
- **Global features (23):** the H→bb and H→ττ candidates and MET.

The network is `PyGVBFGNN`: input encoders, 6 `EdgeConvWithAttr` layers (width 128;
sum/mean/max aggregation), mean+max pooling, and quantile + point heads. It has
6,401,538 parameters per member. Targets are learned in `signed_log1p` space and decoded
with `signed_expm1`.

## Members and routing

Five members, one per fold. Each event is predicted by **exactly one** member,
`k = event % 5` (the CMS `event` branch). Member k never saw events with `event % 5 == k`
during training, so its prediction for them is a true out-of-fold prediction. There is no
averaging across members.

| fold | epoch | train events | val events | selection score | val loss |
|---|---|---|---|---|---|
| 0 | 160 | 1,329,936 | 331,768 | 0.6109 | 0.3474 |
| 1 | 170 | 1,329,207 | 332,497 | 0.6114 | 0.3491 |
| 2 | 187 | 1,329,061 | 332,643 | 0.6093 | 0.3482 |
| 3 | 177 | 1,329,233 | 332,471 | 0.6128 | 0.3500 |
| 4 | 145 | 1,329,379 | 332,325 | 0.6079 | 0.3470 |

- **Source runs:** `pyg_vbf_kfold_evmod_fold{k}/best_model.pt` of the training campaign,
  selected on best selection score. The optimiser state is stripped from the shipped
  copies.
- **Hashes:** `config_hash 3fef998e66c87985`, `dataset_hash 78b7f45952655811`.
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

## Performance (out-of-fold)

These are OOF results over all 1,661,704 training-selection events. For each event, the
prediction comes from the member that did not train on it, which is exactly what the
release returns for these events. MAE uses the point head in GeV. coverage68 is the
fraction of events with the truth inside [q16, q84], before calibration.

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

**Release check.** The first training file (CV = 1, C2V = 0, C3 = 1) was rebuilt with the
release and compared event by event with the OOF predictions:
- 34,012 events, all present in the OOF file,
- 0 events routed to a different member,
- identical truth,
- prediction difference: median 2.4e-7, max 1.0e-4 in model space (float32 rounding;
  two different members differ by ~0.08).

## Quantile calibration

- **What is shipped:** one set of shift tables per member, `calibrations/fold{k}/`.
  Member k's tables are fitted on member k's OOF rows, which are exactly the events routed
  to it. They are applied only to those rows. Settings: shifts binned in the member's own
  q50, 15 linear bins, at least 20 events per bin. The point head is never calibrated.
- **Default:** calibration is off in the Python API
  (`VBFNetEnsemble(use_quantile_calibration=False)`) and on in
  `example_bdt_input/bdt_inputs_config.yaml`.
- **How it was decided.** The decision rule was fixed in writing before any calibration
  numbers were looked at. It was first set on 2026-09-23 for an earlier training
  campaign and applied here unchanged:
  - Each member's calibration was fitted on half of its OOF rows and scored on the other
    half.
  - Primary criterion: the median over members of the median over the 8 targets of
    |coverage68 − 0.68|.
- **Outcome:** **enable**.
  - The primary criterion went from 0.00320 raw to 0.00128 calibrated, and every member
    improved.
  - 7 of 8 targets improved. q2_pz got slightly worse (0.00140 → 0.00181). It was already
    at the statistical floor of the held-out half.
  - The pre-registered secondary criterion (pinball loss) was not evaluated, because the
    fitting script does not report it.
  - A post-hoc bin-count scan (5–30 bins) gave no reason to change from 15.
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
  extrapolation and has not been validated.
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

## Integrity

`python scripts/verify_release.py --strict` recomputes the sha256 of every member,
calibration file and source file, compares them with `RELEASE_MANIFEST.json`, and
loads every member.
