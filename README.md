# VBF-Net — VBF-quark regression for HH→bbττ

VBF-Net is a graph neural network that regresses the **two VBF quarks** of HH→bbττ VBF
events from reconstructed jets, the H→bb and H→ττ candidates and MET. Two model sets ship
in this package. You ask for targets by name, and the set that regresses them is loaded
for you.

| target | quantity | unit | model set |
|---|---|---|---|
| `q1_E`, `q1_px`, `q1_py`, `q1_pz` | four-vector of q1, the quark with the larger (+ve) η | GeV | `p4` |
| `q2_E`, `q2_px`, `q2_py`, `q2_pz` | four-vector of q2 | GeV | `p4` |
| `mjj` | m_qq, invariant mass of the two quarks | GeV | `hl` |
| `deta` | \|Δη_qq\| | — | `hl` |
| `eta_prod` | η_q1 · η_q2 | — | `hl` |
| `ptsum` | pT_q1 + pT_q2 | GeV | `hl` |

Each target has a point estimate plus the 16 / 50 / 84 % quantiles. The `hl` quantiles are
**quantiles of the observable itself**, so `[q16, q84]` is a per-event 68 % interval for
m_qq, |Δη|, η₁·η₂ and the pT sum.

The `p4` set also gives observables **derived** from its predicted four-vectors:
`q{1,2}_{pt,eta,phi,mass}` and `mjj_p4`, `deta_p4`, `eta_prod_p4`, `ptsum_p4`.

**Name rule.** A plain name always means the set that regresses it: `mjj` is the `hl`
prediction, and the m_qq computed from the `p4` four-vectors is `mjj_p4`. A key never
changes meaning with which sets are loaded.

| | |
|---|---|
| Models | 2 sets × 5 k-fold members; each event is predicted by exactly one member of each set |
| Calibration | per-member quantile calibration for both sets included (optional) |
| Details | [MODEL_CARD.md](MODEL_CARD.md): training data, performance, validity domain |

## Install

The member checkpoints in `ensembles/<set>/models/` are stored with **git-lfs**. Without
it, a clone contains small pointer files instead of the weights.

```bash
git lfs install
git clone https://github.com/Sayan-Dhani/vbf-net.git
cd vbf-net
pip install -e ".[calibration]"
python scripts/verify_release.py --strict
```

**Packages:**
- **Required:** Python ≥ 3.9, numpy, torch, torch-geometric, uproot, awkward, PyYAML.
- **For calibration:** correctionlib, installed by the `[calibration]` extra.
- **For the BDT example only:** PyROOT.

**What `verify_release.py --strict` checks:** every source file against
`RELEASE_MANIFEST.json`, each set's manifest, and through it every checkpoint and
calibration file of that set; then it loads every checkpoint. It reports an un-fetched
git-lfs pointer; run `git lfs pull` to fix that.

## Quickstart

```python
from vbfnet_ensemble import VBFNet

net = VBFNet(use_quantile_calibration=True)  # every set: p4 + hl, 10 verified members
out = net.predict_root("signal.root", tree_name="Events", max_events=100)

out["pred_phys"]["q1_E"]["point"]  # p4: central value per event (GeV)
out["pred_phys_cal"]["q1_E"]["q84"]  # p4: calibrated 84 % quantile
out["pred_phys_cal"]["mjj"]["q84"]  # hl: calibrated 84 % quantile of m_qq
out["pred_phys"]["mjj_p4"]["point"]  # m_qq derived from the p4 four-vectors
out["ensemble_of"]["mjj"]  # "hl": which set produced a key
out["fold_id"]  # which member predicted each row (= event % 5, in both sets)
out["run"], out["lumi"], out["event"]  # CMS event id of each row
out["event_index"]  # ROOT entry of each row (single file)
```

### Choosing the targets

```python
VBFNet(targets=["q1_E", "q2_pz"])  # loads p4 only
VBFNet(targets=["mjj", "deta"])  # loads hl only
VBFNet(targets=["q1_E", "mjj"])  # loads both
VBFNet(targets="hl")  # a whole set, by name
VBFNet()  # the same as targets="all": every set
```

`targets` decides which sets load. Every target of a loaded set is returned, because a
set's network predicts all of its targets at once. A name that no set provides raises an
error listing the valid ones. `net.target_keys` lists what you get, in the order of the
`*_full` arrays: the p4 targets first (positions 0–7, as in v1.0.0), then the hl ones.

With both sets loaded, the event graphs are **built once**, and each event goes through
one member of each set, i.e. two forward passes.

`scripts/run_infer_ensemble.py` is a runnable version:

```bash
python3 scripts/run_infer_ensemble.py --root_file /path/to/signal.root --max_events 500 --calibrate
python3 scripts/run_infer_ensemble.py --root_file /path/to/signal.root --targets q1_E mjj
```

`VBFNetEnsemble(ensemble="p4")` or `VBFNetEnsemble(ensemble="hl")` runs a single set, with
the same methods. It is the engine behind `VBFNet`; use it for your own checkpoints
(`checkpoint=`, `calibration_dir=`).

### Input files

- **Tree:** a TTree (default `Events`) with the VBF-jet, H→bb, H→ττ and MET branches the
  models read. [`branch_map.yaml`](branch_map.yaml) lists all of them. Both sets read the
  same branches.
- **Event id:** the branches `run`, `luminosityBlock` and `event` are required, because
  routing uses `event`. A file without them is refused.
- **Different branch names:** use a branch map, see below.

### Files with different branch names

[`branch_map.yaml`](branch_map.yaml) maps each branch the models expect (left) to the
branch in your tree that holds the same quantity (right). The shipped file is the identity
map, i.e. the names of the training files. If your names differ:

1. Copy `branch_map.yaml` and change the right-hand side, e.g. `met_pt: PuppiMET_pt`.
   Never change the left-hand side.
2. Check the map against one of your files:

   ```bash
   python3 scripts/check_branch_map.py --root_file your.root --branch_map my_branch_map.yaml
   ```

   It lists, for every input, the branch it will read and whether that branch exists with
   the right shape (one value per event, or one per jet).
3. Pass the map when you predict:

   ```python
   out = net.predict_root("your.root", branch_map="my_branch_map.yaml")  # or a dict
   ```

   The scripts take `--branch_map my_branch_map.yaml`, and the BDT example also takes
   `inputs.branch_map` in its YAML.

**A map only renames.** It cannot convert units, recompute a variable or select objects.
Map each input to a branch with the same meaning: the same objects (VBF-jet candidates,
the two H→bb candidate jets, the two H→ττ visible legs), the same algorithm (DeepJet
b-tag scores) and the same units (GeV, φ in radians). A branch with a different meaning
still runs, but gives wrong predictions without any error. With an identity map the
inputs are bit-identical to reading the original names.

### What each branch in the map means

These are the left-hand names of [`branch_map.yaml`](branch_map.yaml). "Per jet" branches
hold one value per VBF-jet candidate; all others hold one value per event. Energies,
momenta and masses are in GeV, φ in radians.

**Event id** (required, never a model input)

| branch | per | type | meaning |
|---|---|---|---|
| `run`, `luminosityBlock` | event | integer | CMS run and luminosity-block number |
| `event` | event | integer | CMS event number. It selects the member that predicts the event (`event % 5`), so it must be the real event number, not an entry index. |

**VBF-jet candidates** (required)

| branch | per | type | meaning |
|---|---|---|---|
| `nVBFJet` | event | integer | number of VBF-jet candidates. The first `nVBFJet` entries of each `VBFJet_*` array are read, and the count also enters the model as the jet multiplicity. |
| `VBFJet_pt`, `VBFJet_eta`, `VBFJet_phi`, `VBFJet_mass` | jet | float | four-momentum of each candidate. Any order: the model sorts the jets itself. |
| `VBFJet_btagDeepFlavB` | jet | float | DeepJet b-tag score (0–1) |
| `VBFJet_btagDeepFlavCvB` | jet | float | DeepJet charm-vs-b score (0–1) |
| `VBFJet_btagDeepFlavCvL` | jet | float | DeepJet charm-vs-light score (0–1) |
| `VBFJet_btagDeepFlavQG` | jet | float | DeepJet quark-vs-gluon score (0–1) |
| `VBFJet_nConstituents` | jet | integer | number of particle-flow constituents of the jet |

Every candidate is a node of the event graph. The four scores must come from DeepJet
(DeepFlavour); scores of another tagger have different distributions and cannot be mapped
onto them.

**H→bb candidate** (required)

| branch | per | type | meaning |
|---|---|---|---|
| `b1_pt`, `b1_eta`, `b1_phi`, `b1_mass` | event | float | first jet of the H→bb candidate |
| `b2_pt`, `b2_eta`, `b2_phi`, `b2_mass` | event | float | second jet of the H→bb candidate |

Only the sum b1 + b2 enters the model, so the order of the two jets does not matter. In
the analysis ntuples an event without a valid H→bb candidate has b1 and b2 set to 0. Such
events were not in training ([validity domain](MODEL_CARD.md#validity-domain)).

**H→ττ candidate** (required)

| branch | per | type | meaning |
|---|---|---|---|
| `tau1_pt`, `tau1_eta`, `tau1_phi`, `tau1_mass` | event | float | first visible leg of the H→ττ candidate |
| `tau2_pt`, `tau2_eta`, `tau2_phi`, `tau2_mass` | event | float | second visible leg of the H→ττ candidate |

A leg is the visible τ decay product: a hadronic τ, an electron or a muon, depending on
the decay channel. Neutrinos are not included; they are part of MET. Only the sum
τ1 + τ2 enters the model, so the order of the legs does not matter.

**Missing transverse momentum** (required)

| branch | per | type | meaning |
|---|---|---|---|
| `met_pt` | event | float | magnitude of the missing transverse momentum |
| `met_phi` | event | float | its azimuthal angle |

**Generator level** (needed only with `require_truth=True`, i.e. to compare predictions
with the true quarks on simulation; never read otherwise)

| branch | per | type | meaning |
|---|---|---|---|
| `nLHEPart` | event | integer | number of LHE particles; the training selection requires exactly 6 |
| `LHEPart_pt`, `LHEPart_eta`, `LHEPart_phi`, `LHEPart_mass` | LHE particle | float | LHE particle four-momenta. The two VBF quarks are entries 4 and 5 (0-based); q1 is the one with the larger η. All twelve targets are computed from these two quarks. |
| `Hbb_isValid` | event | bool | the event has a valid H→bb candidate; part of the training selection |

## How an event is predicted

```
event ──► acceptance gate (reco only) ──► k = event % 5 ──► member k of each loaded set ──► prediction
```

| step | what happens |
|---|---|
| acceptance gate | The event needs ≥ 2 VBF jets with pT ≥ 50 GeV and \|η\| ≤ 4.7. This decides which events get a prediction; the graph is still built from **all** of the event's jets. No generator-level information is used. |
| routing | `k = event % 5`, from the CMS `event` branch (`vbfnet_ensemble/vbf_kfold.py`) |
| prediction | member k of each set alone. With calibration on, member k's own tables are used. |

**Why one member per event, not an average.** The training split put every event with
`event % 5 == k` in member k's validation set, so member k is the one model that **never
trained on it**. Both sets were trained with this same split. Routing therefore has three
consequences:

- **Signal training samples get honest predictions.** On those events the prediction is
  the out-of-fold prediction that was measured after training.
- **Signal and background are treated alike.** Background and data were never in
  training; they simply get a deterministic, reproducible member.
- **The calibration applies exactly.** Member k's calibration was fitted on the events
  routed to member k.

### `predict_root` options

| argument | default | meaning |
|---|---|---|
| `require_truth` | `False` | `False`: no generator-level cuts (normal use). `True`: apply the training selection and also return `truth_log`. |
| `acceptance` | `"default"` | `"default"` = ≥ 2 jets with pT ≥ 50 GeV, \|η\| ≤ 4.7; a dict with `min_jets`, `jet_min_pt`, `jet_max_abs_eta`; or `None` (no gate) |
| `calibrate` | `None` | override the calibration switch for this call |
| `branch_map` | `None` | files with other branch names: a `{logical: actual}` dict or a YAML file like [`branch_map.yaml`](branch_map.yaml) |

`predict_events(events)` takes in-memory physics objects; each event dict needs an
**`event`** key (the CMS event number). `predict_loader(loader)` takes any PyG loader whose
graphs carry `data.event`.

### Returned keys

T is the number of targets of the loaded sets (`len(net.target_keys)`: 12 with both, 8 for
p4 alone, 4 for hl alone).

| key | shape | meaning |
|---|---|---|
| `pred_log_full` / `pred_phys_full` | (N, T, 4) | model-space / physical predictions; axis 1 = `net.target_keys`, axis 2 = `["q16","q50","q84","point"]` |
| `pred_phys` | dict | `[target][head]` arrays for every target of the loaded sets, plus the p4-derived `q{1,2}_{pt,eta,phi,mass}` and `{mjj,deta,eta_prod,ptsum}_p4` |
| `pred_*_cal*` | | the calibrated versions (calibration on) |
| `ensemble_of` | dict | the set that produced each key, e.g. `{"mjj": "hl", "q1_E": "p4", "mjj_p4": "p4", …}` |
| `ensembles`, `target_keys` | list | the loaded sets and their targets, in array order |
| `fold_id` | (N,) | the member that predicted the row (the same index in every set) |
| `run`, `lumi`, `event` | (N,) | CMS event id |
| `event_index` | (N,) | ROOT entry of the row |
| `route`, `acceptance` | | `"event % 5"` and the gate that was applied |
| `quantile_crossing_rate` | float | fraction of (event, target) pairs whose raw quantiles crossed before sorting; `…_by_set` per set |
| `truth_log` | (N, T) | only with `require_truth=True` |
| `pred_log_members` | (5, N, T, 4) | only with `return_members=True` (diagnostic: every member on every event) |

Events failing the gate get no row. **Match rows to input events with `event_index` or
the event id, never by position.** MC event numbers can repeat across samples, so a join
across files must also key on the file.

## Using the quantiles

Per target and event you get:
- `point`, the central value;
- `q50`, the predicted median;
- `q16` and `q84`, a 68 % interval: the truth lies below `q16` in 16 % of events and below
  `q84` in 84 %.

```python
q = out["pred_phys_cal"]["mjj"]
sigma = 0.5 * (q["q84"] - q["q16"])  # symmetric ~1 sigma of m_qq
err_down = q["q50"] - q["q16"]
err_up = q["q84"] - q["q50"]
```

**Rules:**
- Use the calibrated keys whenever you use quantiles.
- The `point` head is never calibrated, so it is identical in `pred_phys` and
  `pred_phys_cal`.
- The quantiles of the regressed targets (`q1_E` …, `mjj` …) are quantiles of that target.
  For m_qq, |Δη|, η₁·η₂ and the pT sum use the hl keys (`mjj`, `deta`, `eta_prod`,
  `ptsum`) when you need an uncertainty.
- The `q16`/`q84` entries of the **derived** observables (`mjj_p4`, `q1_pt`, …) are
  computed from the component quantiles. They are not quantiles of the observable, so
  never quote them as uncertainties.

## Quantile calibration

Calibration is **off** by default. Switch it on at construction
(`use_quantile_calibration=True`), at any time (`net.use_quantile_calibration = False`), or
per call (`calibrate=True/False`); the switch acts on every loaded set. Turning it on loads
and verifies every member's tables immediately; it needs `correctionlib`.

**What it does:**
- **Correction:** rows routed to member k get member k's own additive shift, binned in
  member k's raw q50: `q_cal = q_raw + delta_k(bin of raw q50)`. The shift is in the
  target's own unit: GeV for the p4 components, `mjj` and `ptsum`, none for `deta` and
  `eta_prod`.
- **Afterwards:** the quantiles are re-sorted. The point head is untouched.
- **Fit:** each member's shifts were fitted on that member's out-of-fold events, i.e.
  exactly the events routed to it. Each set has its own tables.

The decision to ship the calibration, and its measured effect for each set, are in
[MODEL_CARD.md](MODEL_CARD.md#quantile-calibration).

## Making BDT inputs

[`example_bdt_input/`](example_bdt_input/README.md) runs the classical VBF jet-pair
selection and the routed GNN on one signal and one background file, and writes flat
TTrees with a fixed branch contract: 107 branches with both sets (the shipped config), 79
with `--target_set p4`, 47 with `--target_set hl`. As in the Python API,
`mjj_gnn_point` is the hl set's regressed value and `mjj_gnn_p4_point` the p4-derived one.

```bash
python3 example_bdt_input/make_bdt_inputs.py --config example_bdt_input/bdt_inputs_config.yaml \
    --signal /path/to/signal.root --background /path/to/background.root \
    [--branch_map my_branch_map.yaml] [--target_set p4|hl|both]
```

## Migrating from v1.0.0 or hl-v1.0.0

The weights and calibrations of both sets are unchanged, and so are the predicted values:
on 2,000 signal and 2,000 DY events (CPU), every target, head and calibrated value is
bit-identical to v1.0.0 and hl-v1.0.0. What changed:

- **One install for both models.** The two releases installed the same package name, so
  using both needed two environments. Now one checkout serves both.
- **`VBFNet` loads targets, not one model.** `VBFNet()` loads both sets. To load only the
  v1.0.0 set use `VBFNet(targets="p4")`; only the hl-v1.0.0 set, `VBFNet(targets="hl")`.
  `VBFNet` no longer takes `checkpoint=` / `calibration_dir=`; use
  `VBFNetEnsemble(ensemble=..., checkpoint=..., calibration_dir=...)` for those.
- **p4-derived observables are renamed.** v1.0.0's `pred_phys["mjj"]` (and `deta`,
  `eta_prod`, `ptsum`) is now `pred_phys["mjj_p4"]` etc.; the plain names are the hl set's
  regressed values. In BDT inputs, `<obs>_gnn_point` from the p4 set is now
  `<obs>_gnn_p4_point`.
- **Array positions.** The p4 targets keep positions 0–7 of the `*_full` arrays; with both
  sets loaded the hl targets follow at 8–11. With `targets="hl"` they are at 0–3, as in
  hl-v1.0.0. Index by `net.target_keys` rather than by position.
- **Layout.** `models/`, `calibrations/` and the set manifest moved to
  `ensembles/p4/` and `ensembles/hl/`. The top-level `RELEASE_MANIFEST.json` now hashes
  the code and pins each set's manifest.

## Versions and branches

- **`main`** holds the package. Releases are tags: **v2.0.0** = package 5.0.0, both sets.
- The earlier single-set releases stay available as tags: **v1.0.0** (p4, package 4.0.0)
  and **hl-v1.0.0** (hl, package 4.0.0+hl).
- Other branches are for different training versions of the models.

## Validity domain

Both sets were trained only on VBF HH→bbττ signal (Run3_2022EE) with both VBF quarks at
pT ≥ 50 GeV and |η| < 4.7 and a valid H→bb candidate. The acceptance gate is wider than
that selection, so some events that get a prediction lie outside the training phase
space. Read the [validity domain](MODEL_CARD.md#validity-domain) section of the model card
before using the predictions.

## Layout

```
vbfnet_ensemble/
  unified.py          VBFNet: targets -> model sets, one shared dataset, merged predictions
  predictor.py        VBFNetEnsemble: one set; load, route, calibrate, decode
  routing.py          routing rule, acceptance-gate defaults, quantile helpers
  vbf_kfold.py        the split rule used in training (event % K)
  calibration.py      per-member correctionlib shifts
  pyg_vbf_dataset.py  graph builder (inference mode, event ids, gate)
  pyg_vbf_gnn.py, vbf_config.py, transforms.py   model, config, decode
  validate.py, manifest.py                      member compatibility, release manifests
  branch_map.py       load / validate / check input branch maps
ensembles/
  p4/models/member_fold{k}.pt     the five p4 members (git-lfs)
  p4/calibrations/fold{k}/        their per-member calibration
  p4/RELEASE_MANIFEST.json        sha256 of every p4 member and calibration file
  hl/...                          the same for the hl set
scripts/                     run_infer_ensemble.py, check_branch_map.py, verify_release.py
branch_map.yaml              input branch map template (model name -> your branch)
example_bdt_input/           BDT input production (107 / 79 / 47-branch contracts)
RELEASE_MANIFEST.json        sha256 of every source file and of each set's manifest
```

## License

MIT, see [LICENSE](LICENSE).
