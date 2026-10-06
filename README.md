# VBF-Net — VBF-quark four-vector regression for HH→bbττ

VBF-Net is a graph neural network that regressed the **four-vectors of
the two VBF quarks** in HH→bbττ VBF events from reconstructed jets, the H→bb and H→ττ
candidates and MET.

For each event it predicts 8 targets, `q{1,2}_{E,px,py,pz}` in GeV, where **q1 is the quark
with the larger (+ve) η**. Each target has a point estimate plus the 16 / 50 / 84 % quantiles.
VBF observables (m_jj, Δη, η₁·η₂, pT sum, and the quarks' pT, η, φ, mass) are derived from
the predicted four-vectors.

| | |
|---|---|
| Model | 5 k-fold members; each event is predicted by exactly one of them |
| Calibration | per-member quantile calibration included (optional) |
| Details | [MODEL_CARD.md](MODEL_CARD.md): training data, performance, validity domain |

## Install

The member checkpoints in `models/` are stored with **git-lfs**. Without it, a clone
contains small pointer files instead of the weights.

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

**What `verify_release.py --strict` checks:** every checkpoint, calibration file and
source file against `RELEASE_MANIFEST.json`, and that every checkpoint loads. It reports
an un-fetched git-lfs pointer; run `git lfs pull` to fix that.

## Quickstart

```python
from vbfnet_ensemble import VBFNet  # alias of VBFNetEnsemble

net = VBFNet(use_quantile_calibration=True)  # loads and verifies the 5 bundled members
out = net.predict_root("signal.root", tree_name="Events", max_events=100)

out["pred_phys"]["q1_E"]["point"]  # central value per event (GeV)
out["pred_phys_cal"]["q1_E"]["q84"]  # calibrated 84 % quantile
out["pred_phys"]["mjj"]["point"]  # derived from the predicted four-vectors
out["fold_id"]  # which member predicted each row (= event % 5)
out["run"], out["lumi"], out["event"]  # CMS event id of each row
out["event_index"]  # ROOT entry of each row (single file)
```

`scripts/run_infer_ensemble.py` is a runnable version:

```bash
python3 scripts/run_infer_ensemble.py --root_file /path/to/signal.root --max_events 500 --calibrate
```

### Input files

- **Tree:** a TTree (default `Events`) with the VBF-jet, H→bb, H→ττ and MET branches the
  model reads. [`branch_map.yaml`](branch_map.yaml) lists all of them.
- **Event id:** the branches `run`, `luminosityBlock` and `event` are required, because
  routing uses `event`. A file without them is refused.
- **Different branch names:** use a branch map, see below.

### Files with different branch names

[`branch_map.yaml`](branch_map.yaml) maps each branch the model expects (left) to the
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
| `LHEPart_pt`, `LHEPart_eta`, `LHEPart_phi`, `LHEPart_mass` | LHE particle | float | LHE particle four-momenta. The two VBF quarks are entries 4 and 5 (0-based); q1 is the one with the larger η. |
| `Hbb_isValid` | event | bool | the event has a valid H→bb candidate; part of the training selection |

## How an event is predicted

```
event ──► acceptance gate (reco only) ──► k = event % 5 ──► member k ──► prediction
```

| step | what happens |
|---|---|
| acceptance gate | The event needs ≥ 2 VBF jets with pT ≥ 50 GeV and \|η\| ≤ 4.7. This decides which events get a prediction; the graph is still built from **all** of the event's jets. No generator-level information is used. |
| routing | `k = event % 5`, from the CMS `event` branch (`vbfnet_ensemble/vbf_kfold.py`) |
| prediction | member k alone. With calibration on, member k's own tables are used. |

**Why one member per event, not an average.** The training split put every event with
`event % 5 == k` in member k's validation set, so member k is the one model that **never
trained on it**. Routing therefore has three consequences:

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

| key | shape | meaning |
|---|---|---|
| `pred_log_full` / `pred_phys_full` | (N, 8, 4) | model-space / physical predictions; axis 1 = `net.target_keys`, axis 2 = `["q16","q50","q84","point"]` |
| `pred_phys` | dict | `[target][head]` arrays, plus derived `mjj`, `deta`, `eta_prod`, `ptsum`, `q{1,2}_{pt,eta,phi,mass}` |
| `pred_*_cal*` | | the calibrated versions (calibration on) |
| `fold_id` | (N,) | the member that predicted the row |
| `run`, `lumi`, `event` | (N,) | CMS event id |
| `event_index` | (N,) | ROOT entry of the row |
| `route`, `acceptance` | | `"event % 5"` and the gate that was applied |
| `pred_log_members` | (5, N, 8, 4) | only with `return_members=True` (diagnostic: every member on every event) |

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
q = out["pred_phys_cal"]["q1_E"]
sigma = 0.5 * (q["q84"] - q["q16"])  # symmetric ~1 sigma
err_down = q["q50"] - q["q16"]
err_up = q["q84"] - q["q50"]
```

**Rules:**
- Use the calibrated keys whenever you use quantiles.
- The `point` head is never calibrated, so it is identical in `pred_phys` and
  `pred_phys_cal`.
- The `q16`/`q84` entries of the **derived** observables (`mjj`, …) are computed from the
  component quantiles. They are not quantiles of the observable, so never quote them as
  uncertainties.

## Quantile calibration

Calibration is **off** by default. Switch it on at construction
(`use_quantile_calibration=True`), at any time (`net.use_quantile_calibration = False`), or
per call (`calibrate=True/False`). Turning it on loads and verifies every member's tables
immediately; it needs `correctionlib`.

**What it does:**
- **Correction:** rows routed to member k get member k's own additive shift in GeV,
  binned in member k's raw q50: `q_cal = q_raw + delta_k(bin of raw q50)`.
- **Afterwards:** the quantiles are re-sorted. The point head is untouched.
- **Fit:** each member's shifts were fitted on that member's out-of-fold events, i.e.
  exactly the events routed to it.

The decision to ship the calibration, and its measured effect, are in
[MODEL_CARD.md](MODEL_CARD.md#quantile-calibration).

## Making BDT inputs

[`example_bdt_input/`](example_bdt_input/README.md) runs the classical VBF jet-pair
selection and the routed GNN on one signal and one background file, and writes flat TTrees
with a fixed 79-branch contract.

```bash
python3 example_bdt_input/make_bdt_inputs.py --config example_bdt_input/bdt_inputs_config.yaml \
    --signal /path/to/signal.root --background /path/to/background.root \
    [--branch_map my_branch_map.yaml]
```

## Validity domain

The model was trained only on VBF HH→bbττ signal (Run3_2022EE) with both VBF quarks at
pT ≥ 50 GeV and |η| < 4.7 and a valid H→bb candidate. The acceptance gate is wider than
that selection, so some events that get a prediction lie outside the training phase
space. Read the [validity domain](MODEL_CARD.md#validity-domain) section of the model card
before using the predictions.

## Layout

```
vbfnet_ensemble/
  predictor.py        VBFNetEnsemble (alias VBFNet): load, route, calibrate, decode
  routing.py          routing rule, acceptance-gate defaults, quantile helpers
  vbf_kfold.py        the split rule used in training (event % K)
  calibration.py      per-member correctionlib shifts
  pyg_vbf_dataset.py  graph builder (inference mode, event ids, gate)
  pyg_vbf_gnn.py, vbf_config.py, transforms.py   model, config, decode
  validate.py, manifest.py                      member compatibility, release manifest
  branch_map.py       load / validate / check input branch maps
models/member_fold{k}.pt     the five members (git-lfs)
calibrations/fold{k}/        per-member calibration
scripts/                     run_infer_ensemble.py, check_branch_map.py, verify_release.py
branch_map.yaml              input branch map template (model name -> your branch)
example_bdt_input/           BDT input production (79-branch contract)
RELEASE_MANIFEST.json        sha256 of every member, calibration and source file
```

## License

MIT, see [LICENSE](LICENSE).
