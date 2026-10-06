# example_bdt_input — BDT inputs from VBFNet_Ensemble (HL)

`make_bdt_inputs.py` turns one signal and one background ROOT file into flat TTrees for
BDT training. Each event gets:

- the **classical VBF jet-pair selection** (`*_old`),
- the **GNN regression** (`*_gnn_*`) from ONE of the 5 k-fold members,
  `gnn_fold = event % 5` (the member that never trained on the event), and
- the bookkeeping branches `event_idx` and `label`, the CMS event id (`run`,
  `luminosityBlock`, `event`) and `gnn_fold`.

Only events passing the deployment **acceptance gate** get a row: ≥ 2 VBF jets with
pT ≥ 50 GeV and |η| ≤ 4.7. This is set in `acceptance:` in the YAML. It is a gate only:
the graph keeps all jets, and no generator-level cut is applied.

The shipped YAML sets **`model.target_set: hl`**: the members regress `mjj`, `deta`,
`eta_prod` and `ptsum` directly, and the files have **47 branches**. The script refuses a
release whose members do not regress the configured target set. (`target_set: p4` is for
the p4 model on branch `p4-regression`, whose files have 79 branches.)

## Quick start

Run from the repository root. The script puts the package on `sys.path` itself, so no
`pip install` is needed. It needs numpy, uproot, PyYAML, torch, torch_geometric and
**PyROOT**: every kinematic quantity is computed with `TLorentzVector`.

```bash
python3 example_bdt_input/make_bdt_inputs.py --config example_bdt_input/bdt_inputs_config.yaml \
    --signal /path/to/signal.root --background /path/to/background.root

# a quick test
python3 example_bdt_input/make_bdt_inputs.py --config example_bdt_input/bdt_inputs_config.yaml \
    --signal /path/to/signal.root --background /path/to/background.root \
    --max_events 2000 --outdir bdt_inputs_test
```

If your files name the branches differently, add `--branch_map my_branch_map.yaml` (see the
main [README](../README.md#files-with-different-branch-names)). With an identity map the
output is identical branch for branch.

To process whole files, set `max_events: null` in the YAML. On a T4 GPU the script runs
at about 90 events/s, mostly spent building graphs: 10,000 events per file take about
4 minutes.

## Output

Files go to `outputs.outdir` (default `bdt_inputs_ensemble/`). Each ROOT file holds a
`sig` tree (`label = 1`) and a `bkg` tree (`label = 0`).

| file | branches |
|---|---|
| `bdt_inputs_old_selection.root` | 6 bookkeeping + 13 classical-selection branches (19) |
| `bdt_inputs_gnn.root` | 6 bookkeeping + 28 GNN branches (34) |
| `bdt_inputs_combined.root` | all of them (47) |
| `bdt_inputs_provenance.json` | model version, target set, members and their split modes, routing rule, acceptance, branch renames, calibration on/off, full config, event counts |

### Branch reference

All energies and momenta are in GeV. `q1` is the **more forward** quark/jet (higher η)
and `q2` is the other one.

**Bookkeeping (6)**

| branch | type | meaning |
|---|---|---|
| `event_idx` | int64 | entry number of the event in its input ROOT file |
| `label` | int32 | 1 = signal, 0 = background |
| `run`, `luminosityBlock`, `event` | int64 | CMS event id. MC event numbers can repeat across samples, so join on file + id |
| `gnn_fold` | int32 | the member that produced the row's `*_gnn_*` values, `= event % 5` |

**Classical selection (13).** Among pairs of pT-sorted `VBFJet`s with `|Δη| > deta_cut`,
take the pair with the largest mjj. The event passes if that mjj is `> mjj_cut`.

| branch | meaning |
|---|---|
| `old_pass` | int32: 1 if the event passed, 0 otherwise |
| `mjj_old`, `deta_old`, `eta_prod_old`, `pt_sum_old` | invariant mass, \|Δη\|, η₁·η₂, pT₁+pT₂ of the selected pair |
| `q1_{E,px,py,pz}_old`, `q2_{E,px,py,pz}_old` | four-vectors of the selected jets |

For events with `old_pass = 0`, all other `*_old` branches are `-999`.

**GNN regression (28):** the four observables the members regress.

| branch | meaning |
|---|---|
| `mjj_gnn_point`, `deta_gnn_point`, `eta_prod_gnn_point`, `pt_sum_gnn_point` | central estimate of the routed member (GeV for mjj and pt_sum) |
| `{mjj,deta,eta_prod,pt_sum}_gnn_{q16,q50,q84}_raw` | uncalibrated 16/50/84 % quantiles **of the observable** |
| `{mjj,deta,eta_prod,pt_sum}_gnn_{q16,q50,q84}_cal` | calibrated quantiles; a copy of `_raw` if calibration is off |

`[q16_cal, q84_cal]` is a per-event 68 % interval of the observable. It is calibrated with
the routed member's own tables, which were fitted on exactly the events routed to that
member (see the main [README](../README.md#quantile-calibration)).

The point branches have the same names as the p4 model's derived ones, so a BDT that
reads `*_gnn_point` can switch between the two models.

### The branch list is fixed

`expected_branches()` builds the 47 names from the config. Before writing, the script
checks each tree against that list and **refuses to write if a single name is missing or
extra**.

## Configuration

`bdt_inputs_config.yaml` documents every key. Precedence:
**script defaults < YAML < command line.**

| flag | YAML key | notes |
|---|---|---|
| `--signal`, `--background` | `inputs.signal`, `inputs.background` | required |
| `--tree_name` | `inputs.tree_name` | default `Events` |
| `--branch_map` | `inputs.branch_map` | files with other branch names: a YAML map like [`../branch_map.yaml`](../branch_map.yaml), or an inline `{logical: actual}`. Applied to the GNN inputs **and** the classical selection |
| `--max_events` | `inputs.max_events` | per file; counts events **passing the acceptance gate** |
| — | `acceptance.*` | `min_jets` 2, `jet_min_pt` 50, `jet_max_abs_eta` 4.7 |
| `--signal_max_events`, `--background_max_events` | `inputs.*_max_events` | per-file override |
| `--outdir` | `outputs.outdir` | |
| `--checkpoint` | `model.checkpoint` | directory of member `.pt` files; default = bundled release |
| `--calibration_dir` | `model.calibration_dir` | per-member calibration; default = bundled |
| `--no_calibration` | `model.use_quantile_calibration: false` | `_cal` branches then copy `_raw` |
| — | `model.target_set` | `hl` (this model, 47 branches) or `p4` (79 branches) |
| `--deta_cut`, `--mjj_cut` | `old_selection.*` | defaults 3.0 and 50 GeV |
| `--batch_size`, `--num_workers`, `--device` | `runtime.*` | |
| `--fill_value` | `outputs.fill_value` | replaces NaN/inf, default −999 |

Leave `model.checkpoint` / `model.calibration_dir` as `null` to use the bundled release,
or point them at a directory holding all five members.

## How it works

For each input file:

1. **Run the routed predictor** (`VBFNetEnsemble.predict_root`, generator-truth cuts off,
   acceptance gate on). Each event goes to member `event % 5`. For every row it keeps, it
   returns the predictions, the ROOT entry number (`event_index`) and the CMS event id.
   Events failing the gate get no row.
2. **Run the classical selection** over exactly the raw entries those events come from,
   giving one row per raw entry.
3. **Align** the two by taking the classical rows at the GNN's entry numbers, so each
   output row describes one event in both halves.
4. **Check** the branch list, then write the three ROOT files and the provenance JSON.

## If you used BDT inputs from the earlier single-model script

The earlier script used a single p4 model and wrote 75 branches. Two of its problems are
fixed here, and they matter for anyone still holding its output files:

- **Row alignment.** The earlier script cut the GNN and classical halves to the same
  length and paired them row by row. After the first event the GNN dropped, every row
  paired one event's GNN values with another event's classical values. The correlation
  between `q1_E_old` and `q1_E_gnn_point` (events with `old_pass = 1`) was −0.01 for
  signal and +0.01 for background in the earlier output, against **+0.95** and **+0.84**
  with this script and the p4 model. A BDT trained on `*_old` and `*_gnn_*` branches from
  an earlier combined file saw scrambled pairs. Each branch group on its own was
  internally consistent.
- **Generator-truth cuts.** The earlier script switched on the training's truth cuts for
  any file that had truth branches, which selected signal events with generator
  information and kept only 2,368 of 175,121 DY events. This script never applies them.

The classical-selection values themselves are unchanged.
