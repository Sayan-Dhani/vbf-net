# VBF-Net: regression of the VBF-quark kinematics in HH→bbττ

VBF-Net is a graph neural network (GNN) that reconstructs the **two VBF quarks** of
vector-boson-fusion HH→bbττ events. Its inputs are reconstructed objects: the VBF-jet
candidates, the H→bb and H→ττ candidates and the missing transverse momentum. Its
outputs are the quark four-vectors and the di-quark observables used to tag VBF
production (m_qq, |Δη_qq|, η₁·η₂, pT sum). Each output comes with a point estimate and
a per-event 68 % interval.

The models can be used from **Python** (PyTorch) and from **C++** (no dependencies). The
two interfaces give the same predictions to float32 precision.

| | |
|---|---|
| Method | message-passing GNN on the VBF-jet candidates, quantile regression |
| Training data | simulated VBF HH→bbττ, Run3_2022EE |
| Models | two model sets (`p4`, `hl`), each a 5-fold ensemble |
| Outputs | point estimate and 16 / 50 / 84 % quantiles per target; optional quantile calibration |
| Details | [MODEL_CARD.md](MODEL_CARD.md): training, performance, validity domain |

**Contents**

1. [What the model predicts](#1-what-the-model-predicts)
2. [How a prediction is made](#2-how-a-prediction-is-made)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [Examples](#5-examples)
6. [Input variables](#6-input-variables)
7. [Output reference](#7-output-reference)
8. [Quantiles and their calibration](#8-quantiles-and-their-calibration)
9. [Validity domain](#9-validity-domain)
10. [Repository layout](#10-repository-layout)

---

## 1. What the model predicts

The quarks are labelled by pseudorapidity: **q1** is the quark with the larger η, **q2**
the other one. Two model sets are provided, and each predicts its targets directly:

| target | quantity | unit | model set |
|---|---|---|---|
| `q1_E`, `q1_px`, `q1_py`, `q1_pz` | four-momentum of q1 | GeV | `p4` |
| `q2_E`, `q2_px`, `q2_py`, `q2_pz` | four-momentum of q2 | GeV | `p4` |
| `mjj` | m_qq, invariant mass of the quark pair | GeV | `hl` |
| `deta` | \|Δη_qq\| = \|η_q1 − η_q2\| | — | `hl` |
| `eta_prod` | η_q1 · η_q2 | — | `hl` |
| `ptsum` | pT_q1 + pT_q2 | GeV | `hl` |

From the predicted four-vectors, the `p4` set also computes **derived observables**:

| derived key | quantity |
|---|---|
| `q1_pt`, `q1_eta`, `q1_phi`, `q1_mass` (and `q2_…`) | cylindrical coordinates of each quark |
| `mjj_p4`, `deta_p4`, `eta_prod_p4`, `ptsum_p4` | the di-quark observables, computed from the p4 prediction |

**Naming convention.** A plain name such as `mjj` always refers to the observable
**regressed directly** by the `hl` set. The suffix `_p4` marks the same observable
**computed from the predicted four-vectors**. Both are useful, but they are different
estimators. Only the `hl` quantiles are uncertainties of the observable (see
[section 8](#8-quantiles-and-their-calibration)).

Each target has four **heads**:

| head | meaning |
|---|---|
| `point` | central value (trained with a point-estimate loss) |
| `q50` | predicted median |
| `q16`, `q84` | predicted 16 % and 84 % quantiles: a per-event 68 % interval |

## 2. How a prediction is made

```text
reconstructed event ──► acceptance gate ──► event graph ──► member k = event % 5 ──► network ──► decoding ──► (calibration)
```

1. **Acceptance gate.** An event is predicted if it has at least two VBF-jet candidates
   with pT ≥ 50 GeV and |η| ≤ 4.7. The gate only selects events; the graph is built from
   all jets of the event. No generator-level information is used.
2. **Event graph.** Every VBF-jet candidate is a node, described by its kinematics,
   DeepJet scores and number of constituents. Every ordered pair of jets is an edge,
   described by Δη, Δφ, ΔR, m_ij and similar quantities. The H→bb system (b1 + b2), the
   H→ττ system (τ1 + τ2), the MET, the jet multiplicity and H_T are global features.
3. **Fold routing.** Each model set consists of five networks (members) trained in a
   5-fold cross-validation. The split was defined by the CMS event number: events with
   `event % 5 == k` were the validation set of member k, so member k never saw them in
   training. The event is therefore predicted by **member k = event % 5** of each set.
   - On the simulated signal used for training, every prediction is an out-of-fold
     prediction, i.e. unbiased.
   - Background and data, which were never used in training, get a deterministic and
     reproducible member.
   - Because routing needs the event number, the `event` branch is required.
4. **Network and decoding.** The network predicts all targets of its set at once, in a
   transformed space (e.g. asinh(m_qq), signed log(1+|p|)). The predictions are decoded
   to physical units.
5. **Calibration (optional).** The quantiles are shifted with the tables of member k so
   that their coverage matches the nominal 16 / 50 / 84 %
   ([section 8](#8-quantiles-and-their-calibration)).

## 3. Installation

The network weights are stored with **git-lfs**. Without git-lfs, a clone contains small
pointer files instead of the weights.

```bash
git lfs install
git clone https://github.com/Sayan-Dhani/vbf-net.git
cd vbf-net
```

**Python**

```bash
pip install -e ".[calibration]"
python3 scripts/verify_release.py --strict        # checks every file against its sha256 and loads all networks
```

Requirements: Python ≥ 3.9, numpy, torch, torch-geometric, uproot, awkward, PyYAML.
correctionlib is needed only for the calibration and is installed by `[calibration]`.

**C++**

```bash
python3 scripts/export_cpp_weights.py              # once: converts the weights for C++ (needs torch, numpy)
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
./cpp/build/vbfnet_minimal .                       # predicts one example event
```

Requirements: a C++17 compiler and CMake. ROOT is needed only for the RDataFrame
examples and the `vbfnet_predict_tree` tool. Other build options (single `g++` command,
CMSSW, ROOT ACLiC) are described in [cpp/README.md](cpp/README.md).

To check that C++ and Python agree on your own files:

```bash
python3 scripts/check_cpp_parity.py --root_file /path/to/file.root --max_entries 2000
```

## 4. Quick start

**Python**

```python
from vbfnet_ensemble import VBFNet

net = VBFNet(use_quantile_calibration=True)       # loads both model sets
out = net.predict_root("signal.root", tree_name="Events")

mjj = out["pred_phys"]["mjj"]["point"]            # numpy array, one value per accepted event [GeV]
mjj_lo = out["pred_phys_cal"]["mjj"]["q16"]       # calibrated 68 % interval
mjj_hi = out["pred_phys_cal"]["mjj"]["q84"]
events = out["event"]                             # CMS event number of each row
```

**C++**

```cpp
#include "vbfnet/VBFNet.h"

vbfnet::VBFNet::Options opt;
opt.calibrate = true;
const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"), opt);   // both model sets

vbfnet::Event ev;                                 // fill from your event, see section 5.3
vbfnet::Prediction p = net.predict(ev);
if (p.accepted()) {
  double mjj    = p.get("mjj", "point");          // [GeV]
  double mjj_lo = p.get("mjj", "q16", true);      // calibrated 68 % interval
  double mjj_hi = p.get("mjj", "q84", true);
}
```

## 5. Examples

Each example shows Python first, then C++.

### 5.1 Predict all events of a ROOT file

**Python:** `predict_root` reads the file, applies the gate and returns one row per
accepted event.

```python
from vbfnet_ensemble import VBFNet

net = VBFNet(use_quantile_calibration=True)
out = net.predict_root(["sig_1.root", "sig_2.root"], tree_name="Events", max_events=10000)

print(len(out["event"]), "events predicted")
print("q1 energy, median:", out["pred_phys"]["q1_E"]["q50"][:5])
print("which member predicted each row:", out["fold_id"][:5])     # = event % 5
```

The same from the command line:

```bash
python3 scripts/run_infer_ensemble.py --root_file signal.root --max_events 500 --calibrate
```

**C++:** `vbfnet_predict_tree` writes a ROOT file with one row per input entry, in the
input order, so it can be used as a friend tree of the input.

```bash
./cpp/build/vbfnet_predict_tree --input signal.root --output signal_vbfnet.root --repo . --calibrate
```

```cpp
TFile f("signal.root");
TTree* events = f.Get<TTree>("Events");
events->AddFriend("vbfnet", "signal_vbfnet.root");
events->Draw("vbfnet_mjj_point", "vbfnet_accepted");          // branches: vbfnet_<key>_<head>[_cal]
```

### 5.2 Load only the targets you need

Loading one set halves the memory and the computing time.

**Python:** ask for targets or set names. The set that predicts them is loaded.

```python
VBFNet(targets=["mjj", "deta"])       # hl set only
VBFNet(targets="p4")                  # p4 set only
VBFNet(targets=["q1_E", "mjj"])       # both sets
VBFNet()                              # both sets (default)
```

**C++:** choose the sets by their weight files.

```cpp
vbfnet::VBFNet hl_only(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net", {"hl"}));
vbfnet::VBFNet both(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"));          // p4 and hl
```

### 5.3 Predict a single event from your own objects

Use this when the inputs are not in a ROOT tree, e.g. inside an existing event loop.

**Python:** `predict_events` takes a list of dictionaries. Here `hbb` and `htt` are the
**summed** four-vectors b1 + b2 and τ1 + τ2, given as (pT, η, φ, m).

```python
event = {
    "event": 123456789,                                   # CMS event number
    "vbf_jets": [
        {"pt": 182.4, "eta": 2.31, "phi": 0.42, "mass": 14.2,
         "btagDeepFlavB": 0.021, "btagDeepFlavCvB": 0.18, "btagDeepFlavCvL": 0.09,
         "btagDeepFlavQG": 0.71, "n_constituents": 31},
        {"pt": 96.1, "eta": -2.87, "phi": -2.61, "mass": 9.8,
         "btagDeepFlavB": 0.012, "btagDeepFlavCvB": 0.22, "btagDeepFlavCvL": 0.05,
         "btagDeepFlavQG": 0.64, "n_constituents": 22},
    ],
    "hbb": (150.2, 0.05, -0.61, 118.0),                   # b1 + b2: (pt, eta, phi, mass)
    "htt": (95.3, 1.10, 1.95, 72.4),                      # tau1 + tau2 (visible)
    "met": (41.3, -0.37),                                 # (pt, phi)
}
out = net.predict_events([event])
print(out["pred_phys"]["mjj"]["point"], out["input_index"])   # input_index: which inputs passed the gate
```

Always give the four DeepJet scores and `n_constituents`. Missing values are set to 0,
which changes the prediction.

**C++:** fill a `vbfnet::Event`. `b1`, `b2`, `tau1`, `tau2` are the individual objects;
the library forms the sums. If you only have the summed H→bb four-vector, put it in `b1`
and leave `b2` at zero.

```cpp
vbfnet::Event ev;
ev.event = 123456789;
//          pt     eta    phi   mass  DeepJet: B,    CvB,  CvL,  QG    nConstituents
ev.jets = {{182.4, 2.31, 0.42, 14.2,           0.021, 0.18, 0.09, 0.71, 31},
           {96.1, -2.87, -2.61, 9.8,           0.012, 0.22, 0.05, 0.64, 22}};
ev.b1   = {121.0, 0.31, -1.24, 15.1};   ev.b2   = {58.2, -0.47, 2.20, 9.6};
ev.tau1 = {64.8, 0.85, 1.42, 1.2};      ev.tau2 = {35.5, 1.62, 2.98, 0.1};
ev.met_pt = 41.3;  ev.met_phi = -0.37;

const vbfnet::Prediction p = net.predict(ev);
std::cout << "m_qq = " << p.get("mjj", "point") << " GeV, member " << p.fold() << "\n";
```

The same in an event loop over a TTree, with `TTreeReader`. The array types must match
your tree; this matches the analysis ntuples, where `nConstituents` is stored as
`UChar_t`.

```cpp
TTreeReader r("Events", &file);
TTreeReaderValue<ULong64_t> event(r, "event");
TTreeReaderValue<Int_t> nJet(r, "nVBFJet");
TTreeReaderArray<Float_t> pt(r, "VBFJet_pt"), eta(r, "VBFJet_eta"), phi(r, "VBFJet_phi"), mass(r, "VBFJet_mass"),
    B(r, "VBFJet_btagDeepFlavB"), CvB(r, "VBFJet_btagDeepFlavCvB"), CvL(r, "VBFJet_btagDeepFlavCvL"),
    QG(r, "VBFJet_btagDeepFlavQG");
TTreeReaderArray<UChar_t> nConst(r, "VBFJet_nConstituents");
TTreeReaderValue<Float_t> b1_pt(r, "b1_pt"), b1_eta(r, "b1_eta"), b1_phi(r, "b1_phi"), b1_m(r, "b1_mass");
// ... b2_*, tau1_*, tau2_*, met_pt, met_phi in the same way

while (r.Next()) {
  vbfnet::Event ev;
  ev.event = *event;
  ev.jets = vbfnet::makeJets(*nJet, pt, eta, phi, mass, B, CvB, CvL, QG, nConst);
  ev.b1 = {*b1_pt, *b1_eta, *b1_phi, *b1_m};
  // ... ev.b2, ev.tau1, ev.tau2, ev.met_pt, ev.met_phi
  const vbfnet::Prediction p = net.predict(ev);
  if (!p.accepted()) continue;
  // use p.get(...)
}
```

### 5.4 Per-event uncertainties

The calibrated quantiles of the `hl` targets give a per-event 68 % interval.

**Python**

```python
q = out["pred_phys_cal"]["mjj"]
sigma = 0.5 * (q["q84"] - q["q16"])           # symmetric 1σ estimate
err_down = q["q50"] - q["q16"]                # asymmetric errors around the median
err_up = q["q84"] - q["q50"]
rel_res = sigma / q["q50"]                    # relative resolution, e.g. to select well-measured events
```

**C++**

```cpp
const double lo = p.get("mjj", "q16", true), med = p.get("mjj", "q50", true), hi = p.get("mjj", "q84", true);
const double sigma = 0.5 * (hi - lo);
const double err_down = med - lo, err_up = hi - med;
```

### 5.5 Quark four-vectors and derived observables

**Python:** build the quark four-vectors, e.g. as ROOT or vector objects, or use the derived keys.

```python
P = out["pred_phys"]
q1 = [P["q1_E"]["point"], P["q1_px"]["point"], P["q1_py"]["point"], P["q1_pz"]["point"]]
q1_pt, q1_eta = P["q1_pt"]["point"], P["q1_eta"]["point"]
mjj_direct, mjj_from_p4 = P["mjj"]["point"], P["mjj_p4"]["point"]     # two estimators of m_qq
```

**C++**

```cpp
TLorentzVector q1(p.get("q1_px", "point"), p.get("q1_py", "point"), p.get("q1_pz", "point"), p.get("q1_E", "point"));
TLorentzVector q2(p.get("q2_px", "point"), p.get("q2_py", "point"), p.get("q2_pz", "point"), p.get("q2_E", "point"));
double mjj_from_p4 = p.get("mjj_p4", "point");                       // equals (q1 + q2).M()
```

### 5.6 Inside an RDataFrame analysis

The C++ library can be called from an RDataFrame computation graph, in a Python (PyROOT)
or C++ analysis. `vbfnet::defineExpression` returns the expression that predicts an event
from the tree's columns, for any column types. The network is created once and shared by
all threads; `predict` is thread-safe, so `ROOT.EnableImplicitMT()` can be used.

**Python (PyROOT)**

```python
import ROOT

ROOT.gInterpreter.AddIncludePath("/path/to/vbf-net/cpp/include")
ROOT.gSystem.Load("/path/to/vbf-net/cpp/build/libvbfnet.so")
ROOT.gInterpreter.Declare('#include "vbfnet/VBFNet.h"')
ROOT.gInterpreter.Declare("""
const vbfnet::VBFNet& vbfnet_model() {
  static const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"),
                                  [] { vbfnet::VBFNet::Options o; o.calibrate = true; return o; }());
  return net;
}""")

df = ROOT.RDataFrame("Events", "signal.root")
df = (df.Define("vbfnet", ROOT.vbfnet.defineExpression("&vbfnet_model()"))
        .Filter("vbfnet.accepted()")
        .Define("mjj_gnn", 'vbfnet.get("mjj", "point")')
        .Define("mjj_gnn_sigma", '0.5 * (vbfnet.get("mjj", "q84", true) - vbfnet.get("mjj", "q16", true))'))
h = df.Histo1D(("mjj", ";m_{qq} [GeV];events", 50, 0, 4000), "mjj_gnn")
```

**C++**

```cpp
#include <ROOT/RDataFrame.hxx>
#include "vbfnet/VBFNet.h"

const vbfnet::VBFNet& vbfnet_model() {
  static const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"),
                                  [] { vbfnet::VBFNet::Options o; o.calibrate = true; return o; }());
  return net;
}

auto df = ROOT::RDataFrame("Events", "signal.root")
              .Define("vbfnet", vbfnet::defineExpression("&vbfnet_model()"))   // JIT: needs the header declared to ROOT
              .Filter("vbfnet.accepted()")
              .Define("mjj_gnn", [](const vbfnet::Prediction& p) { return p.get("mjj", "point"); }, {"vbfnet"});
```

For JIT expressions in compiled C++, declare the header to the interpreter once:
`gInterpreter->AddIncludePath(".../cpp/include"); gInterpreter->Declare("#include \"vbfnet/VBFNet.h\"");`.
Alternatively, write the `Define` as a lambda that calls `vbfnet::makeJets` and
`vbfnet::makeEvent` with your column types.

### 5.7 Trees with other branch names

The model expects the branch names of the analysis ntuples. A **branch map** renames
them, in both interfaces. Copy [`branch_map.yaml`](branch_map.yaml) and change the
right-hand side only, e.g. `met_pt: PuppiMET_pt`. Then check it against a file:

```bash
python3 scripts/check_branch_map.py --root_file your.root --branch_map my_branch_map.yaml
```

**Python**

```python
out = net.predict_root("your.root", branch_map="my_branch_map.yaml")   # or a dict {"met_pt": "PuppiMET_pt", ...}
```

**C++**

```cpp
const auto names = vbfnet::loadBranchMap("my_branch_map.yaml");
df.Define("vbfnet", vbfnet::defineExpression("&vbfnet_model()", names));
// or: vbfnet_predict_tree ... --branch-map my_branch_map.yaml
```

A map only renames. Each branch must hold the same quantity, with the same object
definition and units ([section 6](#6-input-variables)); otherwise the prediction is wrong
without any error message.

### 5.8 Changing the acceptance gate

**Python**

```python
net.predict_root("f.root", acceptance=None)                                            # no gate (≥ 2 jets still needed)
net.predict_root("f.root", acceptance={"min_jets": 2, "jet_min_pt": 30.0, "jet_max_abs_eta": 4.7})
```

**C++**

```cpp
vbfnet::VBFNet::Options opt;
opt.override_acceptance = true;
opt.acceptance = vbfnet::Acceptance::none();          // or set min_jets, jet_min_pt, jet_max_abs_eta
```

Events outside the default gate are further from the training phase space
([section 9](#9-validity-domain)).

### 5.9 Comparison with the generator-level quarks (Python)

On simulation, `require_truth=True` applies the training selection and also returns the
true targets, computed from the LHE quarks. This is how resolution and coverage are
measured.

```python
import numpy as np
from vbfnet_ensemble import VBFNet, decode_predictions_array

net = VBFNet(targets="hl", use_quantile_calibration=True)
out = net.predict_root("signal.root", require_truth=True, acceptance=None)
truth = decode_predictions_array(out["truth_log"][:, :, None], net.target_specs)[:, :, 0]   # physical units

i = net.target_keys.index("mjj")
pred = out["pred_phys_cal"]["mjj"]
rel = (out["pred_phys"]["mjj"]["point"] - truth[:, i]) / truth[:, i]
coverage = np.mean((truth[:, i] > pred["q16"]) & (truth[:, i] < pred["q84"]))   # expected ≈ 0.68
print(f"m_qq: median bias {np.median(rel):.3f}, 68 % interval coverage {coverage:.3f}")
```

### 5.10 Inputs for a signal-versus-background BDT (Python)

[`example_bdt_input/`](example_bdt_input/README.md) runs the classical VBF jet-pair
selection and VBF-Net on a signal and a background file, and writes flat TTrees for a
BDT training. In these trees, `mjj_gnn_point` is the `hl` prediction and
`mjj_gnn_p4_point` the p4-derived one.

```bash
python3 example_bdt_input/make_bdt_inputs.py --config example_bdt_input/bdt_inputs_config.yaml \
    --signal /path/to/signal.root --background /path/to/background.root [--target_set p4|hl|both]
```

## 6. Input variables

These are the branch names the model reads (the left-hand side of
[`branch_map.yaml`](branch_map.yaml)). Energies, momenta and masses are in GeV, φ in
radians. "Per jet" branches hold one value per VBF-jet candidate.

**Event identification** (required; used for routing, never as a model input)

| branch | per | meaning |
|---|---|---|
| `run`, `luminosityBlock` | event | CMS run and luminosity-block number |
| `event` | event | CMS event number. It selects the member (`event % 5`), so it must be the true event number, not an entry index. |

**VBF-jet candidates** (required)

| branch | per | meaning |
|---|---|---|
| `nVBFJet` | event | number of candidates; the first `nVBFJet` entries of each array are read. Also a model input (jet multiplicity). |
| `VBFJet_pt`, `VBFJet_eta`, `VBFJet_phi`, `VBFJet_mass` | jet | four-momentum. Any order; the model sorts the jets by pT. |
| `VBFJet_btagDeepFlavB` | jet | DeepJet b-tag score (0–1) |
| `VBFJet_btagDeepFlavCvB`, `VBFJet_btagDeepFlavCvL` | jet | DeepJet charm-vs-b and charm-vs-light scores (0–1) |
| `VBFJet_btagDeepFlavQG` | jet | DeepJet quark-vs-gluon score (0–1) |
| `VBFJet_nConstituents` | jet | number of particle-flow constituents |

The four scores must come from DeepJet (DeepFlavour). Scores of another tagger have
different distributions and cannot be substituted.

**H→bb candidate** (required)

| branch | per | meaning |
|---|---|---|
| `b1_pt`, `b1_eta`, `b1_phi`, `b1_mass` | event | first jet of the H→bb candidate |
| `b2_pt`, `b2_eta`, `b2_phi`, `b2_mass` | event | second jet of the H→bb candidate |

Only the sum b1 + b2 enters the model, so the order does not matter. Events without an
H→bb candidate (b1 and b2 set to 0) were not part of the training.

**H→ττ candidate** (required)

| branch | per | meaning |
|---|---|---|
| `tau1_pt`, `tau1_eta`, `tau1_phi`, `tau1_mass` | event | first visible leg (τh, e or μ) |
| `tau2_pt`, `tau2_eta`, `tau2_phi`, `tau2_mass` | event | second visible leg |

Only the visible decay products enter, through their sum; the neutrinos are part of MET.

**Missing transverse momentum** (required)

| branch | per | meaning |
|---|---|---|
| `met_pt`, `met_phi` | event | magnitude and azimuthal angle of the missing transverse momentum |

**Generator level** (only for `require_truth=True`, [example 5.9](#59-comparison-with-the-generator-level-quarks-python))

| branch | per | meaning |
|---|---|---|
| `nLHEPart` | event | number of LHE particles (training selection: exactly 6) |
| `LHEPart_pt`, `LHEPart_eta`, `LHEPart_phi`, `LHEPart_mass` | LHE particle | the VBF quarks are entries 4 and 5 |
| `Hbb_isValid` | event | valid H→bb candidate (part of the training selection) |

## 7. Output reference

**Python: `predict_root`, `predict_events`.** N is the number of accepted events; T is
the number of regressed targets (`net.target_keys`: 12 with both sets).

| key | type / shape | content |
|---|---|---|
| `pred_phys[key][head]` | array (N,) | prediction in physical units, for every target and derived key |
| `pred_phys_cal[key][head]` | array (N,) | calibrated version (calibration on) |
| `pred_phys_full`, `pred_phys_cal_full` | (N, T, 4) | the same as arrays; axis 1 follows `net.target_keys`, axis 2 the heads `q16, q50, q84, point` |
| `pred_log_full`, `pred_log_cal_full` | (N, T, 4) | network output before decoding (transformed space) |
| `run`, `lumi`, `event` | (N,) | event identification of each row |
| `event_index` | (N,) | entry number in the input tree (`predict_root`) |
| `input_index` | (N,) | position in the input list (`predict_events`) |
| `fold_id` | (N,) | member that predicted the row (`event % 5`) |
| `ensemble_of[key]` | str | model set that produced a key (`"p4"` or `"hl"`) |
| `truth_log` | (N, T) | true targets in transformed space (`require_truth=True` only) |

Events that fail the gate have no row. Match rows to input events with `event_index` or
with (`run`, `lumi`, `event`), never by position.

**C++: `vbfnet::Prediction`** (one per call of `predict`)

| member | content |
|---|---|
| `accepted()` | false if the event failed the gate; all values are then NaN |
| `get(key, head, calibrated = false)` | a prediction in physical units, e.g. `get("mjj", "q84", true)` |
| `value(key_index, head_index, calibrated)` | the same by index (`net.keyIndex(key)`, `net.headIndex(head)`) |
| `logValue(target_index, head_index, calibrated)` | network output before decoding |
| `fold()` | member that predicted the event (`event % 5`) |

`net.keys()` lists all keys, `net.targetKeys()` the regressed targets and
`net.ensembleOf(key)` the set of a key. The full C++ interface is described in
[cpp/README.md](cpp/README.md).

## 8. Quantiles and their calibration

For every target the network predicts the 16 %, 50 % and 84 % quantiles of the target
given the event, trained with a quantile (pinball) loss. Ideally the true value lies below
`q16` in 16 % of the events and below `q84` in 84 % of them, so [`q16`, `q84`] is a 68 %
interval.

**Calibration.** The raw quantiles of a member can deviate slightly from their nominal
coverage. The calibration corrects them with an additive shift:

  q_cal = q_raw + Δ_k(bin of raw q50),

where Δ_k was fitted on the out-of-fold events of member k, i.e. the events that member
k predicts. Each set has its own tables. The quantiles are re-sorted after the shift.
The `point` head is never modified.

| | Python | C++ |
|---|---|---|
| switch on | `VBFNet(use_quantile_calibration=True)` or `predict_root(..., calibrate=True)` | `Options::calibrate = true` |
| read | `out["pred_phys_cal"][key][head]` | `p.get(key, head, true)` |

**Rules for using the quantiles**

- Use the calibrated quantiles whenever you use an interval.
- The quantiles of the regressed targets (`q1_E`, …, `mjj`, `deta`, `eta_prod`, `ptsum`)
  are quantiles of that target. For an uncertainty on m_qq, |Δη_qq|, η₁·η₂ or the pT sum,
  use the `hl` keys.
- The `q16` and `q84` of the **derived** keys (`mjj_p4`, `q1_pt`, …) are computed from
  the component quantiles. They are not quantiles of the derived observable; do not quote
  them as uncertainties.

The measured coverage before and after calibration is given in
[MODEL_CARD.md](MODEL_CARD.md#quantile-calibration).

## 9. Validity domain

The networks were trained on simulated VBF HH→bbττ signal (Run3_2022EE), with both VBF
quarks at pT ≥ 50 GeV and |η| < 4.7 and with a valid H→bb candidate. The acceptance gate
is looser than this selection, so some predicted events lie outside the training phase
space. Background processes are predicted with the same networks; there the output is a
well-defined function of the reconstructed event, not an estimate of quark kinematics.
Read the [validity domain](MODEL_CARD.md#validity-domain) section of the model card before
using the predictions in an analysis.

## 10. Repository layout

```text
vbfnet_ensemble/      Python package: VBFNet, graph building, network, routing, calibration
ensembles/
  p4/models/          the five p4 networks (git-lfs)
  p4/calibrations/    their calibration tables
  hl/...              the same for the hl set
cpp/                  C++ library, examples and the vbfnet_predict_tree tool (cpp/README.md)
scripts/
  run_infer_ensemble.py    predict a ROOT file from the command line
  check_branch_map.py      check a branch map against a file
  export_cpp_weights.py    convert the weights for the C++ library
  check_cpp_parity.py      compare C++ and Python on a file
  verify_release.py        verify every file against its recorded sha256
branch_map.yaml       template for renaming input branches
example_bdt_input/    production of BDT input trees
MODEL_CARD.md         training, performance, calibration and validity domain
```

## License

MIT, see [LICENSE](LICENSE).
