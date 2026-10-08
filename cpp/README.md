# VBF-Net in C++

This directory runs the VBF-Net models from C++, without Python or PyTorch at run time.
It does what `vbfnet_ensemble.VBFNet` does: it builds the event graph, runs member
`event % 5` of each model set, decodes, calibrates the quantiles and computes the
p4-derived observables. The key names, heads and units are those of the Python package
([../README.md](../README.md)).

- **Plain C++17, no dependencies.** It works in a standalone program, in a CMSSW
  package, and in ROOT/RDataFrame (compiled or JIT). ROOT is needed only for the
  `vbfnet_predict_tree` tool.
- **Same numbers as Python, to float32 precision.** Both sides compute in float32, so
  values agree to about 1e-6, not bit for bit. Details are in
  [Agreement with Python](#agreement-with-python).
- **Faster than the Python package.** About 14 ms per event for both sets on one core.

## 1. Export the weights (once)

The C++ library reads one weight file per model set. Create both from the released
checkpoints:

```bash
python3 scripts/export_cpp_weights.py          # -> ensembles/{p4,hl}/cpp/vbfnet_{p4,hl}.bin (128 MB each)
```

- **Needs:** `torch` and `numpy`. torch-geometric is not needed.
- **Verified input:** every checkpoint and calibration file is checked against the
  release manifest before it is read.
- **Deterministic output:** two exports of the same release are byte-identical.
  `--check` verifies existing files; `scripts/verify_release.py` checks them too.
- **What the file holds:** the weights unchanged (float32), each member's calibration
  tables (float64), and the metadata: targets, decode rules, feature order and
  provenance hashes.
- **Re-run after updating the checkout** to a new release; `verify_release.py` reports
  a stale file.

## 2. Build

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
```

This builds `libvbfnet.so`, the example `vbfnet_minimal` and, if ROOT is found, the tool
`vbfnet_predict_tree`.

| option | default | meaning |
|---|---|---|
| `-DVBFNET_ARCH=` | `x86-64-v3` | `-march` value: AVX2/FMA, on every x86 CPU since 2013. `native` targets the build machine only; empty gives the compiler default (runs anywhere, about 2.5× slower). |
| `-DVBFNET_BUILD_EXAMPLES=` | `ON` | build the examples |
| `-DCMAKE_INSTALL_PREFIX=` | | `cmake --install cpp/build` installs the library, headers and a CMake package: `find_package(vbfnet)`, then link `vbfnet::vbfnet` |

**Without CMake.** The library is one header and one source file:

```bash
g++ -std=c++17 -O3 -march=x86-64-v3 -fPIC -shared -Icpp/include cpp/src/VBFNet.cc -o libvbfnet.so
```

Never add `-ffast-math`. The graph features and the decode rely on IEEE semantics.

**In CMSSW.** Copy `include/vbfnet/VBFNet.h` to `<Sub>/<Pkg>/interface/` and `src/VBFNet.cc`
to `<Sub>/<Pkg>/src/`. Change the `#include` to `"<Sub>/<Pkg>/interface/VBFNet.h"`, then
`scram b`. The package's `BuildFile.xml` needs only `<flags CXXFLAGS="-O3"/>` and an
`<export>` block for the library.

**With ROOT only (ACLiC).** In a macro, before loading:

```cpp
gSystem->AddIncludePath("-I/path/to/vbf-net/cpp/include");
gSystem->CompileMacro("/path/to/vbf-net/cpp/src/VBFNet.cc", "kO");   // compiled with optimisation, once
```

## 3. Check it

```bash
./cpp/build/vbfnet_minimal .                    # predicts one hand-filled event, prints every key
python3 scripts/check_cpp_parity.py --root_file /path/to/signal.root --max_entries 2000
```

`check_cpp_parity.py` runs `vbfnet_predict_tree` and the Python package on the same
entries. It requires the same selected events and the same members, and compares every
key and head, raw and calibrated. It ends with `[parity] OK` or lists what differs.

## 4. Use it

### In your own event loop

```cpp
#include "vbfnet/VBFNet.h"

vbfnet::VBFNet::Options opt;
opt.calibrate = true;                                    // also compute the calibrated quantiles
const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"), opt);   // p4 + hl; load once

vbfnet::Event ev;
ev.event = eventNumber;                                  // the CMS event number: picks the member
for (int i = 0; i < nVBFJet; ++i)
  ev.jets.push_back({pt[i], eta[i], phi[i], mass[i], btagB[i], btagCvB[i], btagCvL[i], btagQG[i], nConst[i]});
ev.b1 = {b1_pt, b1_eta, b1_phi, b1_mass};   ev.b2 = {b2_pt, b2_eta, b2_phi, b2_mass};
ev.tau1 = {t1_pt, t1_eta, t1_phi, t1_mass}; ev.tau2 = {t2_pt, t2_eta, t2_phi, t2_mass};
ev.met_pt = met_pt; ev.met_phi = met_phi;

const vbfnet::Prediction p = net.predict(ev);
if (p.accepted()) {
  double mjj     = p.get("mjj", "point");                // hl: regressed m_qq [GeV]
  double mjj_q84 = p.get("mjj", "q84", true);            // its calibrated 84 % quantile
  double q1_E    = p.get("q1_E", "q50");                 // p4: quark 1 energy, median [GeV]
  double mjj_p4  = p.get("mjj_p4", "point");             // m_qq from the p4 four-vectors
}
```

[examples/minimal.cc](examples/minimal.cc) is a complete program.
`VBFNet::releaseFiles(dir, {"hl"})` loads one set only.

### In RDataFrame

`vbfnet::defineExpression(net)` returns the expression that predicts an event from the
tree's columns. It works with any column types, from C++ or from Python:

```python
import ROOT
ROOT.gInterpreter.AddIncludePath("/path/to/vbf-net/cpp/include")
ROOT.gSystem.Load("/path/to/vbf-net/cpp/build/libvbfnet.so")
ROOT.gInterpreter.Declare('#include "vbfnet/VBFNet.h"')
ROOT.gInterpreter.Declare('''
const vbfnet::VBFNet& vbfnet_model() {   // built on first use, shared by all threads
  static const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"),
                                  [] { vbfnet::VBFNet::Options o; o.calibrate = true; return o; }());
  return net;
}''')

df = ROOT.RDataFrame("Events", "signal.root")
df = (df.Define("vbfnet", ROOT.vbfnet.defineExpression("&vbfnet_model()"))
        .Define("mjj_gnn", 'vbfnet.get("mjj", "point")')
        .Define("mjj_gnn_q84", 'vbfnet.get("mjj", "q84", true)'))
```

In C++ it is the same, with `gInterpreter->Declare(...)` and
`df.Define("vbfnet", vbfnet::defineExpression("&vbfnet_model()"))`.
`predict()` is thread-safe, so this works with `ROOT::EnableImplicitMT()`.

### On whole files

```bash
./cpp/build/vbfnet_predict_tree --input signal.root --output signal_vbfnet.root --repo . --calibrate
```

- **Output.** A tree `vbfnet` with `run`, `luminosityBlock`, `event`, `vbfnet_entry`,
  `vbfnet_accepted`, `vbfnet_fold`, and `vbfnet_<key>_<head>` for every key (`_cal`:
  calibrated).
- **One row per input entry.** Single-threaded (the default), the rows are in input order,
  so the file can be a friend tree of the input. With `--threads N`, match rows by
  `vbfnet_entry`.
- **Options.** `--keys mjj,q1_E` limits the branches. `--sets hl` loads one set.
  `--no-acceptance` disables the gate. `--branch-map` renames inputs (next section).

### Trees with other branch names

`vbfnet::loadBranchMap("my_branch_map.yaml")` reads a map in the format of
[../branch_map.yaml](../branch_map.yaml). Pass it as the second argument of
`defineExpression`, or to the tool as `--branch-map`. As in Python, a map only renames;
the meaning of each input is described in [../README.md](../README.md#what-each-branch-in-the-map-means).

## API

**`vbfnet::Event`** holds the model inputs of one event, filled from the branches of the same names:

| field | branches |
|---|---|
| `event` | `event` (selects the member, `event % 5`; must be the real CMS event number) |
| `jets` | the first `nVBFJet` entries of `VBFJet_{pt,eta,phi,mass}`, `VBFJet_btagDeepFlav{B,CvB,CvL,QG}`, `VBFJet_nConstituents`; any order |
| `b1`, `b2` | `b{1,2}_{pt,eta,phi,mass}` (H→bb candidate jets; all 0 when there is no candidate) |
| `tau1`, `tau2` | `tau{1,2}_{pt,eta,phi,mass}` (H→ττ visible legs) |
| `met_pt`, `met_phi` | `met_pt`, `met_phi` |

`vbfnet::makeJets(nVBFJet, pt, eta, …)` fills `jets` from any array type (`std::vector`,
`ROOT::RVec`, `TTreeReaderArray`). `vbfnet::makeEvent(...)` builds an `Event` in one call.

**`vbfnet::VBFNet`**

| member | meaning |
|---|---|
| `VBFNet(files, options)` | load the sets (one weight file each); they are ordered p4 first, as in Python |
| `Options::calibrate` | also compute the calibrated values (default off, as in Python) |
| `Options::override_acceptance`, `Options::acceptance` | another gate; `Acceptance::none()` disables it |
| `predict(event)` | the `Prediction`; thread-safe |
| `keys()`, `targetKeys()`, `headNames()` | the output keys (targets, then the p4-derived ones), the regressed targets (Python `net.target_keys`), and the heads |
| `keyIndex(key)`, `headIndex(head)` | indices for `Prediction::value`, for tight loops |
| `ensembleOf(key)` | the set behind a key: `"p4"` or `"hl"` |

**`vbfnet::Prediction`**

| member | meaning |
|---|---|
| `accepted()` | false if the event failed the gate (≥ 2 jets with pT ≥ 50 GeV, \|η\| ≤ 4.7, or fewer than 2 jets); every value is then NaN |
| `fold()` | the member that predicted the event (`event % 5`, the same in every set) |
| `get(key, head, calibrated = false)` | a value: Python `pred_phys[key][head]`, or with `calibrated = true`, `pred_phys_cal[key][head]` |
| `value(key_index, head_index, calibrated)` | the same, by index |
| `logValue(target_index, head_index, calibrated)` | model-space value of a regressed target (`pred_log_full`) |

The rules for using the quantiles are the same as in Python ([../README.md](../README.md#using-the-quantiles)):
use the calibrated values, and never quote the q16/q84 of a derived key (`mjj_p4`, `q1_pt`, …)
as an uncertainty.

**Differences from the Python API.** These are deliberate:

- **Rejected events.** Events that fail the gate get a `Prediction` with
  `accepted() == false`. Python returns no row for them.
- **Choosing sets.** You choose the sets by the weight files you load, not by target
  names.
- **No truth targets.** `require_truth` and `truth_log` exist only in Python; the
  validation against out-of-fold predictions stays in Python.

## Agreement with Python

`scripts/check_cpp_parity.py` on the first 2000 entries of a signal (VBF HH→bbττ) and a
DY background sample. "Model space" is the network output before decoding: signed log1p
for the p4 components, asinh(m_qq) for `mjj`, and so on.

| | events | same events and members | model-space difference: median / 99.9 % / largest | largest physical difference | calibrated values in a neighbouring bin |
|---|---|---|---|---|---|
| signal | 1735 | yes | 5e-7 / 8e-5 / 3e-3 | 0.4 GeV on a TeV-scale q2_pz; 0.03 GeV on mjj | ~1 in 10⁴ |
| DY | 1496 | yes | 5e-7 / 9e-5 / 5e-4 | 0.09 GeV | ~1 in 10⁵ |
| *Python vs Python, with and without SIMD* | 1735 | yes | 5e-7 / 9e-5 / 4e-4 | | 0 |

**Why the values are not bit-identical.** The C++ code computes every quantity the way
Python does: the same features, in the same precision (float64 where Python uses it,
float32 elsewhere). Rounding still differs in places:

- **Python's own float32 math is not reproducible.** torch evaluates `sin`, `cos`,
  `exp`, matrix products, etc. with vectorised libraries whose last bit depends on the
  CPU instruction set and even the tensor length. Run with and without SIMD, Python
  differs from itself as shown in the last row.
- **The double-precision features match exactly.** Every feature Python computes in
  float64 is bit-identical in C++.
- **The float32 features agree to 1–2 ulp,** except where float32 itself loses digits.
  The largest differences above come from a few events with a jet pair at E ≈ 3 TeV and
  m_ij ≈ 100 GeV. There, m² = E² − p² keeps only about 4 significant digits in float32,
  in Python and in C++ alike. Given the same graph, C++ and Python agree to about 1e-4
  on those events too.
- **Calibration bins.** A calibrated quantile lands in the neighbouring calibration bin
  when the raw q50 lies within this precision of a bin edge.

The tests in `tests/test_cpp.py` check each stage separately: export, every member's
network on random graphs, calibration, graph features, and the full prediction.

## Speed

Measured on one core of an Intel Xeon Silver 4216 (2.1 GHz), with both sets and
calibration on:

| | per event |
|---|---|
| C++, `-march=x86-64-v3` (default) | 14 ms |
| C++, compiler-default `-march` | 34 ms |
| Python package (`predict_root`, 1 thread, including the file read) | 37 ms |

Loading both sets takes about 1 s and 380 MB of memory. Load the network once per job
and share it between threads. RDataFrame with `EnableImplicitMT` scales with the number
of cores.

## Layout

```
cpp/
  include/vbfnet/VBFNet.h   the API
  src/VBFNet.cc             the implementation (graph, network, routing, decode, calibration)
  examples/minimal.cc       the API in a plain program
  examples/predict_tree.cc  vbfnet_predict_tree: ROOT trees in, prediction trees out
  CMakeLists.txt
scripts/export_cpp_weights.py   checkpoints -> ensembles/<set>/cpp/vbfnet_<set>.bin
scripts/check_cpp_parity.py     C++ against Python on a ROOT file
```
