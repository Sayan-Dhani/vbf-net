// VBF-Net C++ inference library.
//
// Runs the released VBF-Net model sets (p4: quark four-vectors, hl: mjj, deta,
// eta_prod, ptsum) without Python: graph building, the GNN forward pass, the
// fold routing (member = event % n_folds), decoding, the per-member quantile
// calibration and the observables derived from the p4 heads. It reproduces
// vbfnet_ensemble.VBFNet (Python) to float32 precision; see cpp/README.md for
// the measured agreement.
//
// Plain C++17, no dependencies. The weights are read from the files written by
// scripts/export_cpp_weights.py (one file per model set).
//
//   vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles("/path/to/vbf-net"));
//   vbfnet::Event ev;   // fill event number, jets, b1, b2, tau1, tau2, MET
//   vbfnet::Prediction p = net.predict(ev);
//   if (p.accepted()) double mjj = p.get("mjj", "point");
//
// Thread safety: after construction every const member function may be called
// concurrently (e.g. from an RDataFrame with ImplicitMT); predict() keeps no
// state between calls.

#ifndef VBFNET_VBFNET_H
#define VBFNET_VBFNET_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vbfnet {

/// Format version of the weight files this library reads.
constexpr int kWeightFormatVersion = 1;

// ─────────────────────────────────────────────────────────────────────────────
// Inputs
// ─────────────────────────────────────────────────────────────────────────────

/// A four-vector in (pt, eta, phi, mass); GeV, phi in radians.
struct PtEtaPhiM {
  double pt = 0.0, eta = 0.0, phi = 0.0, mass = 0.0;
};

/// One VBF-jet candidate. The four scores are DeepJet (DeepFlavour) outputs.
struct Jet {
  double pt = 0.0, eta = 0.0, phi = 0.0, mass = 0.0;
  double btagDeepFlavB = 0.0;
  double btagDeepFlavCvB = 0.0;
  double btagDeepFlavCvL = 0.0;
  double btagDeepFlavQG = 0.0;
  double nConstituents = 0.0;
};

/// Everything the models read from one event (README: "What each branch in the
/// map means"). Pass the branch values unchanged: the library does the same
/// float -> double conversions as the Python package.
struct Event {
  std::uint64_t event = 0;  ///< CMS event number: picks the member, event % n_folds
  std::vector<Jet> jets;    ///< the first nVBFJet VBF-jet candidates, any order
  PtEtaPhiM b1, b2;         ///< the two jets of the H->bb candidate
  PtEtaPhiM tau1, tau2;     ///< the two visible legs of the H->tautau candidate
  double met_pt = 0.0, met_phi = 0.0;
};

/// Event-level reco gate: at least `min_jets` jets with pt >= jet_min_pt and
/// |eta| <= jet_max_abs_eta. It only decides whether an event gets a
/// prediction; the graph is always built from ALL jets. Independently of the
/// gate, an event needs >= 2 jets (a graph needs an edge).
struct Acceptance {
  bool enabled = true;
  int min_jets = 2;
  double jet_min_pt = 50.0;
  double jet_max_abs_eta = 4.7;

  static Acceptance none();  ///< no gate (only the >= 2 jets requirement)
  bool passes(const std::vector<Jet>& jets) const;
  std::string str() const;
};

/// Build the jets of an event from per-jet arrays (std::vector, ROOT::RVec,
/// TTreeReaderArray, ...). The first nJets entries are read; nJets larger than
/// an array is an error. The arrays are taken by forwarding reference because
/// some (TTreeReaderArray) can only be read through a non-const object.
template <class Pt, class Eta, class Phi, class Mass, class B, class CvB, class CvL, class QG, class NC>
std::vector<Jet> makeJets(long long nJets, Pt&& pt, Eta&& eta, Phi&& phi, Mass&& mass, B&& btagB, CvB&& btagCvB,
                          CvL&& btagCvL, QG&& btagQG, NC&& nConstituents);

/// Build an Event from its parts. Convenient inside an RDataFrame expression:
///   vbfnet::makeEvent(event, jets, {b1_pt, b1_eta, b1_phi, b1_mass}, ..., met_pt, met_phi)
Event makeEvent(std::uint64_t event, std::vector<Jet> jets, const PtEtaPhiM& b1, const PtEtaPhiM& b2,
                const PtEtaPhiM& tau1, const PtEtaPhiM& tau2, double met_pt, double met_phi);

// ─────────────────────────────────────────────────────────────────────────────
// Graph (exposed for debugging and validation; predict() builds it for you)
// ─────────────────────────────────────────────────────────────────────────────

/// Names and order of the input features, as stored in the weight file.
struct FeatureSpec {
  std::vector<std::string> node, edge, global;
};

/// The fully connected directed graph of one event: one node per jet (sorted by
/// pt, descending), one edge per ordered jet pair (i != j), source-major.
struct Graph {
  int n_nodes = 0;
  std::vector<float> x;          ///< n_nodes x node features
  std::vector<float> edge_attr;  ///< n_edges x edge features
  std::vector<float> u;          ///< global features
  std::vector<int> src, dst;     ///< edge endpoints
  int n_edges() const { return static_cast<int>(src.size()); }
};

/// Build the graph exactly as vbfnet_ensemble/pyg_vbf_dataset.py does. Throws
/// std::invalid_argument for < 2 jets or an unknown feature name.
Graph buildGraph(const Event& ev, const FeatureSpec& features);

// ─────────────────────────────────────────────────────────────────────────────
// One model set
// ─────────────────────────────────────────────────────────────────────────────

/// One model set (e.g. "p4" or "hl"): K fold members, routed by event % K, and
/// their calibration tables. Loaded from one exported weight file.
class ModelSet {
 public:
  explicit ModelSet(const std::string& weight_file);
  ~ModelSet();
  ModelSet(ModelSet&&) noexcept;
  ModelSet& operator=(ModelSet&&) noexcept;
  ModelSet(const ModelSet&) = delete;
  ModelSet& operator=(const ModelSet&) = delete;

  const std::string& name() const;
  const std::string& file() const;
  int nFolds() const;
  int route(std::uint64_t event) const;  ///< the member that predicts the event: event % nFolds()
  const std::vector<std::string>& targetKeys() const;
  const std::vector<std::string>& decodeRules() const;  ///< per target: identity, sinh, expm1, signed_expm1, exp
  const std::vector<std::string>& headNames() const;    ///< e.g. q16 q50 q84 point
  const FeatureSpec& features() const;
  const Acceptance& defaultAcceptance() const;
  bool hasCalibration() const;
  /// One line per member with its source checkpoint sha256 (provenance).
  std::string describe() const;

  /// Model-space output of member `member` on a graph: nTargets x nHeads values
  /// (row-major, heads in headNames() order, quantile heads sorted).
  std::vector<float> forward(const Graph& g, int member) const;

  /// Calibrate one member's model-space prediction (nTargets x nHeads), as
  /// vbfnet_ensemble.calibration.calibrate_member_log does: decode, add the
  /// member's shift binned in its own raw physical q50, sort the quantiles,
  /// re-encode. The point head is copied unchanged.
  std::vector<double> calibrateLog(const std::vector<double>& log, int member) const;

  /// Physical value of a model-space value of target `t` (its decode rule).
  double decode(int t, double z) const;

 private:
  friend class VBFNet;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Predictions
// ─────────────────────────────────────────────────────────────────────────────

struct Layout;  // the output keys and heads of a VBFNet; shared by its predictions

/// The prediction for one event. Values are in physical units (GeV for the p4
/// components, mjj, ptsum and the derived masses / momenta).
class Prediction {
 public:
  Prediction() = default;

  /// false: the event failed the acceptance gate (or has < 2 jets) and every
  /// value is NaN. Python returns no row for such events.
  bool accepted() const { return accepted_; }
  /// The member that predicted the event (event % n_folds; the same index in
  /// every set), or -1 if not accepted.
  int fold() const { return fold_; }
  /// Whether the calibrated values were computed (VBFNet::Options::calibrate).
  bool calibrated() const { return calibrated_; }

  /// Value of `key` (a target or a derived key, e.g. "mjj", "q1_E", "mjj_p4")
  /// for `head` ("q16", "q50", "q84", "point"). calibrated=true returns the
  /// calibrated value (Python pred_phys_cal); the point head is never calibrated.
  double get(const std::string& key, const std::string& head, bool calibrated = false) const;
  /// The same by index (VBFNet::keyIndex / headIndex), for tight loops.
  double value(int key_index, int head_index, bool calibrated = false) const;
  /// Model-space value of a REGRESSED target (Python pred_log_full / pred_log_cal_full).
  double logValue(int target_index, int head_index, bool calibrated = false) const;

  /// All physical values, [key][head] row-major (VBFNet::keys() x headNames()).
  const std::vector<double>& values(bool calibrated = false) const;

 private:
  friend class VBFNet;
  std::shared_ptr<const Layout> layout_;
  bool accepted_ = false;
  bool calibrated_ = false;
  int fold_ = -1;
  std::vector<double> phys_, phys_cal_;  // keys x heads
  std::vector<double> log_, log_cal_;    // targets x heads
};

// ─────────────────────────────────────────────────────────────────────────────
// The predictor
// ─────────────────────────────────────────────────────────────────────────────

/// One or more model sets run on the same graph, merged into one prediction,
/// like vbfnet_ensemble.VBFNet. Every event goes through member event % 5 of
/// each loaded set.
///
/// Key names follow the Python package: a plain name is the set that regresses
/// it ("mjj" is the hl prediction) and the observables derived from the p4
/// four-vectors carry "_p4" ("mjj_p4"), plus q{1,2}_{pt,eta,phi,mass}.
class VBFNet {
 public:
  struct Options {
    /// Also compute the calibrated values (Prediction::get(..., true)).
    bool calibrate = false;
    /// Use this gate instead of the one stored in the weight files
    /// (>= 2 jets with pt >= 50 GeV and |eta| <= 4.7).
    bool override_acceptance = false;
    Acceptance acceptance;
    /// Print what was loaded.
    bool verbose = true;
  };

  /// Load the given weight files, one per model set (in any order; the sets are
  /// ordered p4 first, then by name, as in Python).
  explicit VBFNet(const std::vector<std::string>& weight_files);
  VBFNet(const std::vector<std::string>& weight_files, const Options& options);
  ~VBFNet();

  /// The weight files of `sets` in a checkout of the repository:
  /// <repo_dir>/ensembles/<set>/cpp/vbfnet_<set>.bin. Empty `sets` = p4 and hl.
  static std::vector<std::string> releaseFiles(const std::string& repo_dir,
                                               const std::vector<std::string>& sets = {});

  /// Predict one event. Thread-safe.
  Prediction predict(const Event& ev) const;

  const Options& options() const { return options_; }
  const Acceptance& acceptance() const { return acceptance_; }
  /// Every key of a Prediction: the regressed targets (targetKeys()), then the
  /// derived ones.
  const std::vector<std::string>& keys() const;
  /// The regressed targets of all loaded sets, p4 first (Python net.target_keys).
  const std::vector<std::string>& targetKeys() const;
  const std::vector<std::string>& headNames() const;
  int keyIndex(const std::string& key) const;    ///< throws std::out_of_range listing the keys
  int headIndex(const std::string& head) const;  ///< throws std::out_of_range listing the heads
  /// The set that produced a key ("p4" or "hl").
  const std::string& ensembleOf(const std::string& key) const;
  const std::vector<std::shared_ptr<const ModelSet>>& sets() const { return sets_; }
  int nFolds() const;

 private:
  void init(const std::vector<std::string>& weight_files);

  Options options_;
  Acceptance acceptance_;
  std::vector<std::shared_ptr<const ModelSet>> sets_;
  std::shared_ptr<const Layout> layout_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Branch names (for ROOT trees with other names)
// ─────────────────────────────────────────────────────────────────────────────

/// The branch names the models were trained with ("logical" names), in the
/// order makeEvent / defineExpression use them.
const std::vector<std::string>& logicalBranches();

/// Read a branch map like branch_map.yaml ({logical: actual}). Only renames;
/// see the README for what each logical name means.
std::map<std::string, std::string> loadBranchMap(const std::string& yaml_file);

/// An RDataFrame expression that predicts one event from the tree's columns:
///   df.Define("vbfnet", vbfnet::defineExpression("my_net"))
/// where `my_net` is a pointer (or shared_ptr) to a VBFNet declared to the
/// interpreter. Branches missing from `branch_map` keep their logical names.
std::string defineExpression(const std::string& net_pointer,
                             const std::map<std::string, std::string>& branch_map = {});

// ─────────────────────────────────────────────────────────────────────────────
// Template implementation
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {
[[noreturn]] void throwArraySize(const char* what, long long n_jets, std::size_t size);
// Length of an array: size() (std::vector, ROOT::RVec) or GetSize() (TTreeReaderArray).
template <class A>
auto arraySize(A& a, int) -> decltype(static_cast<std::size_t>(a.size())) {
  return static_cast<std::size_t>(a.size());
}
template <class A>
auto arraySize(A& a, long) -> decltype(static_cast<std::size_t>(a.GetSize())) {
  return static_cast<std::size_t>(a.GetSize());
}
template <class A>
void checkSize(const char* what, long long n, A& a) {
  const std::size_t size = arraySize(a, 0);
  if (static_cast<long long>(size) < n) throwArraySize(what, n, size);
}
}  // namespace detail

template <class Pt, class Eta, class Phi, class Mass, class B, class CvB, class CvL, class QG, class NC>
std::vector<Jet> makeJets(long long nJets, Pt&& pt, Eta&& eta, Phi&& phi, Mass&& mass, B&& btagB, CvB&& btagCvB,
                          CvL&& btagCvL, QG&& btagQG, NC&& nConstituents) {
  if (nJets < 0) nJets = 0;
  detail::checkSize("VBFJet_pt", nJets, pt);
  detail::checkSize("VBFJet_eta", nJets, eta);
  detail::checkSize("VBFJet_phi", nJets, phi);
  detail::checkSize("VBFJet_mass", nJets, mass);
  detail::checkSize("VBFJet_btagDeepFlavB", nJets, btagB);
  detail::checkSize("VBFJet_btagDeepFlavCvB", nJets, btagCvB);
  detail::checkSize("VBFJet_btagDeepFlavCvL", nJets, btagCvL);
  detail::checkSize("VBFJet_btagDeepFlavQG", nJets, btagQG);
  detail::checkSize("VBFJet_nConstituents", nJets, nConstituents);
  std::vector<Jet> jets(static_cast<std::size_t>(nJets));
  for (long long i = 0; i < nJets; ++i) {
    Jet& j = jets[static_cast<std::size_t>(i)];
    const auto k = static_cast<std::size_t>(i);
    j.pt = static_cast<double>(pt[k]);
    j.eta = static_cast<double>(eta[k]);
    j.phi = static_cast<double>(phi[k]);
    j.mass = static_cast<double>(mass[k]);
    j.btagDeepFlavB = static_cast<double>(btagB[k]);
    j.btagDeepFlavCvB = static_cast<double>(btagCvB[k]);
    j.btagDeepFlavCvL = static_cast<double>(btagCvL[k]);
    j.btagDeepFlavQG = static_cast<double>(btagQG[k]);
    j.nConstituents = static_cast<double>(nConstituents[k]);
  }
  return jets;
}

}  // namespace vbfnet

#endif  // VBFNET_VBFNET_H
