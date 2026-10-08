// VBF-Net C++ inference library: implementation. See include/vbfnet/VBFNet.h.
//
// Every step mirrors a function of the Python package; the comments name it.
// Do not compile with -ffast-math: the graph features and the decode rely on
// IEEE semantics (NaN propagation, exact signed zeros) to match Python.

#include "vbfnet/VBFNet.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vbfnet {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr float kPiF = static_cast<float>(M_PI);
constexpr float kTwoPiF = static_cast<float>(2.0 * M_PI);

std::string join(const std::vector<std::string>& v, const char* sep = " ") {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) out += sep;
    out += v[i];
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Four-vectors: ROOT's TLorentzVector arithmetic, which the Python graph
// builder uses (pyg_vbf_dataset._p4_components / _four_vector_dict).
// ─────────────────────────────────────────────────────────────────────────────

struct LV {
  double x = 0.0, y = 0.0, z = 0.0, t = 0.0;
};

LV fromPtEtaPhiM(double pt, double eta, double phi, double m) {  // TLorentzVector::SetPtEtaPhiM
  pt = std::fabs(pt);
  LV v;
  v.x = pt * std::cos(phi);
  v.y = pt * std::sin(phi);
  v.z = pt * std::sinh(eta);
  const double p2 = v.x * v.x + v.y * v.y + v.z * v.z;
  v.t = (m >= 0) ? std::sqrt(p2 + m * m) : std::sqrt(std::max(p2 - m * m, 0.0));
  return v;
}

LV operator+(const LV& a, const LV& b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.t + b.t}; }

struct P4 {  // pyg_vbf_dataset._four_vector_dict
  double pt, eta, phi, mass, E, px, py, pz;
};

P4 components(const LV& v) {
  P4 p;
  p.px = v.x;
  p.py = v.y;
  p.pz = v.z;
  p.E = v.t;
  p.pt = std::sqrt(v.x * v.x + v.y * v.y);  // TVector3::Perp
  const double mag = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  const double cos_theta = (mag == 0.0) ? 1.0 : v.z / mag;  // TVector3::PseudoRapidity
  if (cos_theta * cos_theta < 1)
    p.eta = -0.5 * std::log((1.0 - cos_theta) / (1.0 + cos_theta));
  else if (v.z == 0)
    p.eta = 0.0;
  else
    p.eta = v.z > 0 ? 10e10 : -10e10;
  p.phi = (v.x == 0.0 && v.y == 0.0) ? 0.0 : std::atan2(v.y, v.x);  // TVector3::Phi
  const double mm = v.t * v.t - (v.x * v.x + v.y * v.y + v.z * v.z);  // TLorentzVector::Mag
  p.mass = mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph features (pyg_vbf_dataset._assemble_graph / _build_edges)
// ─────────────────────────────────────────────────────────────────────────────

// ALL_NODE_FEATURE_NAMES, ALL_GLOBAL_FEATURE_NAMES, ALL_EDGE_FEATURE_NAMES.
const std::vector<std::string>& allNodeFeatures() {
  static const std::vector<std::string> v = {
      "log(pt+1)",      "eta",           "sin(phi)",        "cos(phi)",       "log(mass+1)",    "log(E+1)",
      "sl1p(px)",       "sl1p(py)",      "sl1p(pz)",        "centrality",     "dEta(hbb)",      "sin(dPhi(hbb))",
      "cos(dPhi(hbb))", "dR(hbb)",       "dEta(htt)",       "sin(dPhi(htt))", "cos(dPhi(htt))", "dR(htt)",
      "btagDeepFlavB",  "btagDeepFlavCvB", "btagDeepFlavCvL", "btagDeepFlavQG", "n_constituents"};
  return v;
}
const std::vector<std::string>& allGlobalFeatures() {
  static const std::vector<std::string> v = {
      "hbb_logpt",   "hbb_eta",     "hbb_sinphi",  "hbb_cosphi",  "hbb_logmass", "hbb_logE",
      "hbb_sl1p_px", "hbb_sl1p_py", "hbb_sl1p_pz", "htt_logpt",   "htt_eta",     "htt_sinphi",
      "htt_cosphi",  "htt_logmass", "htt_logE",    "htt_sl1p_px", "htt_sl1p_py", "htt_sl1p_pz",
      "met_logpt",   "met_sinphi",  "met_cosphi",  "nVBFJet_raw", "log(HT_raw)"};
  return v;
}
const std::vector<std::string>& allEdgeFeatures() {
  static const std::vector<std::string> v = {"delta_eta", "eta_product", "sin_dphi",     "cos_dphi",
                                             "delta_r",   "asinh_m_ij",  "log_pt_ratio", "kt_dist"};
  return v;
}

struct FeatureIndex {  // positions of the requested features in the ALL_* lists
  std::vector<int> node, edge, global;
};

std::vector<int> resolveNames(const std::vector<std::string>& names, const std::vector<std::string>& all,
                              const char* group) {
  std::vector<int> idx;
  for (const auto& n : names) {
    auto it = std::find(all.begin(), all.end(), n);
    if (it == all.end())
      throw std::invalid_argument(std::string("vbfnet: unknown ") + group + " feature '" + n +
                                  "'. Known: " + join(all, ", "));
    idx.push_back(static_cast<int>(it - all.begin()));
  }
  return idx;
}

FeatureIndex resolveFeatures(const FeatureSpec& spec) {
  return {resolveNames(spec.node, allNodeFeatures(), "node"), resolveNames(spec.edge, allEdgeFeatures(), "edge"),
          resolveNames(spec.global, allGlobalFeatures(), "global")};
}

// _signed_log1p(np.float32(v)): evaluated in float32 by numpy.
float signedLog1pF(double v) {
  const float x = static_cast<float>(v);
  const float s = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : (x == 0.0f ? 0.0f : x));
  return s * std::log1p(std::fabs(x));
}

double deltaPhi(double a, double b) {  // _delta_phi_np
  double d = a - b;
  while (d > M_PI) d -= 2.0 * M_PI;
  while (d < -M_PI) d += 2.0 * M_PI;
  return d;
}

float deltaPhiF(float a, float b) {  // _delta_phi_torch, float32: (d + pi) % (2 pi) - pi
  const float shifted = (a - b) + kPiF;
  float mod = std::fmod(shifted, kTwoPiF);  // torch.remainder
  if (mod != 0.0f && ((kTwoPiF < 0.0f) != (mod < 0.0f))) mod += kTwoPiF;
  return mod - kPiF;
}

struct JetP4 {
  double pt, eta, phi, mass, E, px, py, pz;
  const Jet* jet;
};

Graph buildGraphIdx(const Event& ev, const FeatureIndex& fi) {
  const std::size_t n = ev.jets.size();
  if (n < 2) throw std::invalid_argument("vbfnet: a graph needs at least 2 jets");

  // _build_reco_graph: summed H->bb and H->tautau four-vectors, jet components.
  const P4 hbb = components(fromPtEtaPhiM(ev.b1.pt, ev.b1.eta, ev.b1.phi, ev.b1.mass) +
                            fromPtEtaPhiM(ev.b2.pt, ev.b2.eta, ev.b2.phi, ev.b2.mass));
  const P4 htt = components(fromPtEtaPhiM(ev.tau1.pt, ev.tau1.eta, ev.tau1.phi, ev.tau1.mass) +
                            fromPtEtaPhiM(ev.tau2.pt, ev.tau2.eta, ev.tau2.phi, ev.tau2.mass));

  std::vector<JetP4> jets;
  jets.reserve(n);
  for (const Jet& j : ev.jets) {
    const LV v = fromPtEtaPhiM(j.pt, j.eta, j.phi, j.mass);
    jets.push_back({j.pt, j.eta, j.phi, j.mass, v.t, v.x, v.y, v.z, &j});
  }
  // sorted(jets, key=pt, reverse=True): stable, descending.
  std::stable_sort(jets.begin(), jets.end(), [](const JetP4& a, const JetP4& b) { return a.pt > b.pt; });

  double eta_max = jets[0].eta, eta_min = jets[0].eta, HT = 0.0;
  for (const auto& j : jets) {
    eta_max = std::max(eta_max, j.eta);
    eta_min = std::min(eta_min, j.eta);
    HT += j.pt;
  }
  const double eta_center = 0.5 * (eta_max + eta_min);
  const double eta_width = std::max(eta_max - eta_min, 1e-3);

  Graph g;
  g.n_nodes = static_cast<int>(n);
  const std::size_t nf = fi.node.size();
  g.x.resize(n * nf);
  std::array<double, 23> nv{};
  for (std::size_t i = 0; i < n; ++i) {
    const JetP4& j = jets[i];
    const double deta_hbb = j.eta - hbb.eta, dphi_hbb = deltaPhi(j.phi, hbb.phi);
    const double deta_htt = j.eta - htt.eta, dphi_htt = deltaPhi(j.phi, htt.phi);
    nv[0] = std::log1p(j.pt);
    nv[1] = j.eta;
    nv[2] = std::sin(j.phi);
    nv[3] = std::cos(j.phi);
    nv[4] = std::log1p(std::max(j.mass, 0.0));
    nv[5] = std::log1p(std::max(j.E, 0.0));
    nv[6] = signedLog1pF(j.px);
    nv[7] = signedLog1pF(j.py);
    nv[8] = signedLog1pF(j.pz);
    nv[9] = 1.0 - 2.0 * std::fabs((j.eta - eta_center) / eta_width);
    nv[10] = deta_hbb;
    nv[11] = std::sin(dphi_hbb);
    nv[12] = std::cos(dphi_hbb);
    nv[13] = std::sqrt(deta_hbb * deta_hbb + dphi_hbb * dphi_hbb);
    nv[14] = deta_htt;
    nv[15] = std::sin(dphi_htt);
    nv[16] = std::cos(dphi_htt);
    nv[17] = std::sqrt(deta_htt * deta_htt + dphi_htt * dphi_htt);
    nv[18] = j.jet->btagDeepFlavB;
    nv[19] = j.jet->btagDeepFlavCvB;
    nv[20] = j.jet->btagDeepFlavCvL;
    nv[21] = j.jet->btagDeepFlavQG;
    nv[22] = j.jet->nConstituents;
    for (std::size_t f = 0; f < nf; ++f) g.x[i * nf + f] = static_cast<float>(nv[fi.node[f]]);
  }

  std::array<double, 23> gv{};
  const P4* objs[2] = {&hbb, &htt};
  for (int o = 0; o < 2; ++o) {
    const P4& p = *objs[o];
    double* d = gv.data() + 9 * o;
    d[0] = std::log1p(p.pt);
    d[1] = p.eta;
    d[2] = std::sin(p.phi);
    d[3] = std::cos(p.phi);
    d[4] = std::log1p(std::max(p.mass, 0.0));
    d[5] = std::log1p(std::max(p.E, 0.0));
    d[6] = signedLog1pF(p.px);
    d[7] = signedLog1pF(p.py);
    d[8] = signedLog1pF(p.pz);
  }
  gv[18] = std::log1p(ev.met_pt);
  gv[19] = std::sin(ev.met_phi);
  gv[20] = std::cos(ev.met_phi);
  gv[21] = static_cast<double>(n);
  gv[22] = std::log1p(HT);
  g.u.resize(fi.global.size());
  for (std::size_t f = 0; f < fi.global.size(); ++f) g.u[f] = static_cast<float>(gv[fi.global[f]]);

  // _build_edges: float32 tensors, all ordered pairs, source-major.
  const std::size_t ne = n * (n - 1), ef = fi.edge.size();
  g.src.reserve(ne);
  g.dst.reserve(ne);
  g.edge_attr.resize(ne * ef);
  std::vector<float> pt(n), eta(n), phi(n), mass(n);
  for (std::size_t i = 0; i < n; ++i) {
    pt[i] = static_cast<float>(jets[i].pt);
    eta[i] = static_cast<float>(jets[i].eta);
    phi[i] = static_cast<float>(jets[i].phi);
    mass[i] = static_cast<float>(jets[i].mass);
  }
  const float eps = 1e-6f;
  std::array<float, 8> evals{};
  std::size_t e = 0;
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t d = 0; d < n; ++d) {
      if (d == s) continue;
      g.src.push_back(static_cast<int>(s));
      g.dst.push_back(static_cast<int>(d));
      const float deta = eta[s] - eta[d];
      const float dphi = deltaPhiF(phi[s], phi[d]);
      const float dR = std::sqrt(deta * deta + dphi * dphi);
      // _inv_mass_pair_torch
      const float pxi = pt[s] * std::cos(phi[s]), pyi = pt[s] * std::sin(phi[s]), pzi = pt[s] * std::sinh(eta[s]);
      const float mi = std::max(mass[s], 0.0f), mj = std::max(mass[d], 0.0f);
      const float Ei = std::sqrt(pxi * pxi + pyi * pyi + pzi * pzi + mi * mi);
      const float pxj = pt[d] * std::cos(phi[d]), pyj = pt[d] * std::sin(phi[d]), pzj = pt[d] * std::sinh(eta[d]);
      const float Ej = std::sqrt(pxj * pxj + pyj * pyj + pzj * pzj + mj * mj);
      const float sE = Ei + Ej, sx = pxi + pxj, sy = pyi + pyj, sz = pzi + pzj;
      const float m2 = sE * sE - sx * sx - sy * sy - sz * sz;
      const float mij = std::sqrt(std::max(m2, 0.0f));
      const float ptmin = std::min(pt[s], pt[d]);
      evals[0] = deta;
      evals[1] = eta[s] * eta[d];
      evals[2] = std::sin(dphi);
      evals[3] = std::cos(dphi);
      evals[4] = dR;
      evals[5] = std::asinh(mij);
      evals[6] = std::log(std::max(pt[s], eps) / std::max(pt[d], eps));
      evals[7] = std::log(std::max(ptmin * ptmin * (dR * dR), eps));
      for (std::size_t f = 0; f < ef; ++f) g.edge_attr[e * ef + f] = evals[fi.edge[f]];
      ++e;
    }
  }
  return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight file
// ─────────────────────────────────────────────────────────────────────────────

struct Tensor {
  std::vector<std::size_t> shape;
  std::vector<float> f;   // dtype 1
  std::vector<double> d;  // dtype 2
  std::size_t numel() const {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

struct WeightFile {
  std::multimap<std::string, std::vector<std::string>> meta;
  std::unordered_map<std::string, Tensor> tensors;
  std::unordered_set<std::string> used;
  std::string path;

  const std::vector<std::string>& one(const std::string& key) const {
    auto range = meta.equal_range(key);
    if (range.first == range.second) throw std::runtime_error("vbfnet: " + path + " has no '" + key + "' record");
    if (std::next(range.first) != range.second)
      throw std::runtime_error("vbfnet: " + path + " has several '" + key + "' records");
    return range.first->second;
  }
  std::string str(const std::string& key) const {
    const auto& v = one(key);
    if (v.size() != 1) throw std::runtime_error("vbfnet: " + path + ": '" + key + "' must have one value");
    return v[0];
  }
  long long integer(const std::string& key) const { return std::stoll(str(key)); }
  bool has(const std::string& name) const { return tensors.count(name) != 0; }
  const Tensor& get(const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("vbfnet: " + path + " has no tensor '" + name + "'");
    used.insert(name);
    return it->second;
  }
};

template <class T>
void readPod(std::istream& in, T& v, const std::string& path) {
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("vbfnet: " + path + " is truncated");
}

WeightFile readWeightFile(const std::string& path) {
  {
    const std::uint16_t probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&probe) != 1)
      throw std::runtime_error("vbfnet: big-endian hosts are not supported");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("vbfnet: cannot open weight file " + path +
                                    " (create it with: python3 scripts/export_cpp_weights.py)");
  WeightFile wf;
  wf.path = path;
  char magic[8];
  in.read(magic, 8);
  if (!in || std::memcmp(magic, "VBFNETW\0", 8) != 0) {
    std::string hint;
    if (in.gcount() > 0 && std::string(magic, static_cast<std::size_t>(in.gcount())).rfind("version", 0) == 0)
      hint = " (it looks like a git-lfs pointer)";
    throw std::runtime_error("vbfnet: " + path + " is not a VBF-Net C++ weight file" + hint);
  }
  std::uint32_t version = 0;
  readPod(in, version, path);
  if (static_cast<int>(version) != kWeightFormatVersion)
    throw std::runtime_error("vbfnet: " + path + " has format version " + std::to_string(version) +
                             ", this library reads version " + std::to_string(kWeightFormatVersion));
  std::uint64_t meta_len = 0;
  readPod(in, meta_len, path);
  std::string meta(meta_len, '\0');
  in.read(&meta[0], static_cast<std::streamsize>(meta_len));
  if (!in) throw std::runtime_error("vbfnet: " + path + " is truncated");
  std::istringstream lines(meta);
  for (std::string line; std::getline(lines, line);) {
    if (line.empty()) continue;
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t tab; (tab = line.find('\t', start)) != std::string::npos; start = tab + 1)
      fields.push_back(line.substr(start, tab - start));
    fields.push_back(line.substr(start));
    const std::string key = fields.front();
    fields.erase(fields.begin());
    wf.meta.emplace(key, fields);
  }
  std::uint64_t n_tensors = 0;
  readPod(in, n_tensors, path);
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    std::uint32_t name_len = 0;
    readPod(in, name_len, path);
    std::string name(name_len, '\0');
    in.read(&name[0], name_len);
    std::uint8_t dtype = 0, ndim = 0;
    readPod(in, dtype, path);
    readPod(in, ndim, path);
    Tensor t;
    for (int k = 0; k < ndim; ++k) {
      std::uint64_t s = 0;
      readPod(in, s, path);
      t.shape.push_back(static_cast<std::size_t>(s));
    }
    const std::size_t count = t.numel();
    if (dtype == 1) {
      t.f.resize(count);
      in.read(reinterpret_cast<char*>(t.f.data()), static_cast<std::streamsize>(count * sizeof(float)));
    } else if (dtype == 2) {
      t.d.resize(count);
      in.read(reinterpret_cast<char*>(t.d.data()), static_cast<std::streamsize>(count * sizeof(double)));
    } else {
      throw std::runtime_error("vbfnet: " + path + ": tensor " + name + " has unknown dtype");
    }
    if (!in) throw std::runtime_error("vbfnet: " + path + " is truncated (in tensor " + name + ")");
    wf.tensors.emplace(std::move(name), std::move(t));
  }
  char end[8];
  in.read(end, 8);
  if (!in || std::memcmp(end, "VBFNETE\0", 8) != 0)
    throw std::runtime_error("vbfnet: " + path + " is truncated or corrupt (no end marker)");
  return wf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Network building blocks (pyg_vbf_gnn.mlp / _norm_layer / _activation)
// ─────────────────────────────────────────────────────────────────────────────

// 8-wide float vectors through the GCC/Clang vector extension: AVX2 with
// -march=x86-64-v3, two SSE halves otherwise; no intrinsics needed.
#if defined(__GNUC__) || defined(__clang__)
// No function takes or returns a vector (that would make the compiler warn about
// the vector calling convention when AVX is off): loads and stores go through an
// unaligned, may_alias view of the float arrays.
typedef float vf8 __attribute__((vector_size(32)));
typedef float vf8u __attribute__((vector_size(32), aligned(4), may_alias));
typedef int vi8 __attribute__((vector_size(32)));
#define VBFNET_LOAD8(p) (*reinterpret_cast<const vf8u*>(p))
#define VBFNET_STORE8(p, v) (*reinterpret_cast<vf8u*>(p) = (v))
#define VBFNET_SPLAT8(a) (vf8{(a), (a), (a), (a), (a), (a), (a), (a)})
#define VBFNET_HAVE_VECTOR_EXT 1
#endif

enum class Act { GELU, ReLU, SiLU, ELU };

Act parseAct(const std::string& s) {
  if (s == "gelu") return Act::GELU;
  if (s == "relu") return Act::ReLU;
  if (s == "silu") return Act::SiLU;
  if (s == "elu") return Act::ELU;
  throw std::runtime_error("vbfnet: unsupported activation '" + s + "'");
}

// GELU(x) = x/2 (1 + erf(x / sqrt 2)), with erf as in torch's vectorised CPU
// kernel (Vectorized<float>::erf, Abramowitz & Stegun 7.1.26, |error| < 1.5e-7):
//   erf(a) = sign(a) (1 - r(t) t exp(-a^2)),  t = 1 / (1 + p |a|)
constexpr float kErfP = 0.3275911f, kErf1 = 0.254829592f, kErf2 = -0.284496736f, kErf3 = 1.421413741f,
                kErf4 = -1.453152027f, kErf5 = 1.061405429f;

inline float geluScalar(float x) {
  const float a = x * static_cast<float>(M_SQRT1_2), ax = std::fabs(a);
  const float t = 1.0f / (kErfP * ax + 1.0f);
  const float r = (((kErf5 * t + kErf4) * t + kErf3) * t + kErf2) * t + kErf1;
  const float e = 1.0f - r * t * std::exp(-a * a);
  return x * 0.5f * (1.0f + std::copysign(e, a));
}

void activate(Act act, float* y, std::size_t n) {
  switch (act) {
    case Act::GELU: {
      std::size_t i = 0;
#ifdef VBFNET_HAVE_VECTOR_EXT
      const vf8 one = VBFNET_SPLAT8(1.0f), zero = VBFNET_SPLAT8(0.0f);
      const vf8 hi = VBFNET_SPLAT8(88.3762626647949f), lo = VBFNET_SPLAT8(-88.3762626647949f);
      for (; i + 8 <= n; i += 8) {
        const vf8 x = VBFNET_LOAD8(y + i);
        const vf8 a = x * VBFNET_SPLAT8(static_cast<float>(M_SQRT1_2));
        const vf8 ax = a < zero ? -a : a;
        const vf8 t = one / (VBFNET_SPLAT8(kErfP) * ax + one);
        const vf8 r = (((VBFNET_SPLAT8(kErf5) * t + VBFNET_SPLAT8(kErf4)) * t + VBFNET_SPLAT8(kErf3)) * t +
                       VBFNET_SPLAT8(kErf2)) * t + VBFNET_SPLAT8(kErf1);
        // exp(-a^2), Cephes expf: range reduction by ln 2, degree-5 polynomial
        vf8 z = -(a * a);
        z = z > lo ? z : lo;
        z = z < hi ? z : hi;
        const vf8 fx = z * VBFNET_SPLAT8(1.44269504088896341f) + VBFNET_SPLAT8(0.5f);
        vf8 fl = __builtin_convertvector(__builtin_convertvector(fx, vi8), vf8);
        fl += __builtin_convertvector(fl > fx, vf8);  // floor: subtract 1 (true = -1) where truncation rounded up
        z -= fl * VBFNET_SPLAT8(0.693359375f);
        z -= fl * VBFNET_SPLAT8(-2.12194440e-4f);
        vf8 e = VBFNET_SPLAT8(1.9875691500e-4f);
        e = e * z + VBFNET_SPLAT8(1.3981999507e-3f);
        e = e * z + VBFNET_SPLAT8(8.3334519073e-3f);
        e = e * z + VBFNET_SPLAT8(4.1665795894e-2f);
        e = e * z + VBFNET_SPLAT8(1.6666665459e-1f);
        e = e * z + VBFNET_SPLAT8(5.0000001201e-1f);
        e = e * (z * z) + z + one;
        const vi8 pow2n = (__builtin_convertvector(fl, vi8) + 127) << 23;
        vf8 scale;
        std::memcpy(&scale, &pow2n, sizeof scale);
        vf8 erf = one - r * t * (e * scale);
        erf = a < zero ? -erf : erf;
        VBFNET_STORE8(y + i, x * VBFNET_SPLAT8(0.5f) * (one + erf));
      }
#endif
      for (; i < n; ++i) y[i] = geluScalar(y[i]);
      break;
    }
    case Act::ReLU:
      for (std::size_t i = 0; i < n; ++i) y[i] = y[i] > 0.0f ? y[i] : 0.0f;
      break;
    case Act::SiLU:
      for (std::size_t i = 0; i < n; ++i) y[i] = y[i] / (1.0f + std::exp(-y[i]));
      break;
    case Act::ELU:
      for (std::size_t i = 0; i < n; ++i) y[i] = y[i] > 0.0f ? y[i] : std::expm1(y[i]);
      break;
  }
}

// Y[M x N] = X[M x K] * W^T + b.
//
// The weights are repacked at load time into panels of 16 output columns,
// each panel a contiguous K x 16 block (zero-padded), and a 4-row x 16-column
// block of Y is accumulated in registers while the panel streams through.
struct Linear {
  static constexpr int P = 16;  // output columns per panel
  int in = 0, out = 0, panels = 0;
  std::vector<float> wp;  // panels x in x P
  std::vector<float> bp;  // panels x P (zeros when the layer has no bias)

  void pack(const std::vector<float>& w, const std::vector<float>* bias) {  // w: out x in (PyTorch)
    panels = (out + P - 1) / P;
    wp.assign(static_cast<std::size_t>(panels) * in * P, 0.0f);
    bp.assign(static_cast<std::size_t>(panels) * P, 0.0f);
    for (int o = 0; o < out; ++o) {
      const int p = o / P, j = o % P;
      for (int i = 0; i < in; ++i)
        wp[(static_cast<std::size_t>(p) * in + i) * P + j] = w[static_cast<std::size_t>(o) * in + i];
      if (bias) bp[static_cast<std::size_t>(o)] = (*bias)[static_cast<std::size_t>(o)];
    }
  }

  void forward(const float* X, int M, float* Y) const {
    if (wp.empty()) throw std::logic_error("vbfnet: internal error, this layer is evaluated in split form");
    const int K = in;
    float tmp[4][P];
    for (int p = 0; p < panels; ++p) {
      const float* W = wp.data() + static_cast<std::size_t>(p) * K * P;
      const float* B = bp.data() + static_cast<std::size_t>(p) * P;
      const int n0 = p * P, ncols = std::min(P, out - n0);
      int m = 0;
#ifdef VBFNET_HAVE_VECTOR_EXT
      for (; m + 4 <= M; m += 4) {
        const float* x0 = X + static_cast<std::size_t>(m) * K;
        const float* x1 = x0 + K;
        const float* x2 = x1 + K;
        const float* x3 = x2 + K;
        vf8 c00 = VBFNET_LOAD8(B), c01 = VBFNET_LOAD8(B + 8);
        vf8 c10 = c00, c11 = c01, c20 = c00, c21 = c01, c30 = c00, c31 = c01;
        const float* w = W;
        for (int k = 0; k < K; ++k, w += P) {
          const vf8 w0 = VBFNET_LOAD8(w), w1 = VBFNET_LOAD8(w + 8);
          const float s0 = x0[k], s1 = x1[k], s2 = x2[k], s3 = x3[k];
          const vf8 a0 = VBFNET_SPLAT8(s0), a1 = VBFNET_SPLAT8(s1), a2 = VBFNET_SPLAT8(s2), a3 = VBFNET_SPLAT8(s3);
          c00 += a0 * w0;
          c01 += a0 * w1;
          c10 += a1 * w0;
          c11 += a1 * w1;
          c20 += a2 * w0;
          c21 += a2 * w1;
          c30 += a3 * w0;
          c31 += a3 * w1;
        }
        const vf8 acc[4][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}};
        for (int r = 0; r < 4; ++r) {
          float* y = Y + static_cast<std::size_t>(m + r) * out + n0;
          if (ncols == P) {
            VBFNET_STORE8(y, acc[r][0]);
            VBFNET_STORE8(y + 8, acc[r][1]);
          } else {
            VBFNET_STORE8(tmp[r], acc[r][0]);
            VBFNET_STORE8(tmp[r] + 8, acc[r][1]);
            std::copy_n(tmp[r], ncols, y);
          }
        }
      }
      for (; m < M; ++m) {
        const float* x = X + static_cast<std::size_t>(m) * K;
        vf8 c0 = VBFNET_LOAD8(B), c1 = VBFNET_LOAD8(B + 8);
        const float* w = W;
        for (int k = 0; k < K; ++k, w += P) {
          const float s = x[k];
          const vf8 a = VBFNET_SPLAT8(s);
          c0 += a * VBFNET_LOAD8(w);
          c1 += a * VBFNET_LOAD8(w + 8);
        }
        VBFNET_STORE8(tmp[0], c0);
        VBFNET_STORE8(tmp[0] + 8, c1);
        std::copy_n(tmp[0], ncols, Y + static_cast<std::size_t>(m) * out + n0);
      }
#else
      for (; m < M; ++m) {
        const float* x = X + static_cast<std::size_t>(m) * K;
        float acc[P];
        std::copy_n(B, P, acc);
        const float* w = W;
        for (int k = 0; k < K; ++k, w += P)
          for (int j = 0; j < P; ++j) acc[j] += x[k] * w[j];
        std::copy_n(acc, ncols, Y + static_cast<std::size_t>(m) * out + n0);
      }
#endif
    }
  }
};

// LayerNorm, eval-mode BatchNorm1d, or Identity.
struct Norm {
  enum Kind { None, Layer, Batch } kind = None;
  int dim = 0;
  float eps = 1e-5f;
  std::vector<float> w, b;  // Batch: folded into scale (w) and shift (b)

  void apply(float* Y, int M) const {
    if (kind == None) return;
    if (kind == Batch) {  // y = x * (gamma / sqrt(var + eps)) + (beta - mean * scale)
      for (int m = 0; m < M; ++m) {
        float* y = Y + static_cast<std::size_t>(m) * dim;
        for (int j = 0; j < dim; ++j) y[j] = y[j] * w[j] + b[j];
      }
      return;
    }
    for (int m = 0; m < M; ++m) {
      float* y = Y + static_cast<std::size_t>(m) * dim;
      double sum = 0.0;
      for (int j = 0; j < dim; ++j) sum += y[j];
      const double mean = sum / dim;
      double var = 0.0;
      for (int j = 0; j < dim; ++j) var += (y[j] - mean) * (y[j] - mean);
      var /= dim;
      const float meanf = static_cast<float>(mean);
      const float rstd = static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
      for (int j = 0; j < dim; ++j) y[j] = (y[j] - meanf) * rstd * w[j] + b[j];
    }
  }
};

struct MLP {
  std::vector<Linear> lin;
  std::vector<Norm> norm;  // after every Linear but the last
  Act act = Act::GELU;

  int in() const { return lin.front().in; }
  int out() const { return lin.back().out; }

  std::vector<float> forward(const float* X, int M) const {
    std::vector<float> z(static_cast<std::size_t>(M) * lin[0].out);
    lin[0].forward(X, M, z.data());
    return forwardRest(std::move(z), M);
  }

  /// Continue from the output of the first Linear (M x lin[0].out).
  std::vector<float> forwardRest(std::vector<float> cur, int M) const {
    std::vector<float> next;
    for (std::size_t i = 0; i + 1 < lin.size(); ++i) {
      norm[i].apply(cur.data(), M);
      activate(act, cur.data(), cur.size());
      next.resize(static_cast<std::size_t>(M) * lin[i + 1].out);
      lin[i + 1].forward(cur.data(), M, next.data());
      cur.swap(next);
    }
    return cur;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Loading a member from its PyTorch parameter names
// ─────────────────────────────────────────────────────────────────────────────

struct Arch {
  int n_layers = 0;
  std::vector<std::string> aggregation;
  std::string pool;
  Act act = Act::GELU;
  int stride = 3;  // positions per hidden layer in nn.Sequential: Linear, norm, act (+ Dropout)
  bool input_bn = false, pair = false;
  float ln_eps = 1e-5f, bn_eps = 1e-5f;
};

Linear loadLinear(WeightFile& wf, const std::string& prefix, bool bias = true) {
  const Tensor& w = wf.get(prefix + ".weight");
  if (w.shape.size() != 2 || w.f.empty()) throw std::runtime_error("vbfnet: " + prefix + ".weight is not a matrix");
  Linear L;
  L.out = static_cast<int>(w.shape[0]);
  L.in = static_cast<int>(w.shape[1]);
  if (bias) {
    const std::vector<float>& b = wf.get(prefix + ".bias").f;
    if (static_cast<int>(b.size()) != L.out) throw std::runtime_error("vbfnet: bad bias size for " + prefix);
    L.pack(w.f, &b);
  } else {
    L.pack(w.f, nullptr);
  }
  return L;
}

// A norm module at `prefix`: LayerNorm (weight, bias), BatchNorm1d (+ running
// stats) or Identity (no parameters).
Norm loadNorm(WeightFile& wf, const std::string& prefix, int dim, const Arch& a) {
  Norm n;
  n.dim = dim;
  if (!wf.has(prefix + ".weight") && !wf.has(prefix + ".running_mean")) return n;  // Identity
  if (wf.has(prefix + ".running_mean")) {
    const auto& mean = wf.get(prefix + ".running_mean").f;
    const auto& var = wf.get(prefix + ".running_var").f;
    std::vector<float> gamma(static_cast<std::size_t>(dim), 1.0f), beta(static_cast<std::size_t>(dim), 0.0f);
    if (wf.has(prefix + ".weight")) {
      gamma = wf.get(prefix + ".weight").f;
      beta = wf.get(prefix + ".bias").f;
    }
    if (static_cast<int>(mean.size()) != dim || static_cast<int>(gamma.size()) != dim)
      throw std::runtime_error("vbfnet: bad BatchNorm size for " + prefix);
    n.kind = Norm::Batch;
    n.eps = a.bn_eps;
    n.w.resize(static_cast<std::size_t>(dim));
    n.b.resize(static_cast<std::size_t>(dim));
    for (int j = 0; j < dim; ++j) {
      const float invstd = 1.0f / std::sqrt(var[j] + n.eps);
      n.w[j] = gamma[j] * invstd;
      n.b[j] = beta[j] - mean[j] * n.w[j];
    }
    return n;
  }
  n.kind = Norm::Layer;
  n.eps = a.ln_eps;
  n.w = wf.get(prefix + ".weight").f;
  n.b = wf.get(prefix + ".bias").f;
  if (static_cast<int>(n.w.size()) != dim) throw std::runtime_error("vbfnet: bad LayerNorm size for " + prefix);
  return n;
}

// pyg_vbf_gnn.mlp(): Linear at 0, stride, 2*stride, ...; between two Linears a
// norm (position +1) and an activation.
MLP loadMLP(WeightFile& wf, const std::string& prefix, const Arch& a) {
  MLP m;
  m.act = a.act;
  for (int pos = 0; wf.has(prefix + "." + std::to_string(pos) + ".weight"); pos += a.stride) {
    m.lin.push_back(loadLinear(wf, prefix + "." + std::to_string(pos)));
    if (wf.has(prefix + "." + std::to_string(pos + a.stride) + ".weight"))
      m.norm.push_back(loadNorm(wf, prefix + "." + std::to_string(pos + 1), m.lin.back().out, a));
  }
  if (m.lin.empty()) throw std::runtime_error("vbfnet: no layers under " + prefix);
  for (std::size_t i = 1; i < m.lin.size(); ++i)
    if (m.lin[i].in != m.lin[i - 1].out) throw std::runtime_error("vbfnet: inconsistent layer sizes in " + prefix);
  return m;
}

// Columns [col0, col0 + ncols) of the first Linear of the MLPs at `prefixes`,
// stacked along the output axis. W [x_a, x_b, ...] = W_a x_a + W_b x_b + ...,
// so an MLP whose input concatenates per-edge copies of node features can do
// the node parts once per node instead of once per edge.
Linear firstLayerBlock(WeightFile& wf, const std::vector<std::string>& prefixes, int col0, int ncols,
                       bool with_bias) {
  std::vector<float> w, b;
  Linear L;
  L.in = ncols;
  for (const auto& p : prefixes) {
    const Tensor& t = wf.get(p + ".0.weight");
    const int out = static_cast<int>(t.shape[0]), in = static_cast<int>(t.shape[1]);
    for (int o = 0; o < out; ++o)
      w.insert(w.end(), t.f.begin() + static_cast<std::ptrdiff_t>(o) * in + col0,
               t.f.begin() + static_cast<std::ptrdiff_t>(o) * in + col0 + ncols);
    const auto& bias = wf.get(p + ".0.bias").f;
    b.insert(b.end(), bias.begin(), bias.end());
    L.out += out;
  }
  L.pack(w, with_bias ? &b : nullptr);
  return L;
}

struct EdgeConv {  // pyg_vbf_gnn.EdgeConvWithAttr
  MLP edge_mlp, gate_mlp, msg_mlp, upd_mlp, global_mlp;
  Norm edge_norm, upd_norm, global_norm;
  Linear res_proj, global_res;
  // First layers in split form (the bias rides on the u block):
  //   edge_mlp over [e, x_src, x_dst, u];  gate_mlp and msg_mlp, stacked, over [x_dst, x_src, e_new, u]
  Linear edge_e, edge_src, edge_dst, edge_u;
  Linear gm_dst, gm_src, gm_e, gm_u;
};

struct Member {  // pyg_vbf_gnn.PyGVBFGNN, eval mode
  Norm node_in, edge_in, global_in;
  MLP node_enc, edge_enc, global_enc;
  std::vector<EdgeConv> layers;
  bool pair = false;
  MLP pair_score, pair_proj;
  MLP regressor;
};

void requireDim(int got, int want, const std::string& what) {
  if (got != want)
    throw std::runtime_error("vbfnet: " + what + " has width " + std::to_string(got) + ", expected " +
                             std::to_string(want));
}

Member loadMember(WeightFile& wf, const std::string& p, const Arch& a, int n_node, int n_edge, int n_global,
                  int n_out) {
  Member m;
  if (a.input_bn) {
    m.node_in = loadNorm(wf, p + "node_input_norm", n_node, a);
    m.edge_in = loadNorm(wf, p + "edge_input_norm", n_edge, a);
    m.global_in = loadNorm(wf, p + "global_input_norm", n_global, a);
  }
  m.node_enc = loadMLP(wf, p + "node_encoder", a);
  m.edge_enc = loadMLP(wf, p + "edge_encoder", a);
  m.global_enc = loadMLP(wf, p + "global_encoder", a);
  requireDim(m.node_enc.in(), n_node, "node encoder input");
  requireDim(m.edge_enc.in(), n_edge, "edge encoder input");
  requireDim(m.global_enc.in(), n_global, "global encoder input");
  const int dn = m.node_enc.out(), de = m.edge_enc.out(), dg = m.global_enc.out();
  const int n_aggr = static_cast<int>(a.aggregation.size());

  for (int l = 0; l < a.n_layers; ++l) {
    const std::string q = p + "conv_layers." + std::to_string(l) + ".";
    EdgeConv c;
    c.edge_mlp = loadMLP(wf, q + "edge_mlp", a);
    c.edge_norm = loadNorm(wf, q + "edge_norm", de, a);
    c.gate_mlp = loadMLP(wf, q + "gate_mlp", a);
    c.msg_mlp = loadMLP(wf, q + "msg_mlp", a);
    c.upd_mlp = loadMLP(wf, q + "upd_mlp", a);
    c.upd_norm = loadNorm(wf, q + "upd_norm", dn, a);
    c.res_proj = loadLinear(wf, q + "res_proj", false);
    c.global_mlp = loadMLP(wf, q + "global_mlp", a);
    c.global_norm = loadNorm(wf, q + "global_norm", dg, a);
    c.global_res = loadLinear(wf, q + "global_res", false);
    const int dm = c.msg_mlp.out();
    c.edge_e = firstLayerBlock(wf, {q + "edge_mlp"}, 0, de, false);
    c.edge_src = firstLayerBlock(wf, {q + "edge_mlp"}, de, dn, false);
    c.edge_dst = firstLayerBlock(wf, {q + "edge_mlp"}, de + dn, dn, false);
    c.edge_u = firstLayerBlock(wf, {q + "edge_mlp"}, de + 2 * dn, dg, true);
    const std::vector<std::string> gm = {q + "gate_mlp", q + "msg_mlp"};
    c.gm_dst = firstLayerBlock(wf, gm, 0, dn, false);
    c.gm_src = firstLayerBlock(wf, gm, dn, dn, false);
    c.gm_e = firstLayerBlock(wf, gm, 2 * dn, de, false);
    c.gm_u = firstLayerBlock(wf, gm, 2 * dn + de, dg, true);
    for (MLP* mlp : {&c.edge_mlp, &c.gate_mlp, &c.msg_mlp}) {  // only the split form is evaluated
      std::vector<float>().swap(mlp->lin[0].wp);
      std::vector<float>().swap(mlp->lin[0].bp);
    }
    requireDim(c.edge_mlp.in(), de + 2 * dn + dg, q + "edge_mlp input");
    requireDim(c.edge_mlp.out(), de, q + "edge_mlp output");
    requireDim(c.gate_mlp.in(), 2 * dn + de + dg, q + "gate_mlp input");
    requireDim(c.gate_mlp.out(), 1, q + "gate_mlp output");
    requireDim(c.msg_mlp.in(), 2 * dn + de + dg, q + "msg_mlp input");
    requireDim(c.upd_mlp.in(), dn + n_aggr * dm + dg, q + "upd_mlp input");
    requireDim(c.upd_mlp.out(), dn, q + "upd_mlp output");
    requireDim(c.global_mlp.in(), dg + 2 * dn + 2 * de, q + "global_mlp input");
    requireDim(c.global_mlp.out(), dg, q + "global_mlp output");
    m.layers.push_back(std::move(c));
  }
  if (wf.has(p + "conv_layers." + std::to_string(a.n_layers) + ".edge_mlp.0.weight"))
    throw std::runtime_error("vbfnet: the file has more message-passing layers than model.n_layers");

  m.pair = a.pair;
  int pair_dim = 0;
  if (a.pair) {
    m.pair_score = loadMLP(wf, p + "edge_pair_score", a);
    m.pair_proj = loadMLP(wf, p + "edge_pair_proj", a);
    requireDim(m.pair_score.in(), de + dg, "edge_pair_score input");
    requireDim(m.pair_score.out(), 1, "edge_pair_score output");
    pair_dim = m.pair_proj.out();
  }
  m.regressor = loadMLP(wf, p + "regressor", a);
  const int pool_dim = a.pool == "mean+max" ? 2 * dn : dn;
  requireDim(m.regressor.in(), pool_dim + dg + pair_dim, "regressor input");
  requireDim(m.regressor.out(), n_out, "regressor output (targets x heads)");
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode / encode / calibration (transforms.py, calibration.py)
// ─────────────────────────────────────────────────────────────────────────────

enum class Rule { Identity, Sinh, Expm1, SignedExpm1, Exp };

Rule parseRule(const std::string& s) {
  if (s == "identity") return Rule::Identity;
  if (s == "sinh") return Rule::Sinh;
  if (s == "expm1") return Rule::Expm1;
  if (s == "signed_expm1") return Rule::SignedExpm1;
  if (s == "exp") return Rule::Exp;
  throw std::runtime_error("vbfnet: unsupported decode rule '" + s + "'");
}

double npSign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : (x == 0 ? 0.0 : x)); }

double decodeValue(Rule r, double z) {  // transforms._decode_array
  switch (r) {
    case Rule::Identity: return z;
    case Rule::Sinh: return std::sinh(z);
    case Rule::Expm1: return std::expm1(z);
    case Rule::SignedExpm1: return npSign(z) * std::expm1(std::fabs(z));
    case Rule::Exp: return std::exp(z);
  }
  return kNaN;
}

double encodeValue(Rule r, double x) {  // transforms._encode_array
  switch (r) {
    case Rule::Identity: return x;
    case Rule::Sinh: return std::asinh(x);
    case Rule::Expm1: return std::log1p(x);
    case Rule::SignedExpm1: return npSign(x) * std::log1p(std::fabs(x));
    case Rule::Exp: return std::log(x);
  }
  return kNaN;
}

// np.sort: ascending, NaN last.
void sortNanLast(double* v, int n) {
  std::sort(v, v + n, [](double a, double b) { return std::isnan(b) ? !std::isnan(a) : a < b; });
}

struct Binning {  // calibration._binning_node / lookup_shift
  std::vector<double> edges, content;
  double lookup(double x) const {
    if (!std::isfinite(x)) return kNaN;
    long idx = static_cast<long>(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) - 1;
    idx = std::max(0L, std::min(idx, static_cast<long>(content.size()) - 1));
    return content[static_cast<std::size_t>(idx)];
  }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance and event helpers
// ─────────────────────────────────────────────────────────────────────────────

Acceptance Acceptance::none() {
  Acceptance a;
  a.enabled = false;
  return a;
}

bool Acceptance::passes(const std::vector<Jet>& jets) const {  // pyg_vbf_dataset.passes_acceptance
  if (!enabled) return true;
  int good = 0;
  for (const Jet& j : jets)
    if (j.pt >= jet_min_pt && std::fabs(j.eta) <= jet_max_abs_eta) ++good;
  return good >= min_jets;
}

std::string Acceptance::str() const {
  if (!enabled) return "none";
  std::ostringstream os;
  os << ">= " << min_jets << " jets with pt >= " << jet_min_pt << " GeV and |eta| <= " << jet_max_abs_eta;
  return os.str();
}

Event makeEvent(std::uint64_t event, std::vector<Jet> jets, const PtEtaPhiM& b1, const PtEtaPhiM& b2,
                const PtEtaPhiM& tau1, const PtEtaPhiM& tau2, double met_pt, double met_phi) {
  Event ev;
  ev.event = event;
  ev.jets = std::move(jets);
  ev.b1 = b1;
  ev.b2 = b2;
  ev.tau1 = tau1;
  ev.tau2 = tau2;
  ev.met_pt = met_pt;
  ev.met_phi = met_phi;
  return ev;
}

namespace detail {
void throwArraySize(const char* what, long long n_jets, std::size_t size) {
  throw std::out_of_range(std::string("vbfnet::makeJets: nVBFJet = ") + std::to_string(n_jets) + " but " + what +
                          " has only " + std::to_string(size) + " entries");
}
}  // namespace detail

Graph buildGraph(const Event& ev, const FeatureSpec& features) { return buildGraphIdx(ev, resolveFeatures(features)); }

// ─────────────────────────────────────────────────────────────────────────────
// ModelSet
// ─────────────────────────────────────────────────────────────────────────────

struct ModelSet::Impl {
  std::string name, file, description;
  int n_folds = 0;
  std::vector<std::string> targets, decode_names, heads;
  std::vector<Rule> rules;
  FeatureSpec features;
  FeatureIndex feature_index;
  Acceptance acceptance;
  Arch arch;
  int n_q = 0;            // quantile heads, first in head order
  int i50 = -1;           // index of q50
  bool has_point = false; // point head, last
  std::vector<Member> members;
  bool has_calibration = false;
  std::vector<std::vector<std::vector<Binning>>> calib;  // [member][target][quantile head]

  std::vector<float> forward(const Graph& g, int k) const;
};

ModelSet::ModelSet(const std::string& weight_file) : impl_(new Impl) {
  Impl& s = *impl_;
  WeightFile wf = readWeightFile(weight_file);
  s.file = weight_file;
  if (wf.one("format").size() < 2 || wf.one("format")[0] != "vbfnet-cpp-weights")
    throw std::runtime_error("vbfnet: " + weight_file + " has an unknown format record");
  s.name = wf.str("set");
  s.n_folds = static_cast<int>(wf.integer("n_folds"));
  if (wf.str("route") != "event % n_folds")
    throw std::runtime_error("vbfnet: " + weight_file + ": unsupported routing rule '" + wf.str("route") + "'");
  s.targets = wf.one("targets");
  s.decode_names = wf.one("decode");
  s.heads = wf.one("heads");
  if (s.decode_names.size() != s.targets.size())
    throw std::runtime_error("vbfnet: " + weight_file + ": one decode rule per target expected");
  for (const auto& r : s.decode_names) s.rules.push_back(parseRule(r));
  for (std::size_t h = 0; h < s.heads.size(); ++h) {
    const std::string& name = s.heads[h];
    if (!name.empty() && name[0] == 'q') {
      if (static_cast<int>(h) != s.n_q) throw std::runtime_error("vbfnet: quantile heads must come first");
      if (name == "q50") s.i50 = static_cast<int>(h);
      ++s.n_q;
    } else if (name == "point" && h + 1 == s.heads.size()) {
      s.has_point = true;
    } else {
      throw std::runtime_error("vbfnet: unsupported head '" + name + "'");
    }
  }
  s.features.node = wf.one("node_features");
  s.features.edge = wf.one("edge_features");
  s.features.global = wf.one("global_features");
  s.feature_index = resolveFeatures(s.features);

  s.arch.n_layers = static_cast<int>(wf.integer("model.n_layers"));
  s.arch.aggregation = wf.one("model.aggregation");
  for (const auto& ag : s.arch.aggregation)
    if (ag != "sum" && ag != "add" && ag != "mean" && ag != "max")
      throw std::runtime_error("vbfnet: unsupported aggregation '" + ag + "'");
  s.arch.pool = wf.str("model.pool");
  if (s.arch.pool != "mean" && s.arch.pool != "max" && s.arch.pool != "mean+max")
    throw std::runtime_error("vbfnet: unsupported pool '" + s.arch.pool + "'");
  s.arch.act = parseAct(wf.str("model.activation"));
  const std::string norm = wf.str("model.norm");
  if (norm != "layernorm" && norm != "batchnorm" && norm != "none" && norm != "identity")
    throw std::runtime_error("vbfnet: unsupported norm '" + norm + "'");
  s.arch.stride = 3 + static_cast<int>(wf.integer("model.dropout_layers"));
  s.arch.input_bn = wf.integer("model.input_batchnorm") != 0;
  s.arch.pair = wf.integer("model.use_edge_pair_summary") != 0;
  s.arch.ln_eps = std::stof(wf.str("model.layernorm_eps"));
  s.arch.bn_eps = std::stof(wf.str("model.batchnorm_eps"));

  const auto& acc = wf.one("acceptance");
  if (acc.size() != 3) throw std::runtime_error("vbfnet: malformed acceptance record");
  s.acceptance.min_jets = std::stoi(acc[0]);
  s.acceptance.jet_min_pt = std::stod(acc[1]);
  s.acceptance.jet_max_abs_eta = std::stod(acc[2]);

  const int n_out = static_cast<int>(s.targets.size() * s.heads.size());
  std::ostringstream desc;
  desc << "set " << s.name << " (" << wf.str("package_version") << ", " << weight_file << "): " << s.n_folds
       << " members, route member = event % " << s.n_folds << ", targets " << join(s.targets) << "\n";
  std::vector<bool> seen(static_cast<std::size_t>(s.n_folds), false);
  auto members = wf.meta.equal_range("member");
  for (auto it = members.first; it != members.second; ++it) {
    const int k = std::stoi(it->second.at(0));
    if (k < 0 || k >= s.n_folds || seen[k]) throw std::runtime_error("vbfnet: bad member record");
    seen[k] = true;
    desc << "  fold " << k << ": " << it->second.at(2) << " sha256 " << it->second.at(1) << "\n";
  }
  for (int k = 0; k < s.n_folds; ++k)
    if (!seen[k]) throw std::runtime_error("vbfnet: " + weight_file + " has no member for fold " + std::to_string(k));
  for (int k = 0; k < s.n_folds; ++k)
    s.members.push_back(loadMember(wf, "member" + std::to_string(k) + ".", s.arch,
                                   static_cast<int>(s.features.node.size()), static_cast<int>(s.features.edge.size()),
                                   static_cast<int>(s.features.global.size()), n_out));

  s.has_calibration = wf.str("calibration") == "per_member";
  if (s.has_calibration) {
    if (s.i50 < 0) throw std::runtime_error("vbfnet: calibration needs a q50 head");
    s.calib.resize(static_cast<std::size_t>(s.n_folds));
    for (int k = 0; k < s.n_folds; ++k) {
      for (const auto& t : s.targets) {
        std::vector<Binning> per_head;
        for (int h = 0; h < s.n_q; ++h) {
          const std::string base = "calib" + std::to_string(k) + "." + t + "." + s.heads[h];
          Binning b{wf.get(base + ".edges").d, wf.get(base + ".content").d};
          if (b.edges.size() != b.content.size() + 1 || b.content.empty())
            throw std::runtime_error("vbfnet: malformed calibration " + base);
          per_head.push_back(std::move(b));
        }
        s.calib[k].push_back(std::move(per_head));
      }
    }
    desc << "  calibration: per member, " << s.n_q << " quantile heads per target\n";
  } else {
    desc << "  calibration: none\n";
  }

  // Every tensor must have been used: an unknown module means the file was made
  // from an architecture this code does not implement.
  for (const auto& kv : wf.tensors)
    if (!wf.used.count(kv.first))
      throw std::runtime_error("vbfnet: " + weight_file + ": tensor '" + kv.first +
                               "' is not used by this library (newer model architecture?)");
  s.description = desc.str();
}

ModelSet::~ModelSet() = default;
ModelSet::ModelSet(ModelSet&&) noexcept = default;
ModelSet& ModelSet::operator=(ModelSet&&) noexcept = default;

const std::string& ModelSet::name() const { return impl_->name; }
const std::string& ModelSet::file() const { return impl_->file; }
int ModelSet::nFolds() const { return impl_->n_folds; }
int ModelSet::route(std::uint64_t event) const {
  return static_cast<int>(event % static_cast<std::uint64_t>(impl_->n_folds));
}
const std::vector<std::string>& ModelSet::targetKeys() const { return impl_->targets; }
const std::vector<std::string>& ModelSet::decodeRules() const { return impl_->decode_names; }
const std::vector<std::string>& ModelSet::headNames() const { return impl_->heads; }
const FeatureSpec& ModelSet::features() const { return impl_->features; }
const Acceptance& ModelSet::defaultAcceptance() const { return impl_->acceptance; }
bool ModelSet::hasCalibration() const { return impl_->has_calibration; }
std::string ModelSet::describe() const { return impl_->description; }
double ModelSet::decode(int t, double z) const { return decodeValue(impl_->rules.at(static_cast<std::size_t>(t)), z); }

std::vector<float> ModelSet::forward(const Graph& g, int member) const {
  if (member < 0 || member >= impl_->n_folds) throw std::out_of_range("vbfnet: no member " + std::to_string(member));
  if (g.x.size() != static_cast<std::size_t>(g.n_nodes) * impl_->features.node.size() ||
      g.u.size() != impl_->features.global.size() ||
      g.edge_attr.size() != static_cast<std::size_t>(g.n_edges()) * impl_->features.edge.size())
    throw std::invalid_argument("vbfnet: graph features do not match set " + impl_->name);
  return impl_->forward(g, member);
}

namespace {

std::vector<float> linear(const Linear& L, const float* X, int M) {
  std::vector<float> y(static_cast<std::size_t>(M) * L.out);
  L.forward(X, M, y.data());
  return y;
}

// z[i] += src_part[src[i]] + dst_part[dst[i]] + u_part, for every edge i.
void addGathered(std::vector<float>& z, int width, const Graph& g, const std::vector<float>& src_part,
                 const std::vector<float>& dst_part, const std::vector<float>& u_part) {
  for (int i = 0; i < g.n_edges(); ++i) {
    float* row = z.data() + static_cast<std::size_t>(i) * width;
    const float* s = src_part.data() + static_cast<std::size_t>(g.src[i]) * width;
    const float* d = dst_part.data() + static_cast<std::size_t>(g.dst[i]) * width;
    for (int j = 0; j < width; ++j) row[j] += s[j] + d[j] + u_part[j];
  }
}

}  // namespace

// PyGVBFGNN.forward for one graph (batch of 1), EdgeConvWithAttr.forward per layer.
std::vector<float> ModelSet::Impl::forward(const Graph& g, int k) const {
  const Member& M = members[static_cast<std::size_t>(k)];
  const int n = g.n_nodes, E = g.n_edges();
  if (n < 1 || E < 1) throw std::invalid_argument("vbfnet: the graph has no edges");

  std::vector<float> x = g.x, ea = g.edge_attr, u0 = g.u;
  M.node_in.apply(x.data(), n);
  M.edge_in.apply(ea.data(), E);
  M.global_in.apply(u0.data(), 1);
  std::vector<float> h = M.node_enc.forward(x.data(), n);
  std::vector<float> e = M.edge_enc.forward(ea.data(), E);
  std::vector<float> u = M.global_enc.forward(u0.data(), 1);
  const int dn = M.node_enc.out(), de = M.edge_enc.out(), dg = M.global_enc.out();

  // in-degree of every node (for the mean aggregation; >= 1 in a complete graph)
  std::vector<int> deg(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < E; ++i) ++deg[g.dst[i]];

  std::vector<float> buf;
  for (const EdgeConv& c : M.layers) {
    // edge update: edge_norm(edge_mlp([e, x_src, x_dst, u]) + e); the first
    // layer as W_e e + W_s x_src + W_d x_dst + (W_u u + b)
    const std::vector<float> ps = linear(c.edge_src, h.data(), n), pd = linear(c.edge_dst, h.data(), n);
    const std::vector<float> pu = linear(c.edge_u, u.data(), 1);
    std::vector<float> z = linear(c.edge_e, e.data(), E);
    addGathered(z, c.edge_e.out, g, ps, pd, pu);
    std::vector<float> e_new = c.edge_mlp.forwardRest(std::move(z), E);
    for (std::size_t i = 0; i < e_new.size(); ++i) e_new[i] += e[i];
    c.edge_norm.apply(e_new.data(), E);

    // gate and message share their input [x_dst, x_src, e_new, u] (x_i, x_j in
    // PyG); their first layers are stacked: columns [0, hg) gate, [hg, hg + hm) message
    const int hg = c.gate_mlp.lin[0].out, hm = c.msg_mlp.lin[0].out;
    const std::vector<float> qs = linear(c.gm_src, h.data(), n), qd = linear(c.gm_dst, h.data(), n);
    const std::vector<float> qu = linear(c.gm_u, u.data(), 1);
    std::vector<float> zgm = linear(c.gm_e, e_new.data(), E);
    addGathered(zgm, hg + hm, g, qs, qd, qu);
    std::vector<float> zg(static_cast<std::size_t>(E) * hg), zm(static_cast<std::size_t>(E) * hm);
    for (int i = 0; i < E; ++i) {
      const float* row = zgm.data() + static_cast<std::size_t>(i) * (hg + hm);
      std::copy_n(row, hg, zg.data() + static_cast<std::size_t>(i) * hg);
      std::copy_n(row + hg, hm, zm.data() + static_cast<std::size_t>(i) * hm);
    }
    std::vector<float> gate = c.gate_mlp.forwardRest(std::move(zg), E);
    std::vector<float> msg = c.msg_mlp.forwardRest(std::move(zm), E);
    const int dm = c.msg_mlp.out();
    for (int i = 0; i < E; ++i) {
      const float s = 1.0f / (1.0f + std::exp(-gate[i]));
      float* mrow = msg.data() + static_cast<std::size_t>(i) * dm;
      for (int j = 0; j < dm; ++j) mrow[j] = s * mrow[j];
    }

    // aggregation at the destination node, in the configured order (MultiAggregation, mode "cat")
    const int n_aggr = static_cast<int>(arch.aggregation.size());
    const int w3 = dn + n_aggr * dm + dg;
    std::vector<float> upd_in(static_cast<std::size_t>(n) * w3, 0.0f);
    for (int a = 0; a < n_aggr; ++a) {
      const std::string& kind = arch.aggregation[a];
      const int off = dn + a * dm;
      if (kind == "max") {
        std::vector<char> any(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < E; ++i) {
          float* out = upd_in.data() + static_cast<std::size_t>(g.dst[i]) * w3 + off;
          const float* mrow = msg.data() + static_cast<std::size_t>(i) * dm;
          if (!any[g.dst[i]]) {
            std::copy_n(mrow, dm, out);
            any[g.dst[i]] = 1;
          } else {
            for (int j = 0; j < dm; ++j) out[j] = std::max(out[j], mrow[j]);
          }
        }
      } else {  // sum / mean
        for (int i = 0; i < E; ++i) {
          float* out = upd_in.data() + static_cast<std::size_t>(g.dst[i]) * w3 + off;
          const float* mrow = msg.data() + static_cast<std::size_t>(i) * dm;
          for (int j = 0; j < dm; ++j) out[j] += mrow[j];
        }
        if (kind == "mean")
          for (int v = 0; v < n; ++v) {
            float* out = upd_in.data() + static_cast<std::size_t>(v) * w3 + off;
            const float cnt = static_cast<float>(std::max(deg[v], 1));
            for (int j = 0; j < dm; ++j) out[j] /= cnt;
          }
      }
    }
    for (int v = 0; v < n; ++v) {
      float* row = upd_in.data() + static_cast<std::size_t>(v) * w3;
      std::copy_n(h.data() + static_cast<std::size_t>(v) * dn, dn, row);
      std::copy_n(u.data(), dg, row + dn + n_aggr * dm);
    }

    // node update: upd_norm(upd_mlp([x, agg, u]) + res_proj(x))
    std::vector<float> h_new = c.upd_mlp.forward(upd_in.data(), n);
    std::vector<float> res(static_cast<std::size_t>(n) * dn);
    c.res_proj.forward(h.data(), n, res.data());
    for (std::size_t i = 0; i < h_new.size(); ++i) h_new[i] += res[i];
    c.upd_norm.apply(h_new.data(), n);

    // global update: global_norm(global_res(u) + global_mlp([u, mean(x), max(x), mean(e), max(e)]))
    const int w4 = dg + 2 * dn + 2 * de;
    std::vector<float> gin(static_cast<std::size_t>(w4), 0.0f);
    std::copy_n(u.data(), dg, gin.data());
    float* nmean = gin.data() + dg;
    float* nmax = nmean + dn;
    float* emean = nmax + dn;
    float* emax = emean + de;
    std::copy_n(h_new.data(), dn, nmax);
    for (int v = 0; v < n; ++v) {
      const float* row = h_new.data() + static_cast<std::size_t>(v) * dn;
      for (int j = 0; j < dn; ++j) {
        nmean[j] += row[j];
        if (v) nmax[j] = std::max(nmax[j], row[j]);
      }
    }
    for (int j = 0; j < dn; ++j) nmean[j] /= static_cast<float>(n);
    std::copy_n(e_new.data(), de, emax);
    for (int i = 0; i < E; ++i) {
      const float* row = e_new.data() + static_cast<std::size_t>(i) * de;
      for (int j = 0; j < de; ++j) {
        emean[j] += row[j];
        if (i) emax[j] = std::max(emax[j], row[j]);
      }
    }
    for (int j = 0; j < de; ++j) emean[j] /= static_cast<float>(E);
    std::vector<float> u_new = c.global_mlp.forward(gin.data(), 1);
    std::vector<float> ures(static_cast<std::size_t>(dg));
    c.global_res.forward(u.data(), 1, ures.data());
    for (int j = 0; j < dg; ++j) u_new[j] = ures[j] + u_new[j];
    c.global_norm.apply(u_new.data(), 1);

    h.swap(h_new);
    e.swap(e_new);
    u.swap(u_new);
  }

  // readout: [pool(x), u, attention-pooled edges] -> regressor
  std::vector<float> combined;
  std::vector<float> mean(static_cast<std::size_t>(dn), 0.0f), mx(h.begin(), h.begin() + dn);
  for (int v = 0; v < n; ++v)
    for (int j = 0; j < dn; ++j) {
      const float val = h[static_cast<std::size_t>(v) * dn + j];
      mean[j] += val;
      if (v) mx[j] = std::max(mx[j], val);
    }
  for (int j = 0; j < dn; ++j) mean[j] /= static_cast<float>(n);
  if (arch.pool != "max") combined.insert(combined.end(), mean.begin(), mean.end());
  if (arch.pool != "mean") combined.insert(combined.end(), mx.begin(), mx.end());
  combined.insert(combined.end(), u.begin(), u.end());

  if (M.pair) {
    const int ws = de + dg;
    buf.resize(static_cast<std::size_t>(E) * ws);
    for (int i = 0; i < E; ++i) {
      float* row = buf.data() + static_cast<std::size_t>(i) * ws;
      std::copy_n(e.data() + static_cast<std::size_t>(i) * de, de, row);
      std::copy_n(u.data(), dg, row + de);
    }
    std::vector<float> logits = M.pair_score.forward(buf.data(), E);
    // torch_geometric.utils.softmax over the edges of the graph
    float lmax = logits[0];
    for (int i = 1; i < E; ++i) lmax = std::max(lmax, logits[i]);
    float lsum = 0.0f;
    for (int i = 0; i < E; ++i) {
      logits[i] = std::exp(logits[i] - lmax);
      lsum += logits[i];
    }
    lsum += 1e-16f;
    std::vector<float> pf = M.pair_proj.forward(e.data(), E);
    const int dp = M.pair_proj.out();
    std::vector<float> z(static_cast<std::size_t>(dp), 0.0f);
    for (int i = 0; i < E; ++i) {
      const float alpha = logits[i] / lsum;
      const float* row = pf.data() + static_cast<std::size_t>(i) * dp;
      for (int j = 0; j < dp; ++j) z[j] += alpha * row[j];
    }
    combined.insert(combined.end(), z.begin(), z.end());
  }

  std::vector<float> raw = M.regressor.forward(combined.data(), 1);
  // sort the quantile heads of each target (torch.sort); the point head stays last
  const std::size_t nh = heads.size();
  for (std::size_t t = 0; t < targets.size(); ++t) {
    float* q = raw.data() + t * nh;
    std::sort(q, q + n_q, [](float a, float b) { return std::isnan(b) ? !std::isnan(a) : a < b; });
  }
  return raw;
}

std::vector<double> ModelSet::calibrateLog(const std::vector<double>& log, int member) const {
  const Impl& s = *impl_;
  if (!s.has_calibration) throw std::logic_error("vbfnet: set " + s.name + " has no calibration");
  const std::size_t nt = s.targets.size(), nh = s.heads.size();
  if (log.size() != nt * nh) throw std::invalid_argument("vbfnet: calibrateLog: wrong size");
  const auto& tables = s.calib.at(static_cast<std::size_t>(member));
  std::vector<double> out = log;
  std::vector<double> cal(static_cast<std::size_t>(s.n_q));
  for (std::size_t t = 0; t < nt; ++t) {
    const Rule rule = s.rules[t];
    const double raw_q50 = decodeValue(rule, log[t * nh + s.i50]);
    for (int j = 0; j < s.n_q; ++j)
      cal[j] = decodeValue(rule, log[t * nh + j]) + tables[t][j].lookup(raw_q50);
    sortNanLast(cal.data(), s.n_q);
    for (int j = 0; j < s.n_q; ++j) out[t * nh + j] = encodeValue(rule, cal[j]);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// VBFNet: several sets, merged predictions (unified.py, transforms.py)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// transforms.P4_TARGET_KEYS / P4_DERIVED_KEYS
const std::vector<std::string> kP4Targets = {"q1_E", "q1_px", "q1_py", "q1_pz", "q2_E", "q2_px", "q2_py", "q2_pz"};
const std::vector<std::string> kP4Derived = {"q1_pt",  "q1_eta", "q1_phi",      "q1_mass",
                                             "q2_pt",  "q2_eta", "q2_phi",      "q2_mass",
                                             "mjj_p4", "deta_p4", "eta_prod_p4", "ptsum_p4"};

}  // namespace

struct Layout {
  std::vector<std::string> keys, targets, heads, owner;  // owner: set name per key
  std::unordered_map<std::string, int> key_index, head_index;
  struct SetInfo {
    int target_offset = 0;   // first target (= key) index of the set
    bool derived = false;    // the set provides the p4-derived keys
    int derived_offset = 0;  // key index of the first derived key
    std::array<int, 8> p4{}; // key index of q1_E ... q2_pz
  };
  std::vector<SetInfo> sets;

  int keyIndex(const std::string& key) const {
    auto it = key_index.find(key);
    if (it == key_index.end())
      throw std::out_of_range("vbfnet: unknown key '" + key + "'. Keys: " + join(keys, ", "));
    return it->second;
  }
  int headIndex(const std::string& head) const {
    auto it = head_index.find(head);
    if (it == head_index.end())
      throw std::out_of_range("vbfnet: unknown head '" + head + "'. Heads: " + join(heads, ", "));
    return it->second;
  }
};

VBFNet::VBFNet(const std::vector<std::string>& weight_files) { init(weight_files); }

VBFNet::VBFNet(const std::vector<std::string>& weight_files, const Options& options) : options_(options) {
  init(weight_files);
}

VBFNet::~VBFNet() = default;

std::vector<std::string> VBFNet::releaseFiles(const std::string& repo_dir, const std::vector<std::string>& sets) {
  const std::vector<std::string> names = sets.empty() ? std::vector<std::string>{"p4", "hl"} : sets;
  std::vector<std::string> files;
  for (const auto& s : names) files.push_back(repo_dir + "/ensembles/" + s + "/cpp/vbfnet_" + s + ".bin");
  return files;
}

void VBFNet::init(const std::vector<std::string>& weight_files) {
  if (weight_files.empty()) throw std::invalid_argument("vbfnet: no weight files given");
  for (const auto& f : weight_files) sets_.push_back(std::make_shared<const ModelSet>(f));
  // p4 first, then by name: the order of manifest.available_ensembles
  std::stable_sort(sets_.begin(), sets_.end(), [](const auto& a, const auto& b) {
    const bool ap = a->name() == "p4", bp = b->name() == "p4";
    return ap != bp ? ap : a->name() < b->name();
  });

  const ModelSet& ref = *sets_.front();
  for (const auto& s : sets_) {
    if (s.get() != &ref && s->name() == ref.name()) throw std::invalid_argument("vbfnet: set " + s->name() + " given twice");
    if (s->features().node != ref.features().node || s->features().edge != ref.features().edge ||
        s->features().global != ref.features().global)
      throw std::invalid_argument("vbfnet: sets " + ref.name() + " and " + s->name() +
                                  " read different features; they cannot share one graph");
    if (s->headNames() != ref.headNames())
      throw std::invalid_argument("vbfnet: sets " + ref.name() + " and " + s->name() + " have different heads");
    if (s->nFolds() != ref.nFolds())
      throw std::invalid_argument("vbfnet: sets " + ref.name() + " and " + s->name() + " have different n_folds");
    const Acceptance& a = s->defaultAcceptance();
    const Acceptance& r = ref.defaultAcceptance();
    if (a.min_jets != r.min_jets || a.jet_min_pt != r.jet_min_pt || a.jet_max_abs_eta != r.jet_max_abs_eta)
      throw std::invalid_argument("vbfnet: sets " + ref.name() + " and " + s->name() + " have different acceptance");
    if (options_.calibrate && !s->hasCalibration())
      throw std::invalid_argument("vbfnet: calibration requested but set " + s->name() + " has none");
  }
  acceptance_ = options_.override_acceptance ? options_.acceptance : ref.defaultAcceptance();

  auto layout = std::make_shared<Layout>();
  layout->heads = ref.headNames();
  for (std::size_t h = 0; h < layout->heads.size(); ++h) layout->head_index[layout->heads[h]] = static_cast<int>(h);
  auto add_key = [&](const std::string& key, const std::string& owner) {
    if (layout->key_index.count(key))
      throw std::invalid_argument("vbfnet: key '" + key + "' is provided by sets " + layout->owner[layout->key_index[key]] +
                                  " and " + owner + "; load only one of them");
    layout->key_index[key] = static_cast<int>(layout->keys.size());
    layout->keys.push_back(key);
    layout->owner.push_back(owner);
  };
  for (const auto& s : sets_) {
    Layout::SetInfo info;
    info.target_offset = static_cast<int>(layout->keys.size());
    for (const auto& t : s->targetKeys()) {
      add_key(t, s->name());
      layout->targets.push_back(t);
    }
    info.derived = std::all_of(kP4Targets.begin(), kP4Targets.end(), [&](const std::string& k) {
      return std::find(s->targetKeys().begin(), s->targetKeys().end(), k) != s->targetKeys().end();
    });
    layout->sets.push_back(info);
  }
  for (std::size_t i = 0; i < sets_.size(); ++i) {
    Layout::SetInfo& info = layout->sets[i];
    if (!info.derived) continue;
    for (int c = 0; c < 8; ++c) info.p4[c] = layout->key_index.at(kP4Targets[c]);
    info.derived_offset = static_cast<int>(layout->keys.size());
    for (const auto& k : kP4Derived) add_key(k, sets_[i]->name());
  }
  layout_ = layout;

  if (options_.verbose) {
    std::cout << "[vbfnet] loaded " << sets_.size() << " model set(s), route: member = event % " << nFolds()
              << ", calibration " << (options_.calibrate ? "on" : "off") << ", acceptance: " << acceptance_.str()
              << "\n";
    for (const auto& s : sets_) std::cout << "[vbfnet] " << s->describe();
    std::cout << std::flush;
  }
}

int VBFNet::nFolds() const { return sets_.front()->nFolds(); }
const std::vector<std::string>& VBFNet::keys() const { return layout_->keys; }
const std::vector<std::string>& VBFNet::targetKeys() const { return layout_->targets; }
const std::vector<std::string>& VBFNet::headNames() const { return layout_->heads; }
int VBFNet::keyIndex(const std::string& key) const { return layout_->keyIndex(key); }
int VBFNet::headIndex(const std::string& head) const { return layout_->headIndex(head); }
const std::string& VBFNet::ensembleOf(const std::string& key) const {
  return layout_->owner[static_cast<std::size_t>(layout_->keyIndex(key))];
}

namespace {

// transforms._derive_vbf_observables_from_p4, one head of one event.
void deriveP4(const double* phys, int nh, const Layout::SetInfo& info, double* out_base, int h) {
  auto val = [&](int c) { return phys[static_cast<std::size_t>(info.p4[c]) * nh + h]; };
  auto put = [&](int d, double v) { out_base[static_cast<std::size_t>(info.derived_offset + d) * nh + h] = v; };
  const double E1 = val(0), px1 = val(1), py1 = val(2), pz1 = val(3);
  const double E2 = val(4), px2 = val(5), py2 = val(6), pz2 = val(7);
  const double pt1 = std::sqrt(px1 * px1 + py1 * py1), pt2 = std::sqrt(px2 * px2 + py2 * py2);
  const double eta1 = std::asinh(pz1 / std::max(pt1, 1e-9)), eta2 = std::asinh(pz2 / std::max(pt2, 1e-9));
  auto mass = [](double E, double px, double py, double pz) {
    const double m2 = E * E - px * px - py * py - pz * pz;
    return std::sqrt(std::max(m2, 0.0));
  };
  put(0, pt1);
  put(1, eta1);
  put(2, std::atan2(py1, px1));
  put(3, mass(E1, px1, py1, pz1));
  put(4, pt2);
  put(5, eta2);
  put(6, std::atan2(py2, px2));
  put(7, mass(E2, px2, py2, pz2));
  put(8, mass(E1 + E2, px1 + px2, py1 + py2, pz1 + pz2));
  put(9, std::fabs(eta1 - eta2));
  put(10, eta1 * eta2);
  put(11, pt1 + pt2);
}

}  // namespace

Prediction VBFNet::predict(const Event& ev) const {
  const Layout& L = *layout_;
  const std::size_t nk = L.keys.size(), nh = L.heads.size(), nt = L.targets.size();
  Prediction p;
  p.layout_ = layout_;
  p.calibrated_ = options_.calibrate;
  p.phys_.assign(nk * nh, kNaN);
  p.log_.assign(nt * nh, kNaN);
  if (options_.calibrate) {
    p.phys_cal_.assign(nk * nh, kNaN);
    p.log_cal_.assign(nt * nh, kNaN);
  }
  if (ev.jets.size() < 2 || !acceptance_.passes(ev.jets)) return p;

  p.accepted_ = true;
  p.fold_ = sets_.front()->route(ev.event);
  const Graph g = buildGraphIdx(ev, sets_.front()->impl_->feature_index);
  const int n_q = sets_.front()->impl_->n_q;

  for (std::size_t si = 0; si < sets_.size(); ++si) {
    const ModelSet& s = *sets_[si];
    const Layout::SetInfo& info = L.sets[si];
    const std::size_t st = s.targetKeys().size();
    const std::vector<float> raw = s.impl_->forward(g, p.fold_);
    std::vector<double> log(raw.begin(), raw.end());
    for (std::size_t t = 0; t < st; ++t) sortNanLast(log.data() + t * nh, n_q);  // routing.sort_quantile_heads
    std::copy(log.begin(), log.end(), p.log_.begin() + static_cast<std::ptrdiff_t>(info.target_offset * nh));
    for (std::size_t t = 0; t < st; ++t)
      for (std::size_t h = 0; h < nh; ++h)
        p.phys_[(info.target_offset + t) * nh + h] = s.decode(static_cast<int>(t), log[t * nh + h]);
    if (options_.calibrate) {
      std::vector<double> cal = s.calibrateLog(log, p.fold_);
      for (std::size_t t = 0; t < st; ++t) sortNanLast(cal.data() + t * nh, n_q);
      std::copy(cal.begin(), cal.end(), p.log_cal_.begin() + static_cast<std::ptrdiff_t>(info.target_offset * nh));
      for (std::size_t t = 0; t < st; ++t)
        for (std::size_t h = 0; h < nh; ++h)
          p.phys_cal_[(info.target_offset + t) * nh + h] = s.decode(static_cast<int>(t), cal[t * nh + h]);
    }
    if (info.derived) {
      for (std::size_t h = 0; h < nh; ++h) {
        deriveP4(p.phys_.data(), static_cast<int>(nh), info, p.phys_.data(), static_cast<int>(h));
        if (options_.calibrate)
          deriveP4(p.phys_cal_.data(), static_cast<int>(nh), info, p.phys_cal_.data(), static_cast<int>(h));
      }
    }
  }
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction accessors
// ─────────────────────────────────────────────────────────────────────────────

namespace {
const std::vector<double>& requireCal(const std::vector<double>& v, bool calibrated) {
  if (calibrated && v.empty())
    throw std::logic_error("vbfnet: calibrated values were not computed; set VBFNet::Options::calibrate = true");
  return v;
}
}  // namespace

double Prediction::get(const std::string& key, const std::string& head, bool calibrated) const {
  if (!layout_) throw std::logic_error("vbfnet: empty Prediction");
  return value(layout_->keyIndex(key), layout_->headIndex(head), calibrated);
}

double Prediction::value(int key_index, int head_index, bool calibrated) const {
  const std::vector<double>& v = requireCal(calibrated ? phys_cal_ : phys_, calibrated);
  return v.at(static_cast<std::size_t>(key_index) * layout_->heads.size() + static_cast<std::size_t>(head_index));
}

double Prediction::logValue(int target_index, int head_index, bool calibrated) const {
  const std::vector<double>& v = requireCal(calibrated ? log_cal_ : log_, calibrated);
  return v.at(static_cast<std::size_t>(target_index) * layout_->heads.size() + static_cast<std::size_t>(head_index));
}

const std::vector<double>& Prediction::values(bool calibrated) const {
  return requireCal(calibrated ? phys_cal_ : phys_, calibrated);
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch names
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<std::string>& logicalBranches() {
  static const std::vector<std::string> v = {
      "event",         "nVBFJet",        "VBFJet_pt",   "VBFJet_eta",  "VBFJet_phi",
      "VBFJet_mass",   "VBFJet_btagDeepFlavB", "VBFJet_btagDeepFlavCvB", "VBFJet_btagDeepFlavCvL",
      "VBFJet_btagDeepFlavQG", "VBFJet_nConstituents", "b1_pt", "b1_eta", "b1_phi", "b1_mass",
      "b2_pt",         "b2_eta",         "b2_phi",      "b2_mass",     "tau1_pt",
      "tau1_eta",      "tau1_phi",       "tau1_mass",   "tau2_pt",     "tau2_eta",
      "tau2_phi",      "tau2_mass",      "met_pt",      "met_phi"};
  return v;
}

std::map<std::string, std::string> loadBranchMap(const std::string& yaml_file) {
  std::ifstream in(yaml_file);
  if (!in) throw std::runtime_error("vbfnet: cannot open branch map " + yaml_file);
  auto trim = [](std::string s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return std::string();
    const auto e = s.find_last_not_of(" \t\r");
    s = s.substr(b, e - b + 1);
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front()) s = s.substr(1, s.size() - 2);
    return s;
  };
  std::map<std::string, std::string> out;
  int lineno = 0;
  for (std::string line; std::getline(in, line);) {
    ++lineno;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    if (trim(line).empty()) continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("vbfnet: " + yaml_file + ":" + std::to_string(lineno) + ": expected 'name: branch'");
    const std::string key = trim(line.substr(0, colon)), value = trim(line.substr(colon + 1));
    if (value.empty()) {
      if (key != "branch_map")
        throw std::runtime_error("vbfnet: " + yaml_file + ":" + std::to_string(lineno) + ": no branch for '" + key + "'");
      continue;
    }
    const auto& known = logicalBranches();
    if (std::find(known.begin(), known.end(), key) == known.end() && key != "run" && key != "luminosityBlock")
      throw std::runtime_error("vbfnet: " + yaml_file + ":" + std::to_string(lineno) + ": unknown model branch '" +
                               key + "'");
    out[key] = value;
  }
  return out;
}

std::string defineExpression(const std::string& net_pointer, const std::map<std::string, std::string>& branch_map) {
  auto b = [&](const char* logical) {
    auto it = branch_map.find(logical);
    return it == branch_map.end() ? std::string(logical) : it->second;
  };
  auto p4 = [&](const std::string& o) {
    return "vbfnet::PtEtaPhiM{double(" + b((o + "_pt").c_str()) + "), double(" + b((o + "_eta").c_str()) +
           "), double(" + b((o + "_phi").c_str()) + "), double(" + b((o + "_mass").c_str()) + ")}";
  };
  std::string jets = "vbfnet::makeJets(" + b("nVBFJet");
  for (const char* j : {"VBFJet_pt", "VBFJet_eta", "VBFJet_phi", "VBFJet_mass", "VBFJet_btagDeepFlavB",
                        "VBFJet_btagDeepFlavCvB", "VBFJet_btagDeepFlavCvL", "VBFJet_btagDeepFlavQG",
                        "VBFJet_nConstituents"})
    jets += ", " + b(j);
  jets += ")";
  return "(" + net_pointer + ")->predict(vbfnet::makeEvent(" + b("event") + ", " + jets + ", " + p4("b1") + ", " +
         p4("b2") + ", " + p4("tau1") + ", " + p4("tau2") + ", " + b("met_pt") + ", " + b("met_phi") + "))";
}

}  // namespace vbfnet
