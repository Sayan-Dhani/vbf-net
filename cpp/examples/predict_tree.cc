// vbfnet_predict_tree: run VBF-Net on ROOT trees and write the predictions.
//
//   vbfnet_predict_tree --input sig.root --output sig_vbfnet.root --repo /path/to/vbf-net [options]
//
// It is RDataFrame code: the prediction column is the same expression you
// would Define in your own RDataFrame analysis (vbfnet::defineExpression), so
// any branch types work and a branch map renames inputs.
//
// The output tree has ONE ROW PER INPUT ENTRY: events that fail the
// acceptance gate get vbfnet_accepted = false and NaN values. Run
// single-threaded (the default) and the rows are in input order, so the file
// can be used as a friend tree of the input; with --threads N match rows by
// vbfnet_entry or (run, luminosityBlock, event).
//
// Branches: vbfnet_entry, run, luminosityBlock, event, vbfnet_accepted,
// vbfnet_fold, and vbfnet_<key>_<head> (+ _cal with --calibrate) for every key
// of the loaded sets; --model-space adds vbfnet_log_<target>_<head>[_cal].

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ROOT/RDataFrame.hxx>
#include <TInterpreter.h>
#include <TROOT.h>

#include "vbfnet/VBFNet.h"

namespace {

void usage(const char* prog) {
  std::cerr
      << "usage: " << prog << " --input FILE [--input FILE ...] --output FILE\n"
      << "         (--repo DIR [--sets p4,hl] | --weights FILE [--weights FILE ...])\n"
      << "         [--tree Events] [--out-tree vbfnet] [--calibrate] [--no-acceptance]\n"
      << "         [--branch-map YAML] [--keys k1,k2,...] [--max-entries N] [--threads N] [--model-space]\n";
}

std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> inputs, weights, sets, keys;
  std::string output, repo, tree = "Events", out_tree = "vbfnet", branch_map_file;
  bool calibrate = false, no_acceptance = false, model_space = false;
  long long max_entries = -1;
  int threads = 0;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--input") inputs.push_back(next());
      else if (a == "--output") output = next();
      else if (a == "--repo") repo = next();
      else if (a == "--sets") sets = splitComma(next());
      else if (a == "--weights") weights.push_back(next());
      else if (a == "--tree") tree = next();
      else if (a == "--out-tree") out_tree = next();
      else if (a == "--calibrate") calibrate = true;
      else if (a == "--no-acceptance") no_acceptance = true;
      else if (a == "--branch-map") branch_map_file = next();
      else if (a == "--keys") keys = splitComma(next());
      else if (a == "--max-entries") max_entries = std::stoll(next());
      else if (a == "--threads") threads = std::stoi(next());
      else if (a == "--model-space") model_space = true;
      else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
      else throw std::invalid_argument("unknown option " + a);
    }
    if (inputs.empty() || output.empty() || (repo.empty() == weights.empty()))
      throw std::invalid_argument("need --input, --output and one of --repo / --weights");
    if (max_entries >= 0 && threads > 0) throw std::invalid_argument("--max-entries needs single-threaded running");
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage(argv[0]);
    return 2;
  }

  try {
    if (!repo.empty()) weights = vbfnet::VBFNet::releaseFiles(repo, sets);
    vbfnet::VBFNet::Options options;
    options.calibrate = calibrate;
    if (no_acceptance) {
      options.override_acceptance = true;
      options.acceptance = vbfnet::Acceptance::none();
    }
    // The interpreter reaches the network through this pointer; it lives until main returns.
    static std::unique_ptr<const vbfnet::VBFNet> net;
    net = std::make_unique<const vbfnet::VBFNet>(weights, options);
    if (keys.empty()) keys = net->keys();
    for (const auto& k : keys) net->keyIndex(k);  // fail early on a typo

    const auto branch_map =
        branch_map_file.empty() ? std::map<std::string, std::string>{} : vbfnet::loadBranchMap(branch_map_file);
    auto col = [&](const std::string& logical) {
      auto it = branch_map.find(logical);
      return it == branch_map.end() ? logical : it->second;
    };

    if (threads > 0) ROOT::EnableImplicitMT(static_cast<unsigned>(threads));
#ifdef VBFNET_INCLUDE_DIR
    gInterpreter->AddIncludePath(VBFNET_INCLUDE_DIR);
#endif
    if (!gInterpreter->Declare("#include \"vbfnet/VBFNet.h\""))
      throw std::runtime_error("ROOT's interpreter could not include vbfnet/VBFNet.h; add its directory to "
                               "ROOT_INCLUDE_PATH");
    std::ostringstream decl;
    decl << "namespace vbfnet_tool { const vbfnet::VBFNet* net = reinterpret_cast<const vbfnet::VBFNet*>("
         << reinterpret_cast<std::uintptr_t>(net.get()) << "ULL); }";
    if (!gInterpreter->Declare(decl.str().c_str())) throw std::runtime_error("could not declare the network");

    ROOT::RDataFrame df(tree, inputs);
    ROOT::RDF::RNode node = df;
    if (max_entries >= 0) node = node.Range(static_cast<ULong64_t>(max_entries));

    std::vector<std::string> columns = {"vbfnet_entry", col("run"), col("luminosityBlock"), col("event"),
                                        "vbfnet_accepted", "vbfnet_fold"};
    node = node.Define("vbfnet_entry", "rdfentry_")
               .Define("vbfnet_pred", vbfnet::defineExpression("vbfnet_tool::net", branch_map))
               .Define("vbfnet_accepted", "vbfnet_pred.accepted()")
               .Define("vbfnet_fold", "vbfnet_pred.fold()");
    for (int cal = 0; cal <= (calibrate ? 1 : 0); ++cal) {
      const std::string sfx = cal ? "_cal" : "", flag = cal ? "true" : "false";
      for (const auto& k : keys)
        for (const auto& h : net->headNames()) {
          const std::string name = "vbfnet_" + k + "_" + h + sfx;
          node = node.Define(name, "vbfnet_pred.value(" + std::to_string(net->keyIndex(k)) + ", " +
                                       std::to_string(net->headIndex(h)) + ", " + flag + ")");
          columns.push_back(name);
        }
      if (model_space)
        for (std::size_t t = 0; t < net->targetKeys().size(); ++t)
          for (const auto& h : net->headNames()) {
            const std::string name = "vbfnet_log_" + net->targetKeys()[t] + "_" + h + sfx;
            node = node.Define(name, "vbfnet_pred.logValue(" + std::to_string(t) + ", " +
                                         std::to_string(net->headIndex(h)) + ", " + flag + ")");
            columns.push_back(name);
          }
    }

    auto n_accepted = node.Filter("vbfnet_accepted").Count();
    auto n_total = node.Count();
    node.Snapshot(out_tree, output, columns);
    std::cout << "[vbfnet] wrote " << *n_total << " rows (" << *n_accepted << " accepted) to " << output << ":"
              << out_tree << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
