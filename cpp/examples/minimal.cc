// The C++ API in a plain event loop, no ROOT needed.
//
//   ./vbfnet_minimal /path/to/vbf-net          # the repository checkout
//
// Fill a vbfnet::Event from your analysis objects and call predict(). Here one
// event is filled by hand.

#include <cstdio>
#include <exception>
#include <iostream>

#include "vbfnet/VBFNet.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <vbf-net checkout> [p4|hl ...]\n";
    return 2;
  }
  try {
    vbfnet::VBFNet::Options options;
    options.calibrate = true;  // also compute the calibrated quantiles
    const std::vector<std::string> sets(argv + 2, argv + argc);  // default: p4 and hl
    const vbfnet::VBFNet net(vbfnet::VBFNet::releaseFiles(argv[1], sets), options);

    vbfnet::Event ev;
    ev.event = 123456789;  // the CMS event number: member 123456789 % 5 = 4 predicts it
    //            pt     eta    phi   mass  DeepJet B, CvB, CvL, QG  nConstituents
    ev.jets = {{182.4, 2.31, 0.42, 14.2, 0.021, 0.18, 0.09, 0.71, 31},
               {96.1, -2.87, -2.61, 9.8, 0.012, 0.22, 0.05, 0.64, 22},
               {64.3, 0.55, 1.90, 8.1, 0.050, 0.30, 0.12, 0.40, 25},
               {38.7, -4.10, 2.75, 6.3, 0.004, 0.41, 0.08, 0.83, 12}};
    ev.b1 = {121.0, 0.31, -1.24, 15.1};  // H->bb candidate jets
    ev.b2 = {58.2, -0.47, 2.20, 9.6};
    ev.tau1 = {64.8, 0.85, 1.42, 1.2};   // H->tautau visible legs
    ev.tau2 = {35.5, 1.62, 2.98, 0.1};
    ev.met_pt = 41.3;
    ev.met_phi = -0.37;

    const vbfnet::Prediction p = net.predict(ev);
    if (!p.accepted()) {
      std::cout << "event failed the acceptance gate (" << net.acceptance().str() << ")\n";
      return 0;
    }
    std::cout << "predicted by member " << p.fold() << " of each set\n\n";
    std::printf("%-12s %-4s %10s %10s %10s %10s   %s\n", "key", "set", "q16", "q50", "q84", "point",
                "(calibrated q16 / q84)");
    for (const std::string& key : net.keys()) {
      std::printf("%-12s %-4s %10.4g %10.4g %10.4g %10.4g   %.4g / %.4g\n", key.c_str(),
                  net.ensembleOf(key).c_str(), p.get(key, "q16"), p.get(key, "q50"), p.get(key, "q84"),
                  p.get(key, "point"), p.get(key, "q16", true), p.get(key, "q84", true));
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
