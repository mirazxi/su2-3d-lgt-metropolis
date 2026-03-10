#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lattice.hpp"
#include "observables.hpp"
#include "rng.hpp"
#include "update_metropolis.hpp"

namespace fs = std::filesystem;

std::string case_label(int L, double beta) {
    std::ostringstream oss;
    oss << "L" << L << "_beta_" << std::fixed << std::setprecision(1) << beta;
    std::string s = oss.str();
    for (char& c : s) {
        if (c == '.') {
            c = 'p';
        }
    }
    return s;
}

void run_case(
    int L,
    double beta,
    std::uint64_t seed,
    double proposal_size,
    int n_sweeps,
    int thermal_cut,
    int measure_every
) {
    Lattice3D lat(L, L, L);
    lat.cold_start();

    RNG rng(seed);

    fs::path out_dir = fs::path("../../runs") / case_label(L, beta);
    fs::create_directories(out_dir);

    std::ofstream out_plaq(out_dir / "plaquette.csv");
    std::ofstream out_wloop(out_dir / "wilson_loops.csv");

    if (!out_plaq || !out_wloop) {
        std::cerr << "Could not open output files in " << out_dir << "\n";
        std::exit(1);
    }

    out_plaq << "sweep,acceptance,plaquette\n";
    out_wloop << "sweep,R,T,W\n";

    std::cout << "\n=== L = " << L << ", beta = " << beta << " ===\n";
    std::cout << "Initial plaquette = " << obs::average_plaquette(lat) << "\n";

    for (int sweep = 1; sweep <= n_sweeps; ++sweep) {
        double acc = metro::metropolis_sweep(lat, beta, proposal_size, rng);
        double plaq = obs::average_plaquette(lat);

        out_plaq << sweep << "," << acc << "," << plaq << "\n";

        if (sweep > thermal_cut && sweep % measure_every == 0) {
            for (int R = 1; R <= 2; ++R) {
                for (int T = 1; T <= 3; ++T) {
                    double w = obs::average_wilson_loop_spacetime(lat, R, T, 2);
                    out_wloop << sweep << "," << R << "," << T << "," << w << "\n";
                }
            }
        }

        if (sweep == 1 || sweep % 500 == 0) {
            std::cout << "sweep " << std::setw(5) << sweep
                      << "  acceptance = " << acc
                      << "  plaquette = " << plaq << "\n";
        }
    }

    std::cout << "Wrote " << (out_dir / "plaquette.csv") << "\n";
    std::cout << "Wrote " << (out_dir / "wilson_loops.csv") << "\n";
}

int main() {
    const std::vector<int> Ls = {4, 6};
    const std::vector<double> betas = {4.0, 5.0};

    const double proposal_size = 0.20;
    const int n_sweeps = 5000;
    const int thermal_cut = 500;
    const int measure_every = 10;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Volume scan run\n";
    std::cout << "proposal_size = " << proposal_size
              << ", n_sweeps = " << n_sweeps
              << ", thermal_cut = " << thermal_cut
              << ", measure_every = " << measure_every << "\n";

    std::uint64_t seed_base = 123456;

    for (std::size_t i = 0; i < Ls.size(); ++i) {
        for (std::size_t j = 0; j < betas.size(); ++j) {
            run_case(
                Ls[i],
                betas[j],
                seed_base + 10000 * static_cast<std::uint64_t>(i) + 1000 * static_cast<std::uint64_t>(j),
                proposal_size,
                n_sweeps,
                thermal_cut,
                measure_every
            );
        }
    }

    return 0;
}
