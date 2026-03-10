#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "lattice.hpp"
#include "observables.hpp"
#include "rng.hpp"
#include "update_metropolis.hpp"

namespace fs = std::filesystem;

int main() {
    const int L = 8;
    const double beta = 5.0;
    const double proposal_size = 0.20;

    const int n_sweeps = 10000;
    const int thermal_cut = 1000;
    const int measure_every = 20;

    const int R_max = 3;
    const int T_max = 4;

    Lattice3D lat(L, L, L);
    lat.cold_start();

    RNG rng(123456789ull);

    fs::path out_dir = fs::path("../../runs") / "large_loops_L8_beta_5p0";
    fs::create_directories(out_dir);

    std::ofstream out_plaq(out_dir / "plaquette.csv");
    std::ofstream out_wloop(out_dir / "wilson_loops.csv");

    if (!out_plaq || !out_wloop) {
        std::cerr << "Could not open output files in " << out_dir << "\n";
        return 1;
    }

    out_plaq << "sweep,acceptance,plaquette\n";
    out_wloop << "sweep,R,T,W\n";

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Larger-loop run\n";
    std::cout << "L = " << L
              << ", beta = " << beta
              << ", proposal_size = " << proposal_size
              << ", n_sweeps = " << n_sweeps
              << ", thermal_cut = " << thermal_cut
              << ", measure_every = " << measure_every << "\n";
    std::cout << "Measuring R = 1.." << R_max
              << ", T = 1.." << T_max << "\n";
    std::cout << "Initial plaquette = " << obs::average_plaquette(lat) << "\n";

    for (int sweep = 1; sweep <= n_sweeps; ++sweep) {
        double acc = metro::metropolis_sweep(lat, beta, proposal_size, rng);
        double plaq = obs::average_plaquette(lat);

        out_plaq << sweep << "," << acc << "," << plaq << "\n";

        if (sweep > thermal_cut && sweep % measure_every == 0) {
            for (int R = 1; R <= R_max; ++R) {
                for (int T = 1; T <= T_max; ++T) {
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

    return 0;
}
