#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "lattice.hpp"
#include "observables.hpp"
#include "rng.hpp"
#include "smearing.hpp"
#include "update_metropolis.hpp"

namespace fs = std::filesystem;

struct RunCase {
    int L;
    double beta;
};

std::string case_label(int L, double beta) {
    std::ostringstream oss;
    oss << "scaling_L" << L << "_beta_" << std::fixed << std::setprecision(1) << beta;
    std::string s = oss.str();
    for (char& c : s) {
        if (c == '.') c = 'p';
    }
    return s;
}

void write_run_info(
    const fs::path& out_dir,
    int L,
    double beta,
    double proposal_size,
    int n_sweeps,
    int thermal_cut,
    int measure_every,
    int n_ape,
    double alpha_ape,
    int R_max,
    int T_max
) {
    std::ofstream out(out_dir / "run_info.txt");
    out << std::fixed << std::setprecision(12);
    out << "L = " << L << "\n";
    out << "beta = " << beta << "\n";
    out << "proposal_size = " << proposal_size << "\n";
    out << "n_sweeps = " << n_sweeps << "\n";
    out << "thermal_cut = " << thermal_cut << "\n";
    out << "measure_every = " << measure_every << "\n";
    out << "APE_steps = " << n_ape << "\n";
    out << "APE_alpha = " << alpha_ape << "\n";
    out << "R_max = " << R_max << "\n";
    out << "T_max = " << T_max << "\n";
    out << "time_direction = 2\n";
    out << "raw_loops_file = wilson_loops_raw.csv\n";
    out << "ape_loops_file = wilson_loops_ape.csv\n";
}

void run_case(
    const RunCase& rc,
    std::uint64_t seed,
    double proposal_size,
    int n_sweeps,
    int thermal_cut,
    int measure_every,
    int n_ape,
    double alpha_ape
) {
    const int L = 22;
    const double beta = 6.5;

    const int R_max = std::min(5, L / 2);
    const int T_max = std::min(6, L / 2);

    Lattice3D lat(L, L, L);
    lat.cold_start();

    RNG rng(seed);

    fs::path out_dir = fs::path("../runs") / case_label(L, beta);
    fs::create_directories(out_dir);

    std::ofstream out_plaq(out_dir / "plaquette.csv");
    std::ofstream out_raw(out_dir / "wilson_loops_raw.csv");
    std::ofstream out_ape(out_dir / "wilson_loops_ape.csv");

    if (!out_plaq || !out_raw || !out_ape) {
        std::cerr << "Could not open output files in " << out_dir << "\n";
        std::exit(1);
    }

    out_plaq << "sweep,acceptance,plaquette\n";
    out_raw << "sweep,R,T,W\n";
    out_ape << "sweep,R,T,W\n";

    write_run_info(
        out_dir, L, beta, proposal_size, n_sweeps, thermal_cut, measure_every,
        n_ape, alpha_ape, R_max, T_max
    );

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\n=== " << case_label(L, beta) << " ===\n";
    std::cout << "Initial plaquette = " << obs::average_plaquette(lat) << "\n";
    std::cout << "R_max = " << R_max << ", T_max = " << T_max << "\n";

    for (int sweep = 1; sweep <= n_sweeps; ++sweep) {
        double acc = metro::metropolis_sweep(lat, beta, proposal_size, rng);
        double plaq = obs::average_plaquette(lat);

        out_plaq << sweep << "," << acc << "," << plaq << "\n";

        if (sweep > thermal_cut && sweep % measure_every == 0) {
            Lattice3D smeared = lat;
            smear::ape_smear_spatial_only(smeared, L, n_ape, alpha_ape);

            for (int R = 1; R <= R_max; ++R) {
                for (int T = 1; T <= T_max; ++T) {
                    double w_raw = obs::average_wilson_loop_spacetime(lat, R, T, 2);
                    double w_ape = obs::average_wilson_loop_spacetime(smeared, R, T, 2);

                    out_raw << sweep << "," << R << "," << T << "," << w_raw << "\n";
                    out_ape << sweep << "," << R << "," << T << "," << w_ape << "\n";
                }
            }
        }

        if (sweep == 1 || sweep % 1000 == 0) {
            std::cout << "sweep " << std::setw(6) << sweep
                      << "  acceptance = " << acc
                      << "  plaquette = " << plaq << "\n";
        }
    }

    std::cout << "Wrote " << (out_dir / "plaquette.csv") << "\n";
    std::cout << "Wrote " << (out_dir / "wilson_loops_raw.csv") << "\n";
    std::cout << "Wrote " << (out_dir / "wilson_loops_ape.csv") << "\n";
}

int main() {
    const std::vector<RunCase> cases = {
        {10, 4.5},
        {12, 5.0},
        {14, 5.5}
    };

    const double proposal_size = 0.20;
    const int n_sweeps = 30000;
    const int thermal_cut = 4000;
    const int measure_every = 40;

    const int n_ape = 12;
    const double alpha_ape = 0.50;

    std::uint64_t seed_base = 987654321ull;

    std::cout << "Scaling + smearing production run\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
        run_case(
            cases[i],
            seed_base + 10000ull * static_cast<std::uint64_t>(i),
            proposal_size,
            n_sweeps,
            thermal_cut,
            measure_every,
            n_ape,
            alpha_ape
        );
    }

    return 0;
}
