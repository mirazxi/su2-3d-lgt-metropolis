# su2-3d-lgt-metropolis

Metropolis Monte Carlo study of three-dimensional SU(2) lattice gauge theory with plaquette measurements, Wilson loops, APE smearing, effective-potential analysis, Cornell-type fits, scaling trends, and finite-volume stability checks.

## Repository layout

- `cpp/` — C++ simulation code and executables
- `python/` — analysis and plotting scripts
- `runs/` — generated outputs, summary CSV files, and figures used in the paper

## Main components

### C++ simulation programs
- `cpp/src/main.cpp` — baseline production run
- `cpp/src/main_scaling_smear.cpp` — three-coupling scaling and APE-smearing study
- `cpp/src/main_volume_check_beta5.cpp` — finite-volume check at beta = 5.0
- `cpp/src/main_volume_check_beta55.cpp` — finite-volume check at beta = 5.5
- `cpp/src/main_larger_loops.cpp` — larger-loop study

### Python analysis scripts
- `python/analyze_scaling_smear.py` — builds raw/APE effective potentials and comparison plots
- `python/volumecheck_beta5_fit.py` — Cornell-type finite-volume analysis at beta = 5.0
- `python/volumecheck_beta55_fit.py` — Cornell-type finite-volume analysis at beta = 5.5
- `python/plot_sigma_vs_L.py` — finite-volume stability plot for the fitted string tension
- `python/plot_sigma_scaling_final.py` — final scaling plot of the preferred string tension values versus beta

## Build instructions

From the `cpp/` directory:

```bash
cd cpp
rm -rf build
mkdir build
cd build
cmake ..
make
