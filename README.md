# su2-3d-lgt-metropolis

Metropolis Monte Carlo study of three-dimensional SU(2) lattice gauge theory with plaquette measurements, Wilson loops, APE smearing, effective-potential analysis, Cornell-type fits, scaling trends, finite-volume stability checks, and a continuum-facing extrapolation of the string tension.

## Reproducibility

This repository contains both the C++ production code and the Python analysis scripts used to generate the numerical results, summary tables, and figures in the paper.

### Main production programs

The main C++ entry points are:

- `cpp/src/main.cpp` — baseline production run
- `cpp/src/main_scaling_smear.cpp` — scaling study with APE-smearing measurements
- `cpp/src/main_volume_check_beta5.cpp` — finite-volume study at `beta = 5.0`
- `cpp/src/main_volume_check_beta55.cpp` — finite-volume study at `beta = 5.5`
- `cpp/src/main_larger_loops.cpp` — larger-loop production study

### Main analysis scripts

The most important Python scripts are:

- `python/analyze_scaling_smear.py` — constructs raw/APE effective potentials from Wilson-loop data
- `python/plateau_and_scaling_fit.py` — performs plateau extraction and Cornell/linear fit summaries
- `python/volumecheck_beta5_fit.py` — finite-volume Cornell analysis at `beta = 5.0`
- `python/volumecheck_beta55_fit.py` — finite-volume Cornell analysis at `beta = 5.5`
- `python/plot_sigma_vs_L.py` — finite-volume stability plots for the fitted string tension
- `python/plot_sigma_scaling_final.py` — scaling plot of the preferred string-tension values versus `beta`
- `python/make_veff_beta60.py` — builds `veff_ape.csv` and `veff_raw.csv` for the `beta = 6.0` runs
- `python/make_veff_beta65.py` — builds `veff_ape.csv` and `veff_raw.csv` for the `beta = 6.5` runs
- `python/continuum_fit_current.py` — continuum-facing extrapolation of `sqrt(sigma)/g^2`
- `python/shifted_plateau_sensitivity.py` — shifted-window plateau-sensitivity checks

### Typical workflow

A typical end-to-end workflow is:

1. Build the C++ code in `cpp/`
2. Run the desired production executable
3. Generate Wilson-loop derived effective-potential files
4. Perform plateau extraction and Cornell/linear fit comparisons
5. Run finite-volume and scaling summaries
6. Produce the continuum-facing extrapolation and final figures

### Minimal reproduction recipe

From the repository root:

```bash
cd cpp
rm -rf build
mkdir build
cd build
cmake ..
make
