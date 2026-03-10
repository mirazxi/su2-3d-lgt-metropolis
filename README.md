# su2-3d-lgt-metropolis

Metropolis Monte Carlo study of three-dimensional SU(2) lattice gauge theory with analyses of plaquettes, Wilson loops, APE smearing, effective potentials, plateau extraction, and static-potential fits.

## Overview

This repository contains the simulation and analysis workflow used to study nonperturbative observables in three-dimensional SU(2) lattice gauge theory with the Wilson action. The project combines:

- **C++ simulation code** for gauge-field generation with the Metropolis algorithm
- **Python analysis scripts** for error estimation, effective-potential extraction, plateau selection, and static-potential fitting
- **Derived outputs** used to build figures and tables for the manuscript

The current study includes equilibration tests, blocking-based uncertainty analysis, coupling and volume scans, APE-smeared Wilson-loop measurements, plateau-based extraction of the static potential \(V(R)\), and Cornell-type fits of the form

\[
V(R)=V_0+\sigma R-\alpha/R.
\]

## Main results

The repository currently supports a three-coupling scaling study with APE-smeared Wilson loops at:

- \(L=10,\ \beta=4.5\)
- \(L=12,\ \beta=5.0\)
- \(L=14,\ \beta=5.5\)

Key outcomes include:

- explicit cold-start / hot-start equilibration checks
- blocked and blocked-jackknife error estimates
- comparison of unsmeared and APE-smeared effective potentials
- plateau-extracted static-potential points \(V(R)\)
- comparison between linear and Cornell-type fit forms
- scaling trend of the fitted lattice string tension \(\sigma\) with increasing \(\beta\)

## Repository structure

```text
cpp/        C++ simulation code and build files
python/     Python analysis and plotting scripts
runs/       Selected simulation outputs, summary CSV files, and figures
paper/      Manuscript source and paper-ready tables/figures (if included)
