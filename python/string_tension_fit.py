from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = "large_loops_L8_beta_5p0"


def weighted_linear_fit(xs, ys, es):
    # Fit y = a + b x with weights 1/e^2
    ws = [1.0 / (e * e) for e in es]

    S = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

    Delta = S * Sxx - Sx * Sx
    if Delta <= 0.0:
        raise RuntimeError("Weighted fit failed: non-positive determinant.")

    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta

    var_a = Sxx / Delta
    var_b = S / Delta
    cov_ab = -Sx / Delta

    err_a = math.sqrt(var_a)
    err_b = math.sqrt(var_b)

    chi2 = sum(((y - (a + b * x)) / e) ** 2 for x, y, e in zip(xs, ys, es))
    dof = len(xs) - 2
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return {
        "V0": a,
        "sigma": b,
        "err_V0": err_a,
        "err_sigma": err_b,
        "cov_ab": cov_ab,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    in_file = root / "runs" / RUN_DIR / "V_of_R_plateau.csv"
    out_plot = root / "runs" / RUN_DIR / "string_tension_fit.png"
    out_txt = root / "runs" / RUN_DIR / "string_tension_fit_results.txt"

    Rs = []
    Vs = []
    Es = []

    with open(in_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Rs.append(float(row["R"]))
            Vs.append(float(row["V"]))
            Es.append(float(row["err"]))

    if len(Rs) < 2:
        raise RuntimeError("Need at least 2 points for a linear fit.")

    fit = weighted_linear_fit(Rs, Vs, Es)

    print(f"Read plateau data from {in_file}\n")
    print("Fit model: V(R) = V0 + sigma * R\n")
    print(f"V0      = {fit['V0']:.12f} ± {fit['err_V0']:.12f}")
    print(f"sigma   = {fit['sigma']:.12f} ± {fit['err_sigma']:.12f}")
    print(f"chi2    = {fit['chi2']:.12f}")
    print(f"dof     = {fit['dof']}")
    print(f"chi2/dof= {fit['chi2_dof']:.12f}")

    with open(out_txt, "w") as f:
        f.write("Fit model: V(R) = V0 + sigma * R\n")
        f.write(f"Input file: {in_file}\n\n")
        f.write(f"V0       = {fit['V0']:.12f} +/- {fit['err_V0']:.12f}\n")
        f.write(f"sigma    = {fit['sigma']:.12f} +/- {fit['err_sigma']:.12f}\n")
        f.write(f"chi2     = {fit['chi2']:.12f}\n")
        f.write(f"dof      = {fit['dof']}\n")
        f.write(f"chi2/dof = {fit['chi2_dof']:.12f}\n")

    xmin = min(Rs) - 0.2
    xmax = max(Rs) + 0.2
    xfit = [xmin + (xmax - xmin) * i / 200.0 for i in range(201)]
    yfit = [fit["V0"] + fit["sigma"] * x for x in xfit]

    plt.figure(figsize=(7, 5))
    plt.errorbar(Rs, Vs, yerr=Es, marker="o", capsize=3, linestyle="none", label="plateau data")
    plt.plot(xfit, yfit, label=fr"fit: $V_0+\sigma R$")
    plt.xlabel("R")
    plt.ylabel("V(R)")
    plt.title("String-tension fit from plateau-extracted potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"\nSaved:")
    print(out_plot)
    print(out_txt)


if __name__ == "__main__":
    main()
