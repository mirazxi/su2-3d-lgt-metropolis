from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BETAS = [3.0, 3.5, 4.0, 4.5, 5.0]
THERMAL_CUT = 500


def beta_label(beta):
    return f"beta_{beta:.1f}".replace(".", "p")


def read_plaquette_csv(path):
    sweeps = []
    plaquette = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweeps.append(int(row["sweep"]))
            plaquette.append(float(row["plaquette"]))

    return sweeps, plaquette


def read_wilson_csv(path):
    data = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            R = int(row["R"])
            T = int(row["T"])
            W = float(row["W"])
            data[(R, T)].append(W)

    return data


def mean(xs):
    return sum(xs) / len(xs)


def stderr(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "runs"

    beta_vals = []
    plaquette_means = []
    plaquette_errs = []
    veff_r1 = []
    veff_r2 = []

    print("beta   <P>             err(P)          Veff(1,1)        Veff(2,1)")

    for beta in BETAS:
        folder = out_dir / beta_label(beta)

        plaq_file = folder / "plaquette.csv"
        wloop_file = folder / "wilson_loops.csv"

        sweeps, plaquette = read_plaquette_csv(plaq_file)
        kept_plaq = [p for s, p in zip(sweeps, plaquette) if s > THERMAL_CUT]

        plaq_mean = mean(kept_plaq)
        plaq_err = stderr(kept_plaq)

        wdata = read_wilson_csv(wloop_file)
        means = {key: mean(vals) for key, vals in wdata.items()}

        v11 = float("nan")
        v21 = float("nan")

        if (1, 1) in means and (1, 2) in means and means[(1, 1)] > 0 and means[(1, 2)] > 0:
            v11 = -math.log(means[(1, 2)] / means[(1, 1)])

        if (2, 1) in means and (2, 2) in means and means[(2, 1)] > 0 and means[(2, 2)] > 0:
            v21 = -math.log(means[(2, 2)] / means[(2, 1)])

        print(f"{beta:>4.1f}   {plaq_mean:.12f}   {plaq_err:.12f}   {v11:.12f}   {v21:.12f}")

        beta_vals.append(beta)
        plaquette_means.append(plaq_mean)
        plaquette_errs.append(plaq_err)
        veff_r1.append(v11)
        veff_r2.append(v21)

    plt.figure(figsize=(8, 5))
    plt.errorbar(beta_vals, plaquette_means, yerr=plaquette_errs, marker="o", capsize=3)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle P \rangle$")
    plt.title("Mean plaquette vs beta")
    plt.tight_layout()
    plt.savefig(out_dir / "plaquette_vs_beta.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(beta_vals, veff_r1, marker="o", label="R=1, T=1")
    plt.plot(beta_vals, veff_r2, marker="o", label="R=2, T=1")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$V_{\mathrm{eff}}$")
    plt.title("Effective potential vs beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "veff_vs_beta.png", dpi=150)

    print("\nSaved:")
    print(out_dir / "plaquette_vs_beta.png")
    print(out_dir / "veff_vs_beta.png")


if __name__ == "__main__":
    main()
