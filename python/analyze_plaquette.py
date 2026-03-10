from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    sweeps = []
    acceptance = []
    plaquette = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweeps.append(int(row["sweep"]))
            acceptance.append(float(row["acceptance"]))
            plaquette.append(float(row["plaquette"]))

    return sweeps, acceptance, plaquette


def mean(xs):
    return sum(xs) / len(xs)


def std_error(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def main():
    root = Path(__file__).resolve().parent.parent
    data_file = root / "runs" / "run_001" / "plaquette.csv"
    plot_file = root / "runs" / "run_001" / "plaquette_vs_sweep.png"

    sweeps, acceptance, plaquette = read_csv(data_file)

    thermal_cut = 50
    kept_plaquette = [p for s, p in zip(sweeps, plaquette) if s > thermal_cut]
    kept_acceptance = [a for s, a in zip(sweeps, acceptance) if s > thermal_cut]

    print(f"Read {len(sweeps)} measurements from {data_file}")
    print(f"Thermalization cut: sweep <= {thermal_cut}")
    print(f"Mean plaquette after cut  = {mean(kept_plaquette):.12f}")
    print(f"Naive standard error      = {std_error(kept_plaquette):.12f}")
    print(f"Mean acceptance after cut = {mean(kept_acceptance):.12f}")

    plt.figure(figsize=(8, 5))
    plt.plot(sweeps, plaquette, marker="o", markersize=3, linewidth=1)
    plt.axvline(thermal_cut, linestyle="--")
    plt.xlabel("Sweep")
    plt.ylabel("Plaquette")
    plt.title("Plaquette vs Sweep")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)

    print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
