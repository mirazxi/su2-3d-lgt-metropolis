from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    sweeps = []
    plaquette = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweeps.append(int(row["sweep"]))
            plaquette.append(float(row["plaquette"]))

    return sweeps, plaquette


def main():
    root = Path(__file__).resolve().parent.parent

    cold_file = root / "runs" / "run_cold" / "plaquette.csv"
    hot_file = root / "runs" / "run_hot" / "plaquette.csv"
    out_file = root / "runs" / "equilibration_cold_vs_hot.png"

    sweeps_cold, plaq_cold = read_csv(cold_file)
    sweeps_hot, plaq_hot = read_csv(hot_file)

    thermal_cut = 500

    plt.figure(figsize=(9, 5))
    plt.plot(sweeps_cold, plaq_cold, linewidth=1, label="cold start")
    plt.plot(sweeps_hot, plaq_hot, linewidth=1, label="hot start")
    plt.axvline(thermal_cut, linestyle="--", label="thermalization cut")
    plt.xlabel("Sweep")
    plt.ylabel("Plaquette")
    plt.title("Cold-start vs hot-start equilibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)

    print(f"Saved comparison plot to {out_file}")


if __name__ == "__main__":
    main()
