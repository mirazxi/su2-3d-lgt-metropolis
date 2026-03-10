from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    in_file = root / "runs" / "run_cold" / "wilson_loops.csv"
    out_plot = root / "runs" / "run_cold" / "effective_potential_cold.png"

    data = read_wilson_csv(in_file)

    means = {key: mean(vals) for key, vals in data.items()}
    errs = {key: stderr(vals) for key, vals in data.items()}

    print(f"Read Wilson loops from {in_file}\n")
    print("R   T   V_eff(R,T)")
    veff_points = []

    for R in sorted(set(R for R, T in means.keys())):
        Ts = sorted(T for RR, T in means.keys() if RR == R)
        for T in Ts:
            if (R, T + 1) in means and means[(R, T)] > 0 and means[(R, T + 1)] > 0:
                v = -math.log(means[(R, T + 1)] / means[(R, T)])
                veff_points.append((R, T, v))
                print(f"{R}   {T}   {v:.12f}")

    plt.figure(figsize=(8, 5))

    for R in sorted(set(R for R, T, v in veff_points)):
        Ts = [T for RR, T, v in veff_points if RR == R]
        Vs = [v for RR, T, v in veff_points if RR == R]
        plt.plot(Ts, Vs, marker="o", linewidth=1, label=f"R={R}")

    plt.xlabel("T")
    plt.ylabel(r"$V_{\mathrm{eff}}(R,T)$")
    plt.title("Effective potential from cold-start Wilson loops")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"\nSaved effective-potential plot to {out_plot}")


if __name__ == "__main__":
    main()
