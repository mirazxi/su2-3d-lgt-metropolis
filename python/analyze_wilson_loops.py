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
    out_plot = root / "runs" / "run_cold" / "wilson_loops_cold.png"

    data = read_wilson_csv(in_file)

    print(f"Read Wilson loops from {in_file}\n")
    print("R   T   mean(W)          stderr")
    for key in sorted(data.keys()):
        R, T = key
        vals = data[key]
        print(f"{R}   {T}   {mean(vals):.12f}   {stderr(vals):.12f}")

    plt.figure(figsize=(8, 5))

    for R in sorted(set(R for R, T in data.keys())):
        Ts = sorted(T for RR, T in data.keys() if RR == R)
        means = [mean(data[(R, T)]) for T in Ts]
        errs = [stderr(data[(R, T)]) for T in Ts]
        plt.errorbar(Ts, means, yerr=errs, marker="o", capsize=3, label=f"R={R}")

    plt.xlabel("T")
    plt.ylabel("W(R,T)")
    plt.title("Wilson loops from cold-start ensemble")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"\nSaved Wilson-loop plot to {out_plot}")


if __name__ == "__main__":
    main()
