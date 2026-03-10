from pathlib import Path
import csv
import math

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


def mean(xs):
    return sum(xs) / len(xs)


def sample_variance(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def naive_stderr(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    return math.sqrt(sample_variance(xs) / n)


def blocking_stats(xs, block_size):
    n_blocks = len(xs) // block_size
    if n_blocks < 2:
        return None

    trimmed = xs[: n_blocks * block_size]
    block_means = []

    for i in range(n_blocks):
        block = trimmed[i * block_size : (i + 1) * block_size]
        block_means.append(mean(block))

    blocked_mean = mean(block_means)
    blocked_stderr = math.sqrt(sample_variance(block_means) / n_blocks)

    return {
        "block_size": block_size,
        "n_blocks": n_blocks,
        "mean": blocked_mean,
        "stderr": blocked_stderr,
    }


def powers_of_two_up_to(max_value):
    vals = []
    b = 1
    while b <= max_value:
        vals.append(b)
        b *= 2
    return vals


def analyze_run(label, sweeps, plaquette, thermal_cut):
    kept = [p for s, p in zip(sweeps, plaquette) if s > thermal_cut]

    max_block = max(1, len(kept) // 8)
    block_sizes = powers_of_two_up_to(max_block)

    results = []
    for b in block_sizes:
        res = blocking_stats(kept, b)
        if res is not None:
            results.append(res)

    summary = {
        "label": label,
        "n_total": len(sweeps),
        "n_kept": len(kept),
        "mean": mean(kept),
        "naive_stderr": naive_stderr(kept),
        "blocking_results": results,
        "blocked_stderr": results[-1]["stderr"] if results else None,
        "largest_block": results[-1]["block_size"] if results else None,
    }
    return summary


def print_summary(summary):
    print(f"=== {summary['label']} ===")
    print(f"Total measurements      : {summary['n_total']}")
    print(f"Measurements after cut  : {summary['n_kept']}")
    print(f"Mean plaquette          : {summary['mean']:.12f}")
    print(f"Naive stderr            : {summary['naive_stderr']:.12f}")
    if summary["blocked_stderr"] is not None:
        print(f"Blocked stderr          : {summary['blocked_stderr']:.12f}")
        print(f"Largest usable block    : {summary['largest_block']}")
    print()

    if summary["blocking_results"]:
        print("block_size   n_blocks   mean            stderr")
        for r in summary["blocking_results"]:
            print(
                f"{r['block_size']:>10d}   "
                f"{r['n_blocks']:>8d}   "
                f"{r['mean']:.12f}   "
                f"{r['stderr']:.12f}"
            )
        print()


def main():
    root = Path(__file__).resolve().parent.parent

    cold_file = root / "runs" / "run_cold" / "plaquette.csv"
    hot_file = root / "runs" / "run_hot" / "plaquette.csv"
    out_plot = root / "runs" / "cold_hot_blocking_comparison.png"

    thermal_cut = 500

    sweeps_cold, plaq_cold = read_csv(cold_file)
    sweeps_hot, plaq_hot = read_csv(hot_file)

    cold = analyze_run("cold", sweeps_cold, plaq_cold, thermal_cut)
    hot = analyze_run("hot", sweeps_hot, plaq_hot, thermal_cut)

    print(f"Thermalization cut: sweep <= {thermal_cut}\n")
    print_summary(cold)
    print_summary(hot)

    if cold["blocked_stderr"] is not None and hot["blocked_stderr"] is not None:
        diff = abs(cold["mean"] - hot["mean"])
        combined = math.sqrt(cold["blocked_stderr"]**2 + hot["blocked_stderr"]**2)
        sigma = diff / combined if combined > 0 else float("inf")

        print("=== cold vs hot comparison ===")
        print(f"|mean_cold - mean_hot|   = {diff:.12f}")
        print(f"Combined blocked error   = {combined:.12f}")
        print(f"Difference / error       = {sigma:.6f}")
        print()

    plt.figure(figsize=(8, 5))

    if cold["blocking_results"]:
        x_cold = [r["block_size"] for r in cold["blocking_results"]]
        y_cold = [r["stderr"] for r in cold["blocking_results"]]
        plt.plot(x_cold, y_cold, marker="o", linewidth=1, label="cold")

    if hot["blocking_results"]:
        x_hot = [r["block_size"] for r in hot["blocking_results"]]
        y_hot = [r["stderr"] for r in hot["blocking_results"]]
        plt.plot(x_hot, y_hot, marker="o", linewidth=1, label="hot")

    plt.xscale("log", base=2)
    plt.xlabel("Block size")
    plt.ylabel("Blocked standard error")
    plt.title("Cold vs hot blocking analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"Saved blocking comparison plot to {out_plot}")


if __name__ == "__main__":
    main()
