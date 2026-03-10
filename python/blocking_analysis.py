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


def main():
    root = Path(__file__).resolve().parent.parent
    data_file = root / "runs" / "run_001" / "plaquette.csv"
    plot_file = root / "runs" / "run_001" / "blocking_error.png"

    sweeps, acceptance, plaquette = read_csv(data_file)

    thermal_cut = 50
    kept = [p for s, p in zip(sweeps, plaquette) if s > thermal_cut]

    print(f"Read {len(sweeps)} total measurements from {data_file}")
    print(f"Thermalization cut: sweep <= {thermal_cut}")
    print(f"Remaining measurements: {len(kept)}")
    print(f"Mean plaquette = {mean(kept):.12f}")
    print(f"Naive stderr   = {naive_stderr(kept):.12f}")
    print()

    max_block = max(1, len(kept) // 8)
    block_sizes = powers_of_two_up_to(max_block)

    results = []
    for b in block_sizes:
        res = blocking_stats(kept, b)
        if res is not None:
            results.append(res)

    if not results:
        print("Not enough data for blocking analysis.")
        return

    print("Blocking analysis")
    print("block_size   n_blocks   mean            stderr")
    for r in results:
        print(
            f"{r['block_size']:>10d}   "
            f"{r['n_blocks']:>8d}   "
            f"{r['mean']:.12f}   "
            f"{r['stderr']:.12f}"
        )

    best = results[-1]
    print()
    print(f"Largest usable block size = {best['block_size']}")
    print(f"Blocked stderr estimate   = {best['stderr']:.12f}")

    x = [r["block_size"] for r in results]
    y = [r["stderr"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("Block size")
    plt.ylabel("Blocked standard error")
    plt.title("Blocking analysis of plaquette")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)

    print(f"Saved blocking plot to {plot_file}")


if __name__ == "__main__":
    main()
