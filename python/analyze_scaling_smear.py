# python/analyze_scaling_smear.py
from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    "scaling_L10_beta_4p5",
    "scaling_L12_beta_5p0",
    "scaling_L14_beta_5p5",
]

BLOCK_SIZE = 32


def mean(xs):
    return sum(xs) / len(xs)


def stderr(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def read_wilson_csv(path):
    data = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweep = int(row["sweep"])
            R = int(row["R"])
            T = int(row["T"])
            W = float(row["W"])
            data[(R, T)].append((sweep, W))

    for key in data:
        data[key].sort(key=lambda x: x[0])

    return data


def blocked_jackknife_veff(w_t, w_tp1, block_size):
    n = min(len(w_t), len(w_tp1))
    n_blocks = n // block_size
    if n_blocks < 2:
        return None

    n_used = n_blocks * block_size
    w_t = w_t[:n_used]
    w_tp1 = w_tp1[:n_used]

    mt = mean(w_t)
    mt1 = mean(w_tp1)

    if mt <= 0.0 or mt1 <= 0.0:
        return None

    full = -math.log(mt1 / mt)

    jk_vals = []
    for b in range(n_blocks):
        start = b * block_size
        stop = (b + 1) * block_size

        kept_t = w_t[:start] + w_t[stop:]
        kept_t1 = w_tp1[:start] + w_tp1[stop:]

        mt_b = mean(kept_t)
        mt1_b = mean(kept_t1)

        if mt_b <= 0.0 or mt1_b <= 0.0:
            return None

        jk_vals.append(-math.log(mt1_b / mt_b))

    jk_mean = mean(jk_vals)
    jk_err = math.sqrt((n_blocks - 1) / n_blocks * sum((x - jk_mean) ** 2 for x in jk_vals))
    jk_bc = n_blocks * full - (n_blocks - 1) * jk_mean

    return jk_bc, jk_err


def build_veff(data):
    veff = []
    all_R = sorted(set(R for R, T in data.keys()))
    for R in all_R:
        Ts = sorted(T for RR, T in data.keys() if RR == R)
        for T in Ts:
            if (R, T + 1) not in data:
                continue

            dict_t = dict(data[(R, T)])
            dict_t1 = dict(data[(R, T + 1)])
            common = sorted(set(dict_t.keys()) & set(dict_t1.keys()))

            w_t = [dict_t[s] for s in common]
            w_t1 = [dict_t1[s] for s in common]

            res = blocked_jackknife_veff(w_t, w_t1, BLOCK_SIZE)
            if res is None:
                continue

            v, e = res
            veff.append((R, T, v, e))
    return veff


def save_veff_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "T", "Veff", "err"])
        for row in rows:
            writer.writerow(row)


def plot_case(case_dir, veff_raw, veff_ape):
    out_plot = case_dir / "veff_raw_vs_ape.png"

    all_R = sorted(set(R for R, T, v, e in veff_raw) | set(R for R, T, v, e in veff_ape))
    nR = len(all_R)

    fig, axes = plt.subplots(nR, 1, figsize=(7, 3.2 * nR), sharex=True)
    if nR == 1:
        axes = [axes]

    for ax, R in zip(axes, all_R):
        Ts_raw = [T for RR, T, v, e in veff_raw if RR == R]
        Vs_raw = [v for RR, T, v, e in veff_raw if RR == R]
        Es_raw = [e for RR, T, v, e in veff_raw if RR == R]

        Ts_ape = [T for RR, T, v, e in veff_ape if RR == R]
        Vs_ape = [v for RR, T, v, e in veff_ape if RR == R]
        Es_ape = [e for RR, T, v, e in veff_ape if RR == R]

        if Ts_raw:
            ax.errorbar(Ts_raw, Vs_raw, yerr=Es_raw, marker="o", capsize=3, linestyle="-", label="raw")
        if Ts_ape:
            ax.errorbar(Ts_ape, Vs_ape, yerr=Es_ape, marker="s", capsize=3, linestyle="--", label="APE")

        ax.set_ylabel(f"R={R}")
        ax.legend()

    axes[-1].set_xlabel("T")
    fig.suptitle(case_dir.name + ": raw vs APE effective potential")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent / "runs"

    for case in CASES:
        case_dir = root / case
        raw_file = case_dir / "wilson_loops_raw.csv"
        ape_file = case_dir / "wilson_loops_ape.csv"

        raw_data = read_wilson_csv(raw_file)
        ape_data = read_wilson_csv(ape_file)

        veff_raw = build_veff(raw_data)
        veff_ape = build_veff(ape_data)

        save_veff_csv(case_dir / "veff_raw.csv", veff_raw)
        save_veff_csv(case_dir / "veff_ape.csv", veff_ape)
        plot_case(case_dir, veff_raw, veff_ape)

        print(f"\n=== {case} ===")
        print("RAW")
        print("R   T   Veff           err")
        for R, T, v, e in veff_raw:
            print(f"{R}   {T}   {v:.12f}   {e:.12f}")

        print("\nAPE")
        print("R   T   Veff           err")
        for R, T, v, e in veff_ape:
            print(f"{R}   {T}   {v:.12f}   {e:.12f}")

        print(f"\nSaved:")
        print(case_dir / "veff_raw.csv")
        print(case_dir / "veff_ape.csv")
        print(case_dir / "veff_raw_vs_ape.png")


if __name__ == "__main__":
    main()
