from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = "large_loops_L8_beta_5p0"
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


def main():
    root = Path(__file__).resolve().parent.parent
    in_file = root / "runs" / RUN_DIR / "wilson_loops.csv"

    out_w = root / "runs" / RUN_DIR / "wilson_loops_large.png"
    out_v = root / "runs" / RUN_DIR / "effective_potential_large.png"

    data = read_wilson_csv(in_file)

    print(f"Read Wilson loops from {in_file}\n")
    print("R   T   mean(W)          stderr")
    for key in sorted(data.keys()):
        R, T = key
        vals = [w for _, w in data[key]]
        print(f"{R}   {T}   {mean(vals):.12f}   {stderr(vals):.12f}")

    plt.figure(figsize=(8, 5))
    all_R = sorted(set(R for R, T in data.keys()))
    for R in all_R:
        Ts = sorted(T for RR, T in data.keys() if RR == R)
        means = [mean([w for _, w in data[(R, T)]]) for T in Ts]
        errs = [stderr([w for _, w in data[(R, T)]]) for T in Ts]
        plt.errorbar(Ts, means, yerr=errs, marker="o", capsize=3, label=f"R={R}")

    plt.xlabel("T")
    plt.ylabel("W(R,T)")
    plt.title("Larger Wilson loops")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_w, dpi=150)

    print("\nR   T   Veff_jk         err")
    veff_results = []
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
            veff_results.append((R, T, v, e))
            print(f"{R}   {T}   {v:.12f}   {e:.12f}")

    plt.figure(figsize=(8, 5))
    for R in sorted(set(R for R, T, v, e in veff_results)):
        Ts = [T for RR, T, v, e in veff_results if RR == R]
        Vs = [v for RR, T, v, e in veff_results if RR == R]
        Es = [e for RR, T, v, e in veff_results if RR == R]
        plt.errorbar(Ts, Vs, yerr=Es, marker="o", capsize=3, label=f"R={R}")

    plt.xlabel("T")
    plt.ylabel(r"$V_{\mathrm{eff}}(R,T)$")
    plt.title("Effective potential from larger loops")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_v, dpi=150)

    print(f"\nSaved:\n{out_w}\n{out_v}")


if __name__ == "__main__":
    main()
