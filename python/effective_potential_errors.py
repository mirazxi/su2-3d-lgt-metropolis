from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_LABEL = "cold"
BLOCK_SIZE = 64   # measured configurations per block


def read_wilson_csv(path):
    # data[(R,T)] = list of (sweep, W)
    data = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweep = int(row["sweep"])
            R = int(row["R"])
            T = int(row["T"])
            W = float(row["W"])
            data[(R, T)].append((sweep, W))

    # sort by sweep for safety
    for key in data:
        data[key].sort(key=lambda x: x[0])

    return data


def mean(xs):
    return sum(xs) / len(xs)


def blocked_jackknife_veff(w_t, w_tp1, block_size):
    """
    w_t and w_tp1 are matched lists of Wilson-loop values for the same sweeps.
    Returns:
      theta_full, theta_jackknife, theta_err, n_blocks
    """
    n = min(len(w_t), len(w_tp1))
    n_blocks = n // block_size
    if n_blocks < 2:
        return None

    n_used = n_blocks * block_size
    w_t = w_t[:n_used]
    w_tp1 = w_tp1[:n_used]

    mean_t = mean(w_t)
    mean_tp1 = mean(w_tp1)

    if mean_t <= 0.0 or mean_tp1 <= 0.0:
        return None

    theta_full = -math.log(mean_tp1 / mean_t)

    jk_vals = []
    for b in range(n_blocks):
        start = b * block_size
        stop = (b + 1) * block_size

        kept_t = w_t[:start] + w_t[stop:]
        kept_tp1 = w_tp1[:start] + w_tp1[stop:]

        mean_t_b = mean(kept_t)
        mean_tp1_b = mean(kept_tp1)

        if mean_t_b <= 0.0 or mean_tp1_b <= 0.0:
            return None

        theta_b = -math.log(mean_tp1_b / mean_t_b)
        jk_vals.append(theta_b)

    jk_mean = mean(jk_vals)
    jk_err = math.sqrt((n_blocks - 1) / n_blocks * sum((x - jk_mean) ** 2 for x in jk_vals))
    theta_bias_corrected = n_blocks * theta_full - (n_blocks - 1) * jk_mean

    return theta_full, theta_bias_corrected, jk_err, n_blocks


def main():
    root = Path(__file__).resolve().parent.parent
    in_file = root / "runs" / f"run_{RUN_LABEL}" / "wilson_loops.csv"
    out_plot = root / "runs" / f"run_{RUN_LABEL}" / f"effective_potential_{RUN_LABEL}_with_errors.png"

    data = read_wilson_csv(in_file)

    print(f"Read Wilson loops from {in_file}")
    print(f"Run label   : {RUN_LABEL}")
    print(f"Block size  : {BLOCK_SIZE} measured configurations\n")

    results = []

    all_R = sorted(set(R for R, T in data.keys()))
    for R in all_R:
        all_T = sorted(T for RR, T in data.keys() if RR == R)

        for T in all_T:
            if (R, T + 1) not in data:
                continue

            # Match by sweep
            dict_t = dict(data[(R, T)])
            dict_tp1 = dict(data[(R, T + 1)])
            common_sweeps = sorted(set(dict_t.keys()) & set(dict_tp1.keys()))

            w_t = [dict_t[s] for s in common_sweeps]
            w_tp1 = [dict_tp1[s] for s in common_sweeps]

            res = blocked_jackknife_veff(w_t, w_tp1, BLOCK_SIZE)
            if res is None:
                continue

            theta_full, theta_jk, theta_err, n_blocks = res
            results.append((R, T, theta_full, theta_jk, theta_err, n_blocks, len(common_sweeps)))

    if not results:
        print("No valid effective-potential estimates could be produced.")
        return

    print("R   T   Veff(full)       Veff(jk)         jk_err           n_blocks   n_meas")
    for R, T, theta_full, theta_jk, theta_err, n_blocks, n_meas in results:
        print(
            f"{R}   {T}   "
            f"{theta_full:.12f}   "
            f"{theta_jk:.12f}   "
            f"{theta_err:.12f}   "
            f"{n_blocks:>8d}   "
            f"{n_meas:>6d}"
        )

    plt.figure(figsize=(8, 5))
    for R in sorted(set(R for R, T, *_ in results)):
        Ts = [T for RR, T, *_ in results if RR == R]
        Vs = [theta_jk for RR, T, theta_full, theta_jk, theta_err, n_blocks, n_meas in results if RR == R]
        Es = [theta_err for RR, T, theta_full, theta_jk, theta_err, n_blocks, n_meas in results if RR == R]
        plt.errorbar(Ts, Vs, yerr=Es, marker="o", capsize=3, label=f"R={R}")

    plt.xlabel("T")
    plt.ylabel(r"$V_{\mathrm{eff}}(R,T)$")
    plt.title(f"Effective potential with blocked jackknife errors ({RUN_LABEL} start)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"\nSaved effective-potential error plot to {out_plot}")


if __name__ == "__main__":
    main()
