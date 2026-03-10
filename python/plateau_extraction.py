from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = "large_loops_L8_beta_5p0"
BLOCK_SIZE = 32

# Plateau windows in T for each R
PLATEAU_WINDOWS = {
    1: [2, 3],
    2: [2, 3],
    3: [2, 3],
}


def mean(xs):
    return sum(xs) / len(xs)


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


def weighted_plateau(points):
    # points = [(T, value, err), ...]
    good = [(T, v, e) for (T, v, e) in points if e > 0.0]
    if len(good) == 0:
        return None

    weights = [1.0 / (e * e) for _, _, e in good]
    vbar = sum(w * v for w, (_, v, _) in zip(weights, good)) / sum(weights)
    err = math.sqrt(1.0 / sum(weights))

    chi2 = sum(((v - vbar) / e) ** 2 for _, v, e in good)
    dof = len(good) - 1
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return vbar, err, chi2_dof


def main():
    root = Path(__file__).resolve().parent.parent
    in_file = root / "runs" / RUN_DIR / "wilson_loops.csv"

    out_check = root / "runs" / RUN_DIR / "veff_plateau_check.png"
    out_vr = root / "runs" / RUN_DIR / "V_of_R_plateau.png"
    out_csv = root / "runs" / RUN_DIR / "V_of_R_plateau.csv"

    data = read_wilson_csv(in_file)

    veff = {}
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
            veff[(R, T)] = (v, e)

    print(f"Read Wilson loops from {in_file}\n")
    print("Effective-potential points")
    print("R   T   Veff_jk         err")
    for key in sorted(veff.keys()):
        R, T = key
        v, e = veff[key]
        print(f"{R}   {T}   {v:.12f}   {e:.12f}")

    print("\nPlateau extraction")
    print("R   T_window   V(R)            err             chi2/dof")

    plateau_rows = []
    for R in sorted(PLATEAU_WINDOWS.keys()):
        Ts = PLATEAU_WINDOWS[R]
        points = []

        for T in Ts:
            if (R, T) not in veff:
                continue
            v, e = veff[(R, T)]
            points.append((T, v, e))

        res = weighted_plateau(points)
        if res is None:
            continue

        vbar, err, chi2_dof = res
        plateau_rows.append((R, vbar, err, chi2_dof))

        t_text = ",".join(str(t) for t in Ts)
        print(f"{R}   [{t_text}]     {vbar:.12f}   {err:.12f}   {chi2_dof:.12f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "V", "err", "chi2_dof"])
        for row in plateau_rows:
            writer.writerow(row)

    plt.figure(figsize=(8, 5))
    for R in sorted(set(R for R, T in veff.keys())):
        Ts = sorted(T for RR, T in veff.keys() if RR == R)
        Vs = [veff[(R, T)][0] for T in Ts]
        Es = [veff[(R, T)][1] for T in Ts]
        plt.errorbar(Ts, Vs, yerr=Es, marker="o", capsize=3, label=f"R={R}")

        if R in PLATEAU_WINDOWS:
            pts = []
            for T in PLATEAU_WINDOWS[R]:
                if (R, T) in veff:
                    pts.append((T, veff[(R, T)][0], veff[(R, T)][1]))
            res = weighted_plateau(pts)
            if res is not None:
                vbar, err, chi2_dof = res
                xmin = min(PLATEAU_WINDOWS[R]) - 0.1
                xmax = max(PLATEAU_WINDOWS[R]) + 0.1
                plt.hlines(vbar, xmin, xmax, linestyles="--")

    plt.xlabel("T")
    plt.ylabel(r"$V_{\mathrm{eff}}(R,T)$")
    plt.title("Plateau check for effective potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_check, dpi=150)

    if plateau_rows:
        plt.figure(figsize=(7, 5))
        Rs = [r for r, v, e, c in plateau_rows]
        Vs = [v for r, v, e, c in plateau_rows]
        Es = [e for r, v, e, c in plateau_rows]
        plt.errorbar(Rs, Vs, yerr=Es, marker="o", capsize=3)
        plt.xlabel("R")
        plt.ylabel(r"$V(R)$")
        plt.title("Plateau-extracted potential")
        plt.tight_layout()
        plt.savefig(out_vr, dpi=150)

    print(f"\nSaved:")
    print(out_check)
    print(out_vr)
    print(out_csv)


if __name__ == "__main__":
    main()
