from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = "large_loops_L8_beta_5p0"
BLOCK_SIZE = 32

# Analysis variants:
# windows: plateau T choices for each R
# fit_Rs : which R values are included in the linear fit V(R)=V0+sigma R
VARIANTS = [
    {
        "name": "default_allR",
        "windows": {1: [2, 3], 2: [2, 3], 3: [2, 3]},
        "fit_Rs": [1, 2, 3],
    },
    {
        "name": "late_only_allR",
        "windows": {1: [3], 2: [3], 3: [3]},
        "fit_Rs": [1, 2, 3],
    },
    {
        "name": "R1_123_R23_late",
        "windows": {1: [1, 2, 3], 2: [2, 3], 3: [2, 3]},
        "fit_Rs": [1, 2, 3],
    },
    {
        "name": "default_R23_only",
        "windows": {1: [2, 3], 2: [2, 3], 3: [2, 3]},
        "fit_Rs": [2, 3],
    },
    {
        "name": "late_R23_only",
        "windows": {1: [3], 2: [3], 3: [3]},
        "fit_Rs": [2, 3],
    },
]


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


def build_veff_points(data):
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

            veff[(R, T)] = res

    return veff


def weighted_plateau(points):
    # points = [(T, value, err)]
    good = [(T, v, e) for (T, v, e) in points if e > 0.0]
    if not good:
        return None

    weights = [1.0 / (e * e) for _, _, e in good]
    vbar = sum(w * v for w, (_, v, _) in zip(weights, good)) / sum(weights)
    err = math.sqrt(1.0 / sum(weights))

    chi2 = sum(((v - vbar) / e) ** 2 for _, v, e in good)
    dof = len(good) - 1
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return vbar, err, chi2_dof


def weighted_linear_fit(xs, ys, es):
    ws = [1.0 / (e * e) for e in es]

    S = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

    Delta = S * Sxx - Sx * Sx
    if Delta <= 0.0:
        raise RuntimeError("Weighted fit failed: non-positive determinant.")

    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta

    var_a = Sxx / Delta
    var_b = S / Delta

    err_a = math.sqrt(var_a)
    err_b = math.sqrt(var_b)

    chi2 = sum(((y - (a + b * x)) / e) ** 2 for x, y, e in zip(xs, ys, es))
    dof = len(xs) - 2
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return {
        "V0": a,
        "sigma": b,
        "err_V0": err_a,
        "err_sigma": err_b,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }


def extract_plateau_set(veff, windows):
    rows = []
    for R in sorted(windows.keys()):
        pts = []
        for T in windows[R]:
            if (R, T) in veff:
                v, e = veff[(R, T)]
                pts.append((T, v, e))

        res = weighted_plateau(pts)
        if res is None:
            continue

        vbar, err, chi2_dof = res
        rows.append({
            "R": R,
            "V": vbar,
            "err": err,
            "chi2_dof": chi2_dof,
            "Ts": windows[R],
        })
    return rows


def main():
    root = Path(__file__).resolve().parent.parent
    in_file = root / "runs" / RUN_DIR / "wilson_loops.csv"
    out_csv = root / "runs" / RUN_DIR / "fit_systematics_summary.csv"
    out_plot = root / "runs" / RUN_DIR / "fit_systematics_sigma.png"
    out_txt = root / "runs" / RUN_DIR / "fit_systematics_results.txt"

    data = read_wilson_csv(in_file)
    veff = build_veff_points(data)

    print(f"Read Wilson loops from {in_file}\n")
    print("Fit systematics")
    print("name                 sigma            err_sigma        V0               err_V0           chi2/dof")

    results = []

    for variant in VARIANTS:
        plateau_rows = extract_plateau_set(veff, variant["windows"])

        selected = [row for row in plateau_rows if row["R"] in variant["fit_Rs"]]
        if len(selected) < 2:
            continue

        xs = [float(row["R"]) for row in selected]
        ys = [row["V"] for row in selected]
        es = [row["err"] for row in selected]

        fit = weighted_linear_fit(xs, ys, es)

        result = {
            "name": variant["name"],
            "fit_Rs": variant["fit_Rs"],
            "windows": variant["windows"],
            "sigma": fit["sigma"],
            "err_sigma": fit["err_sigma"],
            "V0": fit["V0"],
            "err_V0": fit["err_V0"],
            "chi2_dof": fit["chi2_dof"],
        }
        results.append(result)

        print(
            f"{variant['name']:<20} "
            f"{fit['sigma']:.12f}   "
            f"{fit['err_sigma']:.12f}   "
            f"{fit['V0']:.12f}   "
            f"{fit['err_V0']:.12f}   "
            f"{fit['chi2_dof']:.12f}"
        )

    if not results:
        print("No valid fit variants were produced.")
        return

    sigmas = [r["sigma"] for r in results]
    central = results[0]["sigma"]   # default_allR as central
    stat_err = results[0]["err_sigma"]
    sys_half_range = 0.5 * (max(sigmas) - min(sigmas))

    print("\nRecommended summary")
    print(f"central sigma (default fit) = {central:.12f}")
    print(f"statistical error           = {stat_err:.12f}")
    print(f"systematic half-range       = {sys_half_range:.12f}")
    print(f"sigma range over variants   = [{min(sigmas):.12f}, {max(sigmas):.12f}]")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sigma", "err_sigma", "V0", "err_V0", "chi2_dof"])
        for r in results:
            writer.writerow([r["name"], r["sigma"], r["err_sigma"], r["V0"], r["err_V0"], r["chi2_dof"]])

    with open(out_txt, "w") as f:
        f.write(f"Input file: {in_file}\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"  fit_Rs     = {r['fit_Rs']}\n")
            f.write(f"  windows    = {r['windows']}\n")
            f.write(f"  sigma      = {r['sigma']:.12f} +/- {r['err_sigma']:.12f}\n")
            f.write(f"  V0         = {r['V0']:.12f} +/- {r['err_V0']:.12f}\n")
            f.write(f"  chi2/dof   = {r['chi2_dof']:.12f}\n\n")
        f.write("Recommended summary\n")
        f.write(f"  central sigma         = {central:.12f}\n")
        f.write(f"  statistical error     = {stat_err:.12f}\n")
        f.write(f"  systematic half-range = {sys_half_range:.12f}\n")
        f.write(f"  sigma range           = [{min(sigmas):.12f}, {max(sigmas):.12f}]\n")

    plt.figure(figsize=(9, 5))
    xs = list(range(len(results)))
    ys = [r["sigma"] for r in results]
    es = [r["err_sigma"] for r in results]
    labels = [r["name"] for r in results]

    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linestyle="none")
    plt.axhline(central, linestyle="--")
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.ylabel(r"$\sigma$")
    plt.title("String-tension fit systematics")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print("\nSaved:")
    print(out_csv)
    print(out_txt)
    print(out_plot)


if __name__ == "__main__":
    main()
