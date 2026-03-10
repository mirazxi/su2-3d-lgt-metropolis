from pathlib import Path
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BETAS = [3.0, 3.5, 4.0, 4.5, 5.0]
THERMAL_CUT = 500

# Plaquette is measured every sweep, so use a larger block.
PLAQ_BLOCK_SIZE = 640

# Wilson loops are measured every 10 sweeps in the current setup.
LOOP_BLOCK_SIZE = 64


def beta_label(beta):
    return f"beta_{beta:.1f}".replace(".", "p")


def mean(xs):
    return sum(xs) / len(xs)


def read_plaquette_csv(path):
    sweeps = []
    plaquette = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweeps.append(int(row["sweep"]))
            plaquette.append(float(row["plaquette"]))

    return sweeps, plaquette


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


def blocked_jackknife_mean(xs, block_size):
    n = len(xs)
    n_blocks = n // block_size
    if n_blocks < 2:
        return None

    n_used = n_blocks * block_size
    xs = xs[:n_used]

    full_mean = mean(xs)

    jk_vals = []
    for b in range(n_blocks):
        start = b * block_size
        stop = (b + 1) * block_size
        kept = xs[:start] + xs[stop:]
        jk_vals.append(mean(kept))

    jk_mean = mean(jk_vals)
    jk_err = math.sqrt((n_blocks - 1) / n_blocks * sum((x - jk_mean) ** 2 for x in jk_vals))
    jk_bias_corrected = n_blocks * full_mean - (n_blocks - 1) * jk_mean

    return full_mean, jk_bias_corrected, jk_err, n_blocks, n_used


def blocked_jackknife_veff(w_t, w_tp1, block_size):
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

        jk_vals.append(-math.log(mean_tp1_b / mean_t_b))

    jk_mean = mean(jk_vals)
    jk_err = math.sqrt((n_blocks - 1) / n_blocks * sum((x - jk_mean) ** 2 for x in jk_vals))
    jk_bias_corrected = n_blocks * theta_full - (n_blocks - 1) * jk_mean

    return theta_full, jk_bias_corrected, jk_err, n_blocks, n_used


def matched_loop_lists(data, R, T):
    if (R, T) not in data or (R, T + 1) not in data:
        return None

    dict_t = dict(data[(R, T)])
    dict_tp1 = dict(data[(R, T + 1)])

    common_sweeps = sorted(set(dict_t.keys()) & set(dict_tp1.keys()))
    if len(common_sweeps) < 2:
        return None

    w_t = [dict_t[s] for s in common_sweeps]
    w_tp1 = [dict_tp1[s] for s in common_sweeps]
    return w_t, w_tp1


def main():
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "runs"

    beta_vals = []

    plaq_vals = []
    plaq_errs = []

    veff11_vals = []
    veff11_errs = []

    veff21_vals = []
    veff21_errs = []

    print("beta   <P>_jk          err(P)          Veff(1,1)_jk     err           Veff(2,1)_jk     err")

    for beta in BETAS:
        folder = runs_dir / beta_label(beta)

        plaq_file = folder / "plaquette.csv"
        wloop_file = folder / "wilson_loops.csv"

        sweeps, plaquette = read_plaquette_csv(plaq_file)
        kept_plaq = [p for s, p in zip(sweeps, plaquette) if s > THERMAL_CUT]

        plaq_res = blocked_jackknife_mean(kept_plaq, PLAQ_BLOCK_SIZE)
        if plaq_res is None:
            raise RuntimeError(f"Not enough plaquette data for beta={beta}")

        _, plaq_jk, plaq_err, _, _ = plaq_res

        wdata = read_wilson_csv(wloop_file)

        loops11 = matched_loop_lists(wdata, 1, 1)
        loops21 = matched_loop_lists(wdata, 2, 1)

        v11_jk = float("nan")
        v11_err = float("nan")
        v21_jk = float("nan")
        v21_err = float("nan")

        if loops11 is not None:
            res11 = blocked_jackknife_veff(loops11[0], loops11[1], LOOP_BLOCK_SIZE)
            if res11 is not None:
                _, v11_jk, v11_err, _, _ = res11

        if loops21 is not None:
            res21 = blocked_jackknife_veff(loops21[0], loops21[1], LOOP_BLOCK_SIZE)
            if res21 is not None:
                _, v21_jk, v21_err, _, _ = res21

        print(
            f"{beta:>4.1f}   "
            f"{plaq_jk:.12f}   "
            f"{plaq_err:.12f}   "
            f"{v11_jk:.12f}   "
            f"{v11_err:.12f}   "
            f"{v21_jk:.12f}   "
            f"{v21_err:.12f}"
        )

        beta_vals.append(beta)
        plaq_vals.append(plaq_jk)
        plaq_errs.append(plaq_err)
        veff11_vals.append(v11_jk)
        veff11_errs.append(v11_err)
        veff21_vals.append(v21_jk)
        veff21_errs.append(v21_err)

    plt.figure(figsize=(8, 5))
    plt.errorbar(beta_vals, plaq_vals, yerr=plaq_errs, marker="o", capsize=3)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle P \rangle$")
    plt.title("Mean plaquette vs beta (blocked jackknife)")
    plt.tight_layout()
    plt.savefig(runs_dir / "plaquette_vs_beta_with_errors.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.errorbar(beta_vals, veff11_vals, yerr=veff11_errs, marker="o", capsize=3, label="R=1, T=1")
    plt.errorbar(beta_vals, veff21_vals, yerr=veff21_errs, marker="o", capsize=3, label="R=2, T=1")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$V_{\mathrm{eff}}$")
    plt.title("Effective potential vs beta (blocked jackknife)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(runs_dir / "veff_vs_beta_with_errors.png", dpi=150)

    print("\nSaved:")
    print(runs_dir / "plaquette_vs_beta_with_errors.png")
    print(runs_dir / "veff_vs_beta_with_errors.png")


if __name__ == "__main__":
    main()
