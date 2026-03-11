from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    {
        "name": "scaling_L14_beta_5p5",
        "label": "L=14, beta=5.5",
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3],
        },
    },
    {
        "name": "volumecheck_L16_beta_5p5",
        "label": "L=16, beta=5.5",
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4, 5],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3],
        },
    },
]


def invert_matrix(a):
    n = len(a)
    aug = []
    for i in range(n):
        row = list(a[i]) + [0.0] * n
        row[n + i] = 1.0
        aug.append(row)

    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-14:
            raise RuntimeError("Singular matrix in inversion.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        fac = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= fac

        for r in range(n):
            if r == col:
                continue
            fac = aug[r][col]
            for j in range(2 * n):
                aug[r][j] -= fac * aug[col][j]

    return [row[n:] for row in aug]


def mat_vec_mul(a, x):
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def read_veff_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "R": int(row["R"]),
                "T": int(row["T"]),
                "Veff": float(row["Veff"]),
                "err": float(row["err"]),
            })
    return rows


def weighted_average(points):
    good = [p for p in points if p["err"] > 0.0]
    ws = [1.0 / (p["err"] * p["err"]) for p in good]
    vbar = sum(w * p["Veff"] for w, p in zip(ws, good)) / sum(ws)
    err = math.sqrt(1.0 / sum(ws))
    chi2 = sum(((p["Veff"] - vbar) / p["err"]) ** 2 for p in good)
    dof = len(good) - 1
    chi2_dof = chi2 / dof if dof > 0 else float("nan")
    return vbar, err, chi2, dof, chi2_dof


def weighted_fit(xs, ys, es, basis_functions):
    npar = len(basis_functions)
    ata = [[0.0 for _ in range(npar)] for _ in range(npar)]
    aty = [0.0 for _ in range(npar)]

    for x, y, e in zip(xs, ys, es):
        w = 1.0 / (e * e)
        phi = [f(x) for f in basis_functions]
        for i in range(npar):
            aty[i] += w * phi[i] * y
            for j in range(npar):
                ata[i][j] += w * phi[i] * phi[j]

    cov = invert_matrix(ata)
    pars = mat_vec_mul(cov, aty)
    errs = [math.sqrt(max(cov[i][i], 0.0)) for i in range(npar)]

    def model(x):
        return sum(pars[i] * basis_functions[i](x) for i in range(npar))

    chi2 = sum(((y - model(x)) / e) ** 2 for x, y, e in zip(xs, ys, es))
    dof = len(xs) - npar
    chi2_dof = chi2 / dof if dof > 0 else float("nan")
    return pars, errs, chi2, dof, chi2_dof, model


def save_plateau_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "T_window", "V", "err", "chi2", "dof", "chi2_dof"])
        for row in rows:
            writer.writerow([
                row["R"],
                ",".join(str(t) for t in row["T_window"]),
                row["V"],
                row["err"],
                row["chi2"],
                row["dof"],
                row["chi2_dof"],
            ])


def main():
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "runs"

    all_plateaus = []
    fit_rows = []

    for case in CASES:
        case_dir = runs_dir / case["name"]
        rows = read_veff_csv(case_dir / "veff_ape.csv")

        plateau_rows = []
        for R in sorted(case["windows"].keys()):
            Ts = case["windows"][R]
            pts = [row for row in rows if row["R"] == R and row["T"] in Ts]
            v, e, chi2, dof, chi2_dof = weighted_average(pts)

            plateau_rows.append({
                "R": R,
                "T_window": Ts,
                "V": v,
                "err": e,
                "chi2": chi2,
                "dof": dof,
                "chi2_dof": chi2_dof,
            })
            all_plateaus.append({
                "case": case["name"],
                "label": case["label"],
                "R": R,
                "V": v,
                "err": e,
            })

        save_plateau_csv(case_dir / "plateau_V_of_R_volumecheck.csv", plateau_rows)

        print(f"\n=== {case['name']} ===")
        print("R   T_window        V               err             chi2/dof")
        for row in plateau_rows:
            print(
                f"{row['R']}   {row['T_window']}   "
                f"{row['V']:.12f}   {row['err']:.12f}   {row['chi2_dof']:.12f}"
            )

        use = [r for r in plateau_rows if r["R"] <= 4]
        xs = [float(r["R"]) for r in use]
        ys = [r["V"] for r in use]
        es = [r["err"] for r in use]

        basis = [
            lambda R: 1.0,
            lambda R: R,
            lambda R: -1.0 / R,
        ]
        pars, errs, chi2, dof, chi2_dof, model = weighted_fit(xs, ys, es, basis)

        fit_rows.append({
            "case": case["name"],
            "label": case["label"],
            "V0": pars[0],
            "err_V0": errs[0],
            "sigma": pars[1],
            "err_sigma": errs[1],
            "alpha": pars[2],
            "err_alpha": errs[2],
            "chi2": chi2,
            "dof": dof,
            "chi2_dof": chi2_dof,
        })

        xfit = [0.8 + (4.2 - 0.8) * i / 200.0 for i in range(201)]
        yfit = [model(x) for x in xfit]

        plt.figure(figsize=(7, 5))
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linestyle="none", label="plateau data")
        plt.plot(xfit, yfit, label=r"$V_0+\sigma R-\alpha/R$")
        plt.xlabel("R")
        plt.ylabel("V(R)")
        plt.title(case["label"] + ": Cornell-type fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(case_dir / "cornell_fit_volumecheck.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 5))
    for case in CASES:
        pts = [p for p in all_plateaus if p["case"] == case["name"]]
        xs = [p["R"] for p in pts]
        ys = [p["V"] for p in pts]
        es = [p["err"] for p in pts]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linestyle="none", label=case["label"])
    plt.xlabel("R")
    plt.ylabel("V(R)")
    plt.title(r"Finite-volume check at $\beta=5.5$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(runs_dir / "beta55_volumecheck_V_of_R.png", dpi=150)

    with open(runs_dir / "beta55_volumecheck_fit_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "label", "V0", "err_V0", "sigma", "err_sigma",
            "alpha", "err_alpha", "chi2", "dof", "chi2_dof"
        ])
        for row in fit_rows:
            writer.writerow([
                row["case"], row["label"], row["V0"], row["err_V0"],
                row["sigma"], row["err_sigma"], row["alpha"], row["err_alpha"],
                row["chi2"], row["dof"], row["chi2_dof"]
            ])

    print("\n=== BETA=5.5 VOLUME CHECK FIT SUMMARY ===")
    print("case                       sigma            err_sigma        alpha            err_alpha        chi2/dof")
    for row in fit_rows:
        print(
            f"{row['case']:<26} "
            f"{row['sigma']:.12f}   "
            f"{row['err_sigma']:.12f}   "
            f"{row['alpha']:.12f}   "
            f"{row['err_alpha']:.12f}   "
            f"{row['chi2_dof']:.12f}"
        )

    print("\nSaved:")
    print(runs_dir / "beta55_volumecheck_V_of_R.png")
    print(runs_dir / "beta55_volumecheck_fit_summary.csv")


if __name__ == "__main__":
    main()
