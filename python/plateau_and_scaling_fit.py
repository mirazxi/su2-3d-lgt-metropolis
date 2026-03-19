from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    {
        "name": "scaling_L10_beta_4p5",
        "beta": 4.5,
        "windows": {
            1: [2, 3, 4],
            2: [3, 4],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3, 4],
        },
    },
    {
        "name": "scaling_L12_beta_5p0",
        "beta": 5.0,
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4, 5],
            3: [2, 3, 4, 5],
            4: [3, 4, 5],
            5: [3, 4],
        },
    },
    {
        "name": "scaling_L14_beta_5p5",
        "beta": 5.5,
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3],
        },
    },
    {
       "name": "scaling_L16_beta_6p0",
       "beta": 6.0,
       "windows": {
        1: [2, 3, 4, 5],
        2: [2, 3, 4, 5],
        3: [2, 3, 4, 5],
        4: [3, 4, 5],
        5: [3, 4, 5],
        },
    },
    {
    "name": "scaling_L18_beta_6p0",
    "beta": 6.0,
    "windows": {
        1: [3, 4, 5],
        2: [3, 4, 5],
        3: [3, 4, 5],
        4: [3, 4, 5],
        5: [3, 4, 5],
        },
    },
    {
    "name": "scaling_L20_beta_6p5",
    "beta": 6.5,
    "windows": {
        1: [3, 4, 5],
        2: [3, 4, 5],
        3: [3, 4, 5],
        4: [3, 4, 5],
        5: [3, 4, 5],
        },
    },
    {
    "name": "scaling_L22_beta_6p5",
    "beta": 6.5,
    "windows": {
        1: [3, 4, 5],
        2: [3, 4, 5],
        3: [3, 4, 5],
        4: [3, 4, 5],
        5: [3, 4, 5],
    },
},
]

def mean(xs):
    return sum(xs) / len(xs)


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
            raise RuntimeError("Matrix inversion failed: singular matrix.")
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

    inv = []
    for i in range(n):
        inv.append(aug[i][n:])
    return inv


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
    if not good:
        return None

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

    for case in CASES:
        case_dir = runs_dir / case["name"]
        rows = read_veff_csv(case_dir / "veff_ape.csv")

        plateau_rows = []
        for R in sorted(case["windows"].keys()):
            Ts = case["windows"][R]
            points = [row for row in rows if row["R"] == R and row["T"] in Ts]
            res = weighted_average(points)
            if res is None:
                continue

            v, e, chi2, dof, chi2_dof = res
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
                "beta": case["beta"],
                "R": R,
                "V": v,
                "err": e,
            })

        save_plateau_csv(case_dir / "plateau_V_of_R.csv", plateau_rows)

        print(f"\n=== {case['name']} ===")
        print("R   T_window      V               err             chi2/dof")
        for row in plateau_rows:
            print(
                f"{row['R']}   {row['T_window']}   "
                f"{row['V']:.12f}   {row['err']:.12f}   {row['chi2_dof']:.12f}"
            )

    # combined V(R) plot
    plt.figure(figsize=(8, 5))
    for case in CASES:
        pts = [p for p in all_plateaus if p["case"] == case["name"]]
        xs = [p["R"] for p in pts]
        ys = [p["V"] for p in pts]
        es = [p["err"] for p in pts]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linestyle="none",
                     label=fr"$\beta={case['beta']}$")
    plt.xlabel("R")
    plt.ylabel("V(R)")
    plt.title("Plateau-extracted static potential from APE data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(runs_dir / "scaling_V_of_R_all_betas.png", dpi=150)

    # fits
    fit_rows = []

    for case in CASES:
        pts = [p for p in all_plateaus if p["case"] == case["name"]]

        for tag, rmax in [("main_R1to4", 4), ("stress_R1to5", 5)]:
            use = [p for p in pts if p["R"] <= rmax]
            xs = [float(p["R"]) for p in use]
            ys = [p["V"] for p in use]
            es = [p["err"] for p in use]

            if len(xs) < 2:
                continue

            # linear: V0 + sigma R
            lin_basis = [
                lambda R: 1.0,
                lambda R: R,
            ]
            lin_pars, lin_errs, lin_chi2, lin_dof, lin_chi2_dof, lin_model = weighted_fit(xs, ys, es, lin_basis)

            fit_rows.append({
                "case": case["name"],
                "beta": case["beta"],
                "fit_type": "linear",
                "range_tag": tag,
                "V0": lin_pars[0],
                "err_V0": lin_errs[0],
                "sigma": lin_pars[1],
                "err_sigma": lin_errs[1],
                "alpha": float("nan"),
                "err_alpha": float("nan"),
                "chi2": lin_chi2,
                "dof": lin_dof,
                "chi2_dof": lin_chi2_dof,
            })

            # linear + Coulomb-like: V0 + sigma R - alpha/R
            if len(xs) >= 3:
                coul_basis = [
                    lambda R: 1.0,
                    lambda R: R,
                    lambda R: -1.0 / R,
                ]
                c_pars, c_errs, c_chi2, c_dof, c_chi2_dof, c_model = weighted_fit(xs, ys, es, coul_basis)

                fit_rows.append({
                    "case": case["name"],
                    "beta": case["beta"],
                    "fit_type": "linear_plus_invR",
                    "range_tag": tag,
                    "V0": c_pars[0],
                    "err_V0": c_errs[0],
                    "sigma": c_pars[1],
                    "err_sigma": c_errs[1],
                    "alpha": c_pars[2],
                    "err_alpha": c_errs[2],
                    "chi2": c_chi2,
                    "dof": c_dof,
                    "chi2_dof": c_chi2_dof,
                })

            # plot main fit only
            if tag == "main_R1to4":
                xmin = 0.8
                xmax = max(xs) + 0.2
                xfit = [xmin + (xmax - xmin) * i / 200.0 for i in range(201)]
                yfit = [lin_model(x) for x in xfit]

                plt.figure(figsize=(7, 5))
                plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linestyle="none", label="plateau data")
                plt.plot(xfit, yfit, label=r"$V_0+\sigma R$")
                plt.xlabel("R")
                plt.ylabel("V(R)")
                plt.title(f"{case['name']}: linear fit (R<=4)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(runs_dir / case["name"] / "string_tension_linear_fit_R1to4.png", dpi=150)
                plt.close()

    with open(runs_dir / "scaling_fit_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "beta", "fit_type", "range_tag",
            "V0", "err_V0", "sigma", "err_sigma",
            "alpha", "err_alpha", "chi2", "dof", "chi2_dof"
        ])
        for row in fit_rows:
            writer.writerow([
                row["case"], row["beta"], row["fit_type"], row["range_tag"],
                row["V0"], row["err_V0"], row["sigma"], row["err_sigma"],
                row["alpha"], row["err_alpha"], row["chi2"], row["dof"], row["chi2_dof"]
            ])

    print("\n=== FIT SUMMARY ===")
    print("case                 beta   fit_type            range        sigma            err_sigma        chi2/dof")
    for row in fit_rows:
        print(
            f"{row['case']:<20} "
            f"{row['beta']:.1f}   "
            f"{row['fit_type']:<18} "
            f"{row['range_tag']:<12} "
            f"{row['sigma']:.12f}   "
            f"{row['err_sigma']:.12f}   "
            f"{row['chi2_dof']:.12f}"
        )

    # sigma vs beta plot from main linear fit
    main_linear = [
        row for row in fit_rows
        if row["fit_type"] == "linear" and row["range_tag"] == "main_R1to4"
    ]
    main_linear.sort(key=lambda r: r["beta"])

    betas = [r["beta"] for r in main_linear]
    sigmas = [r["sigma"] for r in main_linear]
    dsigmas = [r["err_sigma"] for r in main_linear]

    plt.figure(figsize=(7, 5))
    plt.errorbar(betas, sigmas, yerr=dsigmas, marker="o", capsize=3, linestyle="none")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma$")
    plt.title(r"String tension from APE plateau fits")
    plt.tight_layout()
    plt.savefig(runs_dir / "sigma_vs_beta.png", dpi=150)

    print("\nSaved:")
    print(runs_dir / "scaling_V_of_R_all_betas.png")
    print(runs_dir / "scaling_fit_summary.csv")
    print(runs_dir / "sigma_vs_beta.png")


if __name__ == "__main__":
    main()
