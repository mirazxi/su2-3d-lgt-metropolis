import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

files = {
    4.5: "runs/scaling_L10_beta_4p5/plateau_V_of_R.csv",
    5.0: "runs/scaling_L12_beta_5p0/plateau_V_of_R.csv",
    5.5: "runs/scaling_L14_beta_5p5/plateau_V_of_R.csv",
}

def linear(R, V0, sigma):
    return V0 + sigma * R

def cornell(R, V0, sigma, alpha):
    return V0 + sigma * R - alpha / R

for beta, path in files.items():
    df = pd.read_csv(path)

    for rmin, rmax in [(2, 5)]:
        sub = df[(df["R"] >= rmin) & (df["R"] <= rmax)].copy()

        R = sub["R"].to_numpy(dtype=float)
        V = sub["V"].to_numpy(dtype=float)
        err = sub["err"].to_numpy(dtype=float)

        popt_lin, pcov_lin = curve_fit(
            linear, R, V, sigma=err, absolute_sigma=True
        )
        Vfit_lin = linear(R, *popt_lin)
        chi2_lin = np.sum(((V - Vfit_lin) / err) ** 2)
        dof_lin = len(R) - len(popt_lin)
        chi2dof_lin = chi2_lin / dof_lin if dof_lin > 0 else np.nan
        err_lin = np.sqrt(np.diag(pcov_lin))

        popt_cor, pcov_cor = curve_fit(
            cornell, R, V, sigma=err, absolute_sigma=True,
            p0=[0.1, 0.1, 0.05], maxfev=20000
        )
        Vfit_cor = cornell(R, *popt_cor)
        chi2_cor = np.sum(((V - Vfit_cor) / err) ** 2)
        dof_cor = len(R) - len(popt_cor)
        chi2dof_cor = chi2_cor / dof_cor if dof_cor > 0 else np.nan
        err_cor = np.sqrt(np.diag(pcov_cor))

        print(f"\nBeta = {beta}, fit range R={rmin}-{rmax}")
        print("  linear:")
        print(f"    sigma = {popt_lin[1]:.6f} +/- {err_lin[1]:.6f}")
        print(f"    chi2/dof = {chi2dof_lin:.6f}")
        print("  Cornell-type:")
        print(f"    sigma = {popt_cor[1]:.6f} +/- {err_cor[1]:.6f}")
        print(f"    alpha = {popt_cor[2]:.6f} +/- {err_cor[2]:.6f}")
        print(f"    chi2/dof = {chi2dof_cor:.6f}")