import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CASES = {
    4.5: {
        "case_dir": Path("runs/scaling_L10_beta_4p5"),
        "windows": {
            1: [2, 3, 4],
            2: [3, 4],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3, 4],
        },
    },
    5.0: {
        "case_dir": Path("runs/scaling_L12_beta_5p0"),
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4, 5],
            3: [2, 3, 4, 5],
            4: [3, 4, 5],
            5: [3, 4],
        },
    },
    5.5: {
        "case_dir": Path("runs/scaling_L14_beta_5p5"),
        "windows": {
            1: [2, 3, 4, 5],
            2: [2, 3, 4],
            3: [2, 3, 4],
            4: [2, 3, 4],
            5: [2, 3],
        },
    },
}

FIT_RMIN = 1
FIT_RMAX = 4


def find_col(df, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise KeyError(f"Could not find columns {candidates}")


def read_veff_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    r_col = find_col(df, ["R", "r", "RR"])
    t_col = find_col(df, ["T", "t", "TT"])
    v_col = find_col(df, ["V", "v", "Veff", "veff"])
    e_col = find_col(df, ["err", "error", "dV", "dveff"])

    out = df[[r_col, t_col, v_col, e_col]].copy()
    out.columns = ["R", "T", "V", "err"]
    out["R"] = out["R"].astype(int)
    out["T"] = out["T"].astype(int)
    out["V"] = out["V"].astype(float)
    out["err"] = out["err"].astype(float)
    return out.sort_values(["R", "T"]).reset_index(drop=True)


def weighted_mean(values: np.ndarray, errors: np.ndarray):
    w = 1.0 / np.square(errors)
    mean = np.sum(w * values) / np.sum(w)
    err = math.sqrt(1.0 / np.sum(w))
    return mean, err


def shift_windows(default_windows, shift, available_t_by_r):
    shifted = {}
    for r, ts in default_windows.items():
        new_ts = [t + shift for t in ts]
        if any(t not in available_t_by_r[r] for t in new_ts):
            return None
        shifted[r] = new_ts
    return shifted


def build_plateau_points(veff_df: pd.DataFrame, windows: dict):
    rows = []
    for r, ts in windows.items():
        sub = veff_df[(veff_df["R"] == r) & (veff_df["T"].isin(ts))].copy()
        sub = sub.sort_values("T")
        if len(sub) != len(ts):
            raise ValueError(f"Missing T points for R={r}, requested {ts}")
        mean_v, err_v = weighted_mean(sub["V"].to_numpy(), sub["err"].to_numpy())
        rows.append(
            {
                "R": r,
                "T_window": ",".join(str(t) for t in ts),
                "V": mean_v,
                "err": err_v,
            }
        )
    return pd.DataFrame(rows).sort_values("R").reset_index(drop=True)


def cornell(R, V0, sigma, alpha):
    return V0 + sigma * R - alpha / R


def fit_cornell(plateau_df: pd.DataFrame, rmin=1, rmax=4):
    sub = plateau_df[(plateau_df["R"] >= rmin) & (plateau_df["R"] <= rmax)].copy()

    R = sub["R"].to_numpy(dtype=float)
    V = sub["V"].to_numpy(dtype=float)
    err = sub["err"].to_numpy(dtype=float)

    p0 = [float(V.min()), 0.1, 0.05]

    popt, pcov = curve_fit(
        cornell,
        R,
        V,
        sigma=err,
        absolute_sigma=True,
        p0=p0,
        maxfev=20000,
    )
    errs = np.sqrt(np.diag(pcov))

    residuals = (V - cornell(R, *popt)) / err
    chi2 = float(np.sum(residuals**2))
    dof = int(len(R) - len(popt))
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return {
        "sigma": popt[1],
        "err_sigma": errs[1],
        "alpha": popt[2],
        "err_alpha": errs[2],
        "chi2_dof": chi2_dof,
    }


def main():
    summary_rows = []

    for beta, info in CASES.items():
        case_dir = info["case_dir"]
        default_windows = info["windows"]
        veff_path = case_dir / "veff_ape.csv"

        veff_df = read_veff_csv(veff_path)
        available_t_by_r = {
            int(r): sorted(group["T"].astype(int).unique().tolist())
            for r, group in veff_df.groupby("R")
        }

        variants = {
            "default": default_windows,
            "shift_Tmin_plus1": shift_windows(default_windows, +1, available_t_by_r),
            "shift_Tmin_minus1": shift_windows(default_windows, -1, available_t_by_r),
        }

        print(f"\n=== beta = {beta} ===")
        for label, windows in variants.items():
            if windows is None:
                print(f"{label}: not valid")
                summary_rows.append(
                    {
                        "beta": beta,
                        "variant": label,
                        "status": "not_valid",
                        "sigma": np.nan,
                        "err_sigma": np.nan,
                        "alpha": np.nan,
                        "err_alpha": np.nan,
                        "chi2_dof": np.nan,
                    }
                )
                continue

            plateau_df = build_plateau_points(veff_df, windows)
            fit = fit_cornell(plateau_df, FIT_RMIN, FIT_RMAX)

            print(
                f"{label}: sigma = {fit['sigma']:.6f} +/- {fit['err_sigma']:.6f}, "
                f"alpha = {fit['alpha']:.6f} +/- {fit['err_alpha']:.6f}, "
                f"chi2/dof = {fit['chi2_dof']:.6f}"
            )

            summary_rows.append(
                {
                    "beta": beta,
                    "variant": label,
                    "status": "ok",
                    "sigma": fit["sigma"],
                    "err_sigma": fit["err_sigma"],
                    "alpha": fit["alpha"],
                    "err_alpha": fit["err_alpha"],
                    "chi2_dof": fit["chi2_dof"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("runs/shifted_window_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary to: {summary_path}")


if __name__ == "__main__":
    main()