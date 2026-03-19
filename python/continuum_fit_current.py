import math
import numpy as np
import matplotlib.pyplot as plt

# Main benchmark sigmas from your controlled fits
# beta, sigma, err_sigma
beta50 = (5.0, 0.10208217794300012, 0.001880097458603782)
beta55 = (5.5, 0.0843678905512597, 0.0011254272647339646)
beta45 = (4.5, 0.14058405121217277, 0.004139782742488154)

# beta = 6.5 combined from L=20 and L=22
sigma65_L20 = 0.059636014474582666
err65_L20 = 0.0005865219707341154
sigma65_L22 = 0.059518785479744096
err65_L22 = 0.0005649131781685578

def weighted_avg(x1, e1, x2, e2):
    w1 = 1.0 / e1**2
    w2 = 1.0 / e2**2
    x = (w1 * x1 + w2 * x2) / (w1 + w2)
    e = math.sqrt(1.0 / (w1 + w2))
    return x, e

sigma65, err65 = weighted_avg(sigma65_L20, err65_L20, sigma65_L22, err65_L22)
beta65 = (6.5, sigma65, err65)

def convert(beta, sigma, err_sigma):
    y = (beta / 4.0) * math.sqrt(sigma)
    dy = (beta / 4.0) * 0.5 / math.sqrt(sigma) * err_sigma
    return y, dy

def fit_line(data):
    x = np.array([1.0 / b for b, s, ds in data], dtype=float)
    y = np.array([convert(b, s, ds)[0] for b, s, ds in data], dtype=float)
    dy = np.array([convert(b, s, ds)[1] for b, s, ds in data], dtype=float)

    w = 1.0 / dy**2
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)

    cov = np.linalg.inv(X.T @ W @ X)
    p = cov @ (X.T @ W @ y)
    errs = np.sqrt(np.diag(cov))

    yfit = X @ p
    chi2 = float(np.sum(((y - yfit) / dy) ** 2))
    dof = len(x) - 2
    chi2_dof = chi2 / dof if dof > 0 else float("nan")

    return {
        "x": x,
        "y": y,
        "dy": dy,
        "c0": p[0],
        "c1": p[1],
        "err_c0": errs[0],
        "err_c1": errs[1],
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }

main_data = [beta50, beta55, beta65]
coarse_check_data = [beta45, beta50, beta55, beta65]

main_fit = fit_line(main_data)
coarse_fit = fit_line(coarse_check_data)

print("=== Converted points ===")
for b, s, ds in [beta45, beta50, beta55, beta65]:
    y, dy = convert(b, s, ds)
    print(f"beta={b:.1f}  sigma={s:.9f} +/- {ds:.9f}   sqrt(sigma)/g^2={y:.9f} +/- {dy:.9f}")

print("\n=== Main continuum fit: beta = 5.0, 5.5, 6.5 ===")
print(f"c0 = {main_fit['c0']:.9f} +/- {main_fit['err_c0']:.9f}")
print(f"c1 = {main_fit['c1']:.9f} +/- {main_fit['err_c1']:.9f}")
print(f"chi2/dof = {main_fit['chi2_dof']:.9f}")

print("\n=== Sensitivity fit including beta = 4.5 ===")
print(f"c0 = {coarse_fit['c0']:.9f} +/- {coarse_fit['err_c0']:.9f}")
print(f"c1 = {coarse_fit['c1']:.9f} +/- {coarse_fit['err_c1']:.9f}")
print(f"chi2/dof = {coarse_fit['chi2_dof']:.9f}")

delta_c0 = abs(main_fit["c0"] - coarse_fit["c0"])
print(f"\nSuggested extrapolation systematic from coarse-point variation: {delta_c0:.9f}")

# plot
x_plot = np.linspace(0.14, 0.23, 200)
y_main = main_fit["c0"] + main_fit["c1"] * x_plot
y_coarse = coarse_fit["c0"] + coarse_fit["c1"] * x_plot

plt.figure(figsize=(6, 4.5))
plt.errorbar(
    [1.0 / b for b, s, ds in [beta45, beta50, beta55, beta65]],
    [convert(b, s, ds)[0] for b, s, ds in [beta45, beta50, beta55, beta65]],
    yerr=[convert(b, s, ds)[1] for b, s, ds in [beta45, beta50, beta55, beta65]],
    fmt="o",
    capsize=3,
    label=r"data",
)
plt.plot(x_plot, y_main, label=r"main fit: $\beta=5.0,5.5,6.5$")
plt.plot(x_plot, y_coarse, label=r"with coarse point $\beta=4.5$")
plt.xlabel(r"$1/\beta$")
plt.ylabel(r"$\sqrt{\sigma}/g^2$")
plt.title(r"Continuum-facing extrapolation of $\sqrt{\sigma}/g^2$")
plt.legend()
plt.tight_layout()
plt.savefig("runs/continuum_extrapolation_sigma.png", dpi=200)

print("\nSaved: runs/continuum_extrapolation_sigma.png")