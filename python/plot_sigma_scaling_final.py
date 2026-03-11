from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out = root / "runs" / "sigma_vs_beta_final.png"

betas = [4.5, 5.0, 5.5]
sigma = [0.1406, 0.1048, 0.08447]
err = [0.0041, 0.0015, 0.00111]

plt.figure(figsize=(7, 5))
plt.errorbar(betas, sigma, yerr=err, marker="o", capsize=3, linestyle="none")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
plt.savefig(out, dpi=150)

print(f"Saved: {out}")
