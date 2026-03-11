from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out = root / "runs" / "sigma_vs_L_volumechecks.png"

# beta = 5.0
L_50 = [12, 14, 16]
sigma_50 = [0.12082177943, 0.103539508145, 0.104807362598]
err_50 = [0.001880097459, 0.001454186988, 0.001492809509]

# beta = 5.5
L_55 = [14, 16]
sigma_55 = [0.084367890551, 0.084474772889]
err_55 = [0.001125427265, 0.001109714541]

plt.figure(figsize=(7, 5))
plt.errorbar(L_50, sigma_50, yerr=err_50, marker="o", capsize=3, linestyle="-", label=r"$\beta=5.0$")
plt.errorbar(L_55, sigma_55, yerr=err_55, marker="s", capsize=3, linestyle="-", label=r"$\beta=5.5$")

plt.xlabel("L")
plt.ylabel(r"$\sigma$")
plt.title(r"Finite-volume stability of the Cornell-type string tension")
plt.legend()
plt.tight_layout()
plt.savefig(out, dpi=150)

print(f"Saved: {out}")
