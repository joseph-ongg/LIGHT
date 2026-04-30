"""Re-plot curvefit_params_pred_vs_true.png from saved gd_fit_results.csv."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("gd_fit_results.csv")

pairs = [
    ("Tmax",  "true_Tmax", "fit_t0",  0),
    ("tau",   "true_tau",  "fit_tE",  2),
    ("umin",  "true_umin", "fit_u0",  4),
    ("fbl",   "true_fbl",  "fit_fbl", 3),
    ("I_bl",  "true_I_bl", "fit_I0",  5),
]

def r2(yt, yp):
    ss_res = np.sum((yp - yt)**2)
    ss_tot = np.sum((yt - yt.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

fig, axs = plt.subplots(3, 2, figsize=(6.4, 4.8))
axs = axs.flatten()
axs[1].axis("off")
for name, tcol, fcol, idx in pairs:
    yt = df[tcol].values
    yp = df[fcol].values
    ax = axs[idx]
    ax.scatter(yt, yp, s=18, alpha=0.6)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Fitted {name}")
    ax.set_title(f"{name}: R^2 = {r2(yt, yp):.3f}")
plt.suptitle("Gradient-descent fitter: fitted vs true", fontsize=16)
plt.tight_layout()
plt.savefig("curvefit_params_pred_vs_true.png", dpi=100)
print("saved curvefit_params_pred_vs_true.png")
