"""
Gradient-descent Paczynski fitter (teammate's model) adapted for this repo.

Reuses cutoff curves already saved in testdata.npz (seed 67), fits each event
with Adam, then saves curvefit_params_pred_vs_true.png and a results CSV.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_prep import parse_params

NUM_EPOCHS = 3000
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


class MicrolensingModel(nn.Module):
    def __init__(self, t_obs, m_obs):
        super().__init__()
        baseline = float(np.median(m_obs))
        t0_guess = float(t_obs[np.argmin(m_obs)])
        t_range = max(float(t_obs.max() - t_obs.min()), 1.0)
        tau_guess = max(t_range / 5, 1.0)
        self.t0 = nn.Parameter(torch.tensor(t0_guess, dtype=torch.float32))
        self.log_tE = nn.Parameter(torch.tensor(np.log(tau_guess), dtype=torch.float32))
        self.log_u0 = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))
        self.I0 = nn.Parameter(torch.tensor(baseline, dtype=torch.float32))
        self.fbl_raw = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

    def forward(self, t):
        tE = torch.exp(self.log_tE) + 1e-6
        u0 = torch.exp(self.log_u0) + 1e-6
        fbl = torch.sigmoid(self.fbl_raw)
        u = torch.sqrt(u0**2 + ((t - self.t0) / tE)**2 + 1e-8)
        A = (u**2 + 2) / (u * torch.sqrt(u**2 + 4))
        flux_factor = fbl * A + (1 - fbl)
        return self.I0 - 2.5 * torch.log10(flux_factor + 1e-8)

    def get_params(self):
        with torch.no_grad():
            return {
                "t0": float(self.t0.cpu()),
                "tE": float(torch.exp(self.log_tE).cpu()),
                "u0": float(torch.exp(self.log_u0).cpu()),
                "I0": float(self.I0.cpu()),
                "fbl": float(torch.sigmoid(self.fbl_raw).cpu()),
            }


def fit_one(t_obs, m_obs, num_epochs=NUM_EPOCHS, lr=LR):
    model = MicrolensingModel(t_obs, m_obs).to(device)
    t = torch.tensor(t_obs, dtype=torch.float32, device=device)
    m = torch.tensor(m_obs, dtype=torch.float32, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(num_epochs):
        opt.zero_grad()
        loss = loss_fn(model(t), m)
        loss.backward()
        opt.step()
    return model.get_params(), float(loss.detach().cpu())


def main():
    d = np.load("testdata.npz", allow_pickle=True)
    cut_curves = d["cut_curves"]
    file_paths = d["file_paths"]
    first_times = d["first_times"]
    ibls = d["i_bl_guesses"]

    rows = []
    for i in range(len(cut_curves)):
        cv = cut_curves[i]
        ft = float(first_times[i])
        ibl = float(ibls[i])
        # Convert back to absolute HJD/mag (cv is shifted by ft, ibl)
        t_obs = cv[:, 0] + ft
        m_obs = cv[:, 1] + ibl

        path = file_paths[i]
        base = os.path.basename(path).replace(".dat", "")
        year = base.split("_")[1]
        cid = base.split("_")[2]
        true_p = parse_params(f"testing/{year}/params_{year}_{cid}.dat")

        try:
            fit, final_loss = fit_one(t_obs, m_obs)
        except Exception as e:
            print(f"FAIL {year}_{cid}: {e}")
            continue

        rows.append({
            "event_id": f"{year}_{cid}",
            "year": int(year),
            "final_loss": final_loss,
            "fit_t0": fit["t0"], "fit_tE": fit["tE"], "fit_u0": fit["u0"],
            "fit_fbl": fit["fbl"], "fit_I0": fit["I0"],
            "true_Tmax": true_p[0], "true_tau": true_p[1], "true_umin": true_p[2],
            "true_fbl": true_p[3], "true_I_bl": true_p[4],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cut_curves)}")

    df = pd.DataFrame(rows)
    df.to_csv("gd_fit_results.csv", index=False)
    print(f"saved gd_fit_results.csv with {len(df)} events")

    # Per-param plots
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
    plt.show()
    print("saved curvefit_params_pred_vs_true.png")


if __name__ == "__main__":
    main()
