import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from trainer import Model
from torch.nn.utils.rnn import pack_padded_sequence
from data_prep import parse_params


def compute_curve(t, Tmax, tau, umin, fbl, I_bl):
    u = np.sqrt(((t-Tmax)/tau)**2 + umin**2)
    a = (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return I_bl - 2.5 * np.log10(fbl * (a - 1) + 1)


def chi2(params, times, mags, errs):
    Tmax, tau, umin, fbl, I_bl = params
    guess = compute_curve(times, Tmax, tau, umin, fbl, I_bl)
    return np.sum(((mags-guess)/errs)**2)


def chi2_with_prior(params, times, mags, errs, prior, sigma):
    base = chi2(params, times, mags, errs)
    penalty = sum(((p - p0) / s)**2 for p, p0, s in zip(params, prior, sigma))
    return base + penalty

def chi2_with_penalty(params, times, mags, errs, prior, sigma):
    base = chi2(params, times, mags, errs)
    penalty = sum(((p-p0)/s)**2 for p,p0,s in zip(params, prior, sigma))
    return base + penalty

def mse(params, times, mags):
    Tmax, tau, umin, fbl, I_bl = params
    guess = compute_curve(times, Tmax, tau, umin, fbl, I_bl)
    return np.mean((mags-guess)**2)


def run_one(year, cid, cv, ft, ibl, model, pmeans, pstds, plot=False):
    """Core inference + scipy refinement for a single curve. No file loading of testdata.npz."""
    true_params = parse_params(f"testing/{year}/params_{year}_{cid}.dat")
    #true_params[1] = np.exp(true_params[1])
    #true_params[3] = 1/(1+np.exp(-true_params[3]))

    X = torch.tensor(cv, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([X.shape[1]])
    X_packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
    with torch.no_grad():
        pred = model(X_packed)
    pmeans_t = torch.tensor(pmeans, dtype=torch.float32)
    pstds_t = torch.tensor(pstds, dtype=torch.float32)
    pred_real = pred * pstds_t + pmeans_t
    pred_real[0,0] += ft
    pred_real[0,4] += ibl
    #pred_real[0,1] = torch.exp(pred_real[0,1])
    #pred_real[0,3] = torch.sigmoid(pred_real[0,3])
    #print(pred_real)

    data = np.loadtxt(f"testing/{year}/curve_{year}_{cid}.dat", comments=["#","col"])
    import glob as _glob
    _matches = _glob.glob(f"data/{year}/curve_{year}_*{int(cid)}.dat")
    _matches = [m for m in _matches if int(m.rsplit("_", 1)[1].split(".")[0]) == int(cid)]
    fulldata = np.loadtxt(_matches[0], comments=["#","col"])
    times = data[:, 0]
    mags = data[:, 1]
    errs = data[:, 2]

    """window = tau_g*10
    mask = (times > Tmax_g - window) & (times < Tmax_g + window)
    times_f, mags_f, errs_f = times[mask], mags[mask], errs[mask]"""
    prior = pred_real.squeeze().tolist()
    sigma = [5.0, 2.0, 0.02, 0.05, 0.05]
    bounds = [(times.min(), times.max()), (0.1, 200), (1e-4, 1.0), (0.05, 1), (10, 25)]
    result = minimize(chi2_with_prior, prior,
                      args=(times, mags, errs, prior, sigma),
                      method='L-BFGS-B', bounds=bounds)
    #print(result.x)
    """hit = []
    for i, (v, (lo, hi)) in enumerate(zip(result.x, bounds)):
        if abs(v-lo) < 1e-6 * max(1, abs(lo)):
            hit.append(f"{i}=LO({lo})")
        elif abs(v-hi)<1e-6*max(1,abs(hi)):
            hit.append(f"{i}=HI({hi})")
    if hit:
        print(f"  BOUND HIT on params {hit} for {year}_{cid}")"""

    #print("true:", true_params)
    #print("pred:", result.x)
    #print(f"chi2: {chi2(result.x, times, mags, errs)} mse: {mse(pred_real.squeeze().numpy(), times, mags)}")
    if plot:
        lstm_start_time = cv[0, 0] + ft
        fd_mask = fulldata[:, 0] >= lstm_start_time
        fulldata = fulldata[fd_mask]
        t_grid = np.linspace(fulldata[:,0].min(), fulldata[:,0].max(), 500)
        residuals = fulldata[:, 1] - compute_curve(fulldata[:, 0], *result.x)
        peak_mag = cv[:,1].min()
        baseline_mag = 0  # already 0 in cv coords, or use I_bl
        cutoff_mag = cv[-1, 1]
        mag_frac = (cutoff_mag - peak_mag) / (baseline_mag - peak_mag)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axs[0].scatter(fulldata[:,0], fulldata[:,1], color="green", s=15, zorder=3, label="Full Curve")
        axs[0].scatter(cv[:, 0] + ft, cv[:, 1] + ibl, color="blue", s=15, zorder=3, label="LSTM input")
        axs[0].plot(t_grid, compute_curve(t_grid, *result.x), color="orange", label="LSTM + Chi2 Optimizer")
        axs[0].plot(t_grid, compute_curve(t_grid, *pred_real.squeeze().numpy()), color="purple", label="LSTM Only")
        axs[0].set_title(f"Predictions of curve-{year}-{cid}, {50+mag_frac*100.0:.0f}% event observed")
        axs[0].set_ylabel("Magnitude")
        axs[0].invert_yaxis()
        axs[0].legend()
        axs[1].scatter(fulldata[:, 0], residuals, s=10, color="black")
        axs[1].axhline(0, color="red", linestyle="--")
        axs[1].set_xlabel("Time (HJD)")
        axs[1].set_ylabel("Residual (mag)")
        plt.tight_layout()
        plt.savefig("degen.png")
        plt.show()
    return true_params, pred_real.squeeze().numpy(), result.x


def one_curve(year, cid, plot, cutoff=None):
    """CLI wrapper: loads testdata.npz + model, then calls run_one."""
    model = Model()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    d = np.load("testdata.npz", allow_pickle=True)
    pmeans = d['pmeans']; pstds = d['pstds']
    file_paths = d['file_paths']
    curves = d['cut_curves']
    first_times = d['first_times']
    ibls = d['i_bl_guesses']

    target = f"curve_{year}_{cid}.dat"
    idx = next(i for i, p in enumerate(file_paths) if target in p)
    return run_one(year, cid, curves[idx], first_times[idx], ibls[idx],
                   model, pmeans, pstds, plot=plot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cid", required=True)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--cutoff", type=int, default=None)
    args = parser.parse_args()
    one_curve(args.year, args.cid, args.plot, args.cutoff)
