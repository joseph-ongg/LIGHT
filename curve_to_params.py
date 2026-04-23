import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from predictor import Model
from torch.nn.utils.rnn import pack_padded_sequence
from ulab_data_edited import parse_params


def compute_curve(t, Tmax, tau, umin, fbl, I_bl):
    u = np.sqrt(((t-Tmax)/tau)**2 + umin**2)
    a = (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return I_bl - 2.5 * np.log10(fbl * (a - 1) + 1)


def chi2(params, times, mags, errs):
    Tmax, tau, umin, fbl, I_bl = params
    guess = compute_curve(times, Tmax, tau, umin, fbl, I_bl)
    return np.sum(((mags-guess)/errs)**2)


def mse(params, times, mags):
    Tmax, tau, umin, fbl, I_bl = params
    guess = compute_curve(times, Tmax, tau, umin, fbl, I_bl)
    return np.mean((mags-guess)**2)


def one_curve(year, cid, plot, cutoff=None):
    model = Model()
    model.load_state_dict(torch.load("best_model_422.pt"))
    model.eval()
    d = np.load("test_data.npz", allow_pickle=True)
    pmeans = d['pmeans']; pstds = d['pstds']
    file_paths = d['file_paths']
    curves = d['curve_list']
    first_times = d['first_times']
    ibls = d['i_bl_guesses']

    target = f"curve_{year}_{cid}.dat"
    idx = next(i for i, p in enumerate(file_paths) if target in p)
    cv = curves[idx]
    ft = first_times[idx]
    ibl = ibls[idx]
    true_params = parse_params(f"testing 2/{year}/params_{year}_{cid}.dat")
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
    print(pred_real)

    data = np.loadtxt(f"testing 2/{year}/curve_{year}_{cid}.dat", comments=["#","col"])
    import glob as _glob
    _matches = _glob.glob(f"data/{year}/curve_{year}_*{int(cid)}.dat")
    _matches = [m for m in _matches if int(m.rsplit("_", 1)[1].split(".")[0]) == int(cid)]
    fulldata = np.loadtxt(_matches[0], comments=["#","col"])
    times = data[:, 0]
    mags = data[:, 1]
    errs = data[:, 2]

    Tmax_g, tau_g, umin_g, fbl_g, I_bl_g = pred_real.squeeze().tolist()
    """window = tau_g*10
    mask = (times > Tmax_g - window) & (times < Tmax_g + window)
    times_f, mags_f, errs_f = times[mask], mags[mask], errs[mask]"""
    bounds = [(times.min(), times.max()), (0.1, 200), (1e-4, 1.5), (0.05, 1), (10, 25)]
    result = minimize(chi2, [Tmax_g, tau_g, umin_g, fbl_g, I_bl_g],
                      args=(times, mags, errs),
                      method='L-BFGS-B', bounds=bounds)
    print(result.x)
    """hit = []
    for i, (v, (lo, hi)) in enumerate(zip(result.x, bounds)):
        if abs(v-lo) < 1e-6 * max(1, abs(lo)):
            hit.append(f"{i}=LO({lo})")
        elif abs(v-hi)<1e-6*max(1,abs(hi)):
            hit.append(f"{i}=HI({hi})")
    if hit:
        print(f"  BOUND HIT on params {hit} for {year}_{cid}")"""

    print("true:", true_params)
    print("pred:", result.x)
    print(f"chi2: {chi2(result.x, times, mags, errs)} mse: {mse(pred_real.squeeze().numpy(), times, mags)}")
    if plot:
        t_grid = np.linspace(fulldata[:,0].min(), fulldata[:,0].max(), 500)
        plt.scatter(fulldata[:,0], fulldata[:,1], color="green",s=15, zorder=3, label="Full Curve")
        plt.scatter(cv[:, 0] + ft, cv[:, 1] + ibl, color="blue", s=15, zorder=3, label="LSTM input")
        plt.plot(t_grid, compute_curve(t_grid, *result.x), color="orange", label = "LSTM + SciPy")
        plt.plot(t_grid, compute_curve(t_grid, *pred_real.squeeze().numpy()), color="purple",label="LSTM Only")
        plt.title(f"Predictions of curve-{year}-{cid}")
        plt.xlabel("Time (HJD)")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig("degen.png")
        plt.show()
    return true_params, result.x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cid", required=True)
    parser.add_argument("--plot", type=bool, required=True)
    parser.add_argument("--cutoff", type=int, default=None)
    args = parser.parse_args()
    one_curve(args.year, args.cid, args.plot, args.cutoff)
