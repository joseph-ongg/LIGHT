import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("testing/curve_2024_053.dat")                                                                       
times = data[:, 0]
mags = data[:, 1] 
errs = data[:, 2]  
t_err = mags + errs/2
b_err = mags - errs/2

I_bl_g = np.median(mags[:5])
fbl_g = 1
Tmax_g = times[np.argmin(mags)]
tau_g = (times[-1]-times[0])/6
umin_g = 1.0/(10**((I_bl_g-np.min(mags))/2.5))

def compute_curve(t, Tmax, tau, umin, fbl, I_bl):
    u = np.sqrt(((t-Tmax)/tau)**2 + umin**2)
    a = (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return I_bl - 2.5 * np.log10(fbl * (a - 1) + 1)

def chi2(params):
    Tmax, tau, umin, fbl, I_bl = params
    guess = compute_curve(times, Tmax, tau, umin, fbl, I_bl)
    return np.sum(((mags-guess)/errs)**2)

result = minimize(chi2, [Tmax_g, tau_g, umin_g, fbl_g, I_bl_g], method = 'Nelder-Mead')
print(result.x)
plt.errorbar(times, mags, yerr = [t_err-b_err], fmt="o")
plt.show()