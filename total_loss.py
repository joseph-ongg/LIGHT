import numpy as np
import os
from curve_to_params import one_curve

d = np.load("test_data_newdata.npz", allow_pickle=True)
pmeans = d['pmeans']; pstds = d['pstds']
file_paths = d['file_paths']
curves = d['curve_list']
first_times = d['first_times']
ibls = d['i_bl_guesses']
all_trues, all_preds = [], []

for i in range(len(curves)):
    path = file_paths[i]
    year, cid = os.path.basename(path).split("_")[1], os.path.basename(path).replace(".dat", "").split("_")[2]
    trues, preds = one_curve(year, cid, False)
    all_trues.append(trues)
    all_preds.append(preds)
all_trues = np.array(all_trues)
all_preds = np.array(all_preds)

ss_res = ((all_preds - all_trues)**2).sum(axis=0)          # shape (5,) — sum down rows
ss_tot = ((all_trues - all_trues.mean(axis=0))**2).sum(axis=0)  # shape (5,)
r2_per_param = 1 - ss_res / ss_tot                  # shape (5,)

names = ["Tmax", "tau", "umin", "fbl", "I_bl"]
for n, r in zip(names, r2_per_param):
    print(f"{n}: {r:.3f}")
"""tau_err = np.abs(all_preds[:, 1] - all_trues[:, 1])                                                     
worst = np.argsort(tau_err)[-10:]                                                                       
for i in worst:                                                                                         
    name = os.path.basename(file_paths[i])                                                              
    print(f"{name}: true={all_trues[i,1]:.1f} pred={all_preds[i,1]:.1f} fbl_pred={all_preds[i,3]:.3f} umin_pred={all_preds[i,2]:.3f}")"""



    