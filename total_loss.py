import numpy as np
import os
import matplotlib.pyplot as plt
from curve_to_params import one_curve

d = np.load("testdata.npz", allow_pickle=True)
all_preds = d['preds_scipy']
all_trues = d['trues']

ss_res = ((all_preds - all_trues)**2).sum(axis=0)          # shape (5,) — sum down rows
ss_tot = ((all_trues - all_trues.mean(axis=0))**2).sum(axis=0)  # shape (5,)
r2_per_param = 1 - ss_res / ss_tot                  # shape (5,)

fig, axs = plt.subplots(3,2)
axs[0,0].scatter(all_trues[:,0], all_preds[:,0])
axs[0,0].plot([all_trues[:,0].min(), all_trues[:,0].max()], [all_trues[:,0].min(), all_trues[:,0].max()], "r--")
axs[0,0].set_title(f"Tmax: R^2 = {r2_per_param[0]}")
axs[1,0].scatter(all_trues[:,1], all_preds[:,1])
axs[1,0].plot([all_trues[:,1].min(), all_trues[:,1].max()], [all_trues[:,1].min(), all_trues[:,1].max()], "r--")
axs[1,0].set_title(f"tau: R^2 = {r2_per_param[1]}")
axs[2,0].scatter(all_trues[:,2], all_preds[:,2])
axs[2,0].plot([all_trues[:,2].min(), all_trues[:,2].max()], [all_trues[:,2].min(), all_trues[:,2].max()], "r--")
axs[2,0].set_title(f"umin: R^2 = {r2_per_param[2]}")
axs[1,1].scatter(all_trues[:,3], all_preds[:,3])
axs[1,1].plot([all_trues[:,3].min(), all_trues[:,3].max()], [all_trues[:,3].min(), all_trues[:,3].max()], "r--")
axs[1,1].set_title(f"fbl: R^2 = {r2_per_param[3]}")
axs[2,1].scatter(all_trues[:,4], all_preds[:,4])
axs[2,1].plot([all_trues[:,4].min(), all_trues[:,4].max()], [all_trues[:,4].min(), all_trues[:,4].max()], "r--")
axs[2,1].set_title(f"Ibl: R^2 = {r2_per_param[4]}")
plt.tight_layout()
plt.savefig("params.png")
plt.show()

residuals = all_preds-all_trues
fig, axs = plt.subplots(3,2)
axs[0,0].hist(residuals[:,0],bins=40)
axs[0,0].set_title("Tmax residuals")
axs[1,0].hist(residuals[:,1],bins=40)
axs[1,0].set_title("tau residuals")
axs[2,0].hist(residuals[:,2],bins=40)
axs[2,0].set_title("umin residuals")
axs[1,1].hist(residuals[:,3],bins=40)
axs[1,1].set_title("fbl residuals")
axs[2,1].hist(residuals[:,4],bins=40)
axs[2,1].set_title("Ibl residuals")
plt.tight_layout()
plt.savefig("residuals.png")
plt.show()

"""tau_err = np.abs(all_preds[:, 1] - all_trues[:, 1])                                                     
worst = np.argsort(tau_err)[-10:]                                                                       
for i in worst:                                                                                         
    name = os.path.basename(file_paths[i])                                                              
    print(f"{name}: true={all_trues[i,1]:.1f} pred={all_preds[i,1]:.1f} fbl_pred={all_preds[i,3]:.3f} umin_pred={all_preds[i,2]:.3f}")"""



    