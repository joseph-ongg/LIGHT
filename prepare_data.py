from ulab_data_edited import load_data
from curve_to_params import run_one
from predictor import Model
import numpy as np
import os
import torch

train_dataset, _ = load_data("training 3")
test_dataset, _ = load_data("testing 3 cutoff", [train_dataset.pmeans, train_dataset.pstds])

model = Model()
model.load_state_dict(torch.load("best_model_filter.pt", map_location="cpu"))
model.eval()

all_trues, all_preds, all_preds_scipy = [], [], []
for i in range(len(test_dataset.curve_list)):
    path = test_dataset.file_paths[i]
    year = os.path.basename(path).split("_")[1]
    cid = os.path.basename(path).replace(".dat", "").split("_")[2]
    trues, preds, preds_scipy = run_one(
        year, cid,
        test_dataset.curve_list[i],
        test_dataset.first_times[i],
        test_dataset.i_bl_guesses[i],
        model, train_dataset.pmeans, train_dataset.pstds,
        plot=False,
    )
    all_trues.append(trues)
    all_preds.append(preds)
    all_preds_scipy.append(preds_scipy)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_dataset.curve_list)}")

all_trues = np.array(all_trues)
all_preds = np.array(all_preds)
all_preds_scipy = np.array(all_preds_scipy)

np.savez("testdata.npz",
    pmeans=train_dataset.pmeans,
    pstds=train_dataset.pstds,
    file_paths=np.array(test_dataset.file_paths),
    curve_list=np.array(test_dataset.curve_list, dtype=object),
    first_times=np.array(test_dataset.first_times),
    i_bl_guesses=np.array(test_dataset.i_bl_guesses),
    trues=all_trues,
    preds=all_preds,
    preds_scipy=all_preds_scipy,
)
print(f"saved {len(all_trues)} events to testdata.npz")
