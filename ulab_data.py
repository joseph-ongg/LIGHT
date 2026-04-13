import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import sys

def compute_gap_mask(t): # makes mask for natural gaps in data
    threshold = 50 # most gaps are around 100*(2.5*10e6) so half of that

    dt = np.diff(t)
    g_mask = dt <= threshold
    g_mask = np.concatenate([[True], g_mask])
    return g_mask

class MicrolensingDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # load the raw data
        data = np.loadtxt(self.file_paths[idx])

        # pick a random cut-off index + ensure there is at least some minimum history and some future to predict
        total_length = len(data)

        # find the peak of the event (np.argmin finds the index of the lowest value in an array)
        peak_idx = np.argmin(data[:, 1])

        # set the minimum history to be after the peak
        min_history_after_peak = peak_idx + 5

        # if the file is weird and the peak is at the very end, we can't cut after it.
        if min_history_after_peak >= total_length - 2:
            cutoff_idx = total_length - 1  # Just predict the very last point

        else:
            #pick a random cut-off that is AFTER the peak but before the very end of the file.
            cutoff_idx = random.randint(min_history_after_peak, total_length - 2)

        # split into input (X) and target (y)
        # X: everything up to the cutoff (Shape: [cutoff_idx, 2])
        # y: everything after the cutoff (Shape: [remaining_length, 1])
        X = data[:cutoff_idx, :]
        y = data[cutoff_idx:, 1:2]

        # extract the time column from input data
        time_steps = X[:, 0]

        dt = np.diff(time_steps)

        # call gap mask function
        numpy_gap_mask = compute_gap_mask(time_steps)

        # convert everything to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X_length = X_tensor.shape[0]

        # convert your mask to a boolean tensor
        gap_mask_tensor = torch.tensor(numpy_gap_mask, dtype=torch.bool)

        return X_tensor, y_tensor, X_length, gap_mask_tensor

def pad_collate(batch):
    # separate the batch into individual lists
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    x_lengths = [item[2] for item in batch]

    # pad with zeros
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)

    # convert lengths to a tensor
    x_lengths_tensor = torch.tensor(x_lengths, dtype=torch.int64)

    # Grabs the 4th item from every file in the batch
    gap_masks_list = [item[3] for item in batch]

    # Pads them together
    gap_masks_padded = pad_sequence(gap_masks_list, batch_first=True, padding_value=False)

    return x_padded, y_padded, x_lengths_tensor, gap_masks_padded

#-------

file_path = "C:/Users/lilym/OneDrive/Documents/2014/*.dat"
lightcurves = glob.glob(file_path)

print(f"Successfully found {len(lightcurves)} .dat files!")

# make sure it actually found files
if len(lightcurves) > 0:
    dataset = MicrolensingDataset(lightcurves)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=pad_collate, shuffle=True)

#------------

print("\nGetting graph of original data vs what will be passed onto model")

# create a temporary dataloader WITHOUT shuffling so we can match the file
test_loader = DataLoader(dataset, batch_size=8, collate_fn=pad_collate, shuffle=False)

for X_padded, y_padded, lengths, gap_masks in test_loader:

    # recreate the exact masking logic
    max_len = X_padded.size(1)
    pad_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    final_mask = pad_mask & gap_masks

    # isolate the FIRST sequence (index 0)
    idx = 0
    x_seq = X_padded[idx].numpy()
    p_mask = pad_mask[idx].numpy()
    f_mask = final_mask[idx].numpy()

    hjd = x_seq[:, 0]
    mag = x_seq[:, 1]
    true_len = lengths[idx].item()

    # LOAD THE UNTAINTED ORIGINAL DATA
    # Because shuffle=False, index 0 matches the first file in your dataset list
    original_file = dataset.file_paths[idx]

    # Extracts just the file name from the long folder path
    clean_file_name = os.path.basename(original_file)

    raw_data = np.loadtxt(original_file)
    raw_hjd = raw_data[:, 0]
    raw_mag = raw_data[:, 1]

    # Print stats
    print(f"\n--- Sequence {idx} Statistics ---")

    print(f"Analyzing File:                         {clean_file_name}")

    print(f"Full Original Event Length:             {len(raw_hjd)} points")
    print(f"Cut Training Sequence (before padding): {true_len} points")
    print(f"Valid Points (Kept for Training):       {f_mask.sum()} points")
    print(f"Points removed due to gaps (>30d):      {true_len - f_mask.sum()} points")

    plt.figure(figsize=(12, 6))

    # Plot the FULL original untainted event (Faint Dashed Gray)
    plt.plot(raw_hjd, raw_mag, color='lightgray', linestyle='--', zorder=1,
             label='Hidden Future (Original Untainted Event)')

    # Plot the solid line for the CUT sequence the model actually sees (Solid Black)
    plt.plot(hjd[:true_len], mag[:true_len], color='black', alpha=0.4, zorder=2,
            label='Cut Training Sequence')

    # Plot the VALID data points (Blue)
    plt.scatter(hjd[f_mask], mag[f_mask], color='blue', zorder=4,label='Valid Data (Passed Mask)')

    # Draw RED BARS for every gap
    is_gap = p_mask & (~f_mask)
    gap_indices = np.where(is_gap)[0]

    # We use a flag so we only add 'Data Gap' to the legend once,
    # instead of creating a new legend entry for every single bar.
    added_gap_label = False

    for idx in gap_indices:
        # Draw a vertical red span from the last valid point (idx-1)
        # to the point immediately following the gap (idx)
        plt.axvspan(
            hjd[idx - 1],
            hjd[idx],
            color='red',
            alpha=0.3,
            zorder=3,
            label='Data Gap' if not added_gap_label else ""
        )
        added_gap_label = True

    # Formatting
    plt.gca().invert_yaxis()  # Invert Y axis for magnitudes
    plt.title("Original Data vs Training Data Passed onto Model", fontsize=14)
    plt.xlabel("HJD")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    break  # Stop after the first batch

#---- Asks if you'd like to continue onto training and testing the model

print("Graphing complete!")

# Ask user to continue
response = input("Does everything look correct? Should I move onto training and testing the model (yes/no): ").lower()

if response not in ("yes", "y"):
    print("Stopping execution.")
    sys.exit()

print("Training and testing starting...")

#-------
