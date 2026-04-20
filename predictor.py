from ulab_data_edited import MicrolensingDataset, pad_collate, compute_gap_mask, load_data
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64,num_layers=3,dropout=0.2)
        self.fc1 = nn.Linear(32,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,5)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc1(hidden[-1])
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = Model()
train_dataset, train_dataloader = load_data("training")
test_dataset, test_dataloader = load_data("testing 2", [train_dataset.pmeans, train_dataset.pstds])
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
best_test_loss = float("inf")

#train
for epoch in range(50):
    model.train()
    epoch_loss = 0
    n_batches = 0
    for X_padded, y_stacked, lengths, gap_masks, ft_stacked, ibl_stacked in train_dataloader:
        X = pack_padded_sequence(X_padded, lengths, batch_first = True, enforce_sorted=False)
        pred = model(X)
        loss = ((pred-y_stacked) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for X_padded, y_stacked, lengths, gap_masks, ft_stacked, ibl_stacked in test_dataloader:
            X = pack_padded_sequence(X_padded, lengths, batch_first=True, enforce_sorted=False)
            pred = model(X)
            test_loss += ((pred - y_stacked) ** 2).mean().item()
            test_batches += 1
    avg_test = test_loss/test_batches
    print(f"Epoch {epoch}: train = {epoch_loss/n_batches:.4f}, test = {test_loss/test_batches:.4f}")
    if avg_test < best_test_loss:
          best_test_loss = avg_test
          torch.save(model.state_dict(), "best_model.pt")
          print(f"  saved new best (test = {avg_test:.4f})")


#test
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
total_loss = 0
batch_count = 0
pmeans_t = torch.tensor(test_dataset.pmeans, dtype=torch.float32)                                                      
pstds_t = torch.tensor(test_dataset.pstds, dtype=torch.float32)
per_param_sq = torch.zeros(5)

with torch.no_grad():
    for X_padded, y_stacked, lengths, gap_masks, ft_stacked, ibl_stacked in test_dataloader:
        X = pack_padded_sequence(X_padded, lengths, batch_first = True, enforce_sorted=False)
        pred = model(X)
        loss = ((pred-y_stacked) ** 2).mean()
        total_loss += loss.item()
        per_param_sq += ((pred-y_stacked)**2).mean(dim=0)
        batch_count += 1                                                                                       
    pred_real = pred * pstds_t + pmeans_t                                                                                   
    pred_real[:, 0] += ft_stacked
    pred_real[:, 4] += ibl_stacked
    true_real = y_stacked * pstds_t + pmeans_t
    true_real[:, 0] += ft_stacked
    true_real[:, 4] += ibl_stacked

    for i in range(len(pred_real)):
        print(f"  pred: {pred_real[i].numpy()}")
        print(f"  true: {true_real[i].numpy()}")
        print()
        

print("Test loss (avg):", total_loss/batch_count)
print("Per-param MSE:", per_param_sq/batch_count)

