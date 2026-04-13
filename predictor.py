import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32,num_layers=2)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    pred = model(X)
    loss = ((pred-Y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()