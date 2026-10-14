"""A minimal PyTorch MLP: non-linear classification with XOR.

Run: python lessons/mlp.py
"""
import torch
from torch import nn

torch.manual_seed(42)

# XOR cannot be solved by a linear classifier.
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.BCEWithLogitsLoss()

for step in range(501):
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step={step:03d} loss={loss.item():.4f}")

with torch.no_grad():
    probabilities = model(x).sigmoid().squeeze()
    print("XOR probabilities:", probabilities.tolist())
    print("XOR predictions: ", (probabilities > 0.5).int().tolist())
