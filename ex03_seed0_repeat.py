"""
Exercise 4 — Set random seed to 0 and redo 2 & 3
"""
import torch
torch.manual_seed(0)
t2 = torch.rand(7, 7)
v  = torch.rand(1, 7)
res = t2 @ v.T
print("Seeded result shape:", res.shape)
print("First few values (flattened):", res.flatten()[:5])
