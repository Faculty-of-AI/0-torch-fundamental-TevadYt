"""
Exercise 10 — Squeeze a (1,1,1,10) tensor to (10).
"""
import torch
x = torch.rand(1, 1, 1, 10)
print("Before:", x.shape)
x_sq = torch.squeeze(x)
print("After:", x_sq.shape)
