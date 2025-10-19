"""
Exercise 3 — Matrix multiply: (7,7) @ (7,1) from a (1,7) vector transposed.
Expected output shape: (7, 1)
"""
import torch
t2 = torch.rand(7, 7)    # (7,7)
v  = torch.rand(1, 7)    # (1,7)
res = t2 @ v.T           # (7,7) @ (7,1) -> (7,1)
print("Result:\n", res)
print("Shape:", res.shape)
