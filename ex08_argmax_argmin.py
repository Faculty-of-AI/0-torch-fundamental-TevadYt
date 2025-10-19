"""
Exercise 9 — Argmax and Argmin (indices) of the output of (7).
We recompute the matmul here to keep it self-contained.
"""
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.rand(2, 3, device=device)
b = torch.rand(2, 3, device=device)
res = a @ b.T  # (2,2)

idx_max = torch.argmax(res)
idx_min = torch.argmin(res)
print("argmax (flattened index):", idx_max.item())
print("argmin (flattened index):", idx_min.item())

# Row/col positions
max_pos = torch.nonzero(res == res.max(), as_tuple=False)[0].tolist()
min_pos = torch.nonzero(res == res.min(), as_tuple=False)[0].tolist()
print("argmax (row, col):", tuple(max_pos))
print("argmin (row, col):", tuple(min_pos))
