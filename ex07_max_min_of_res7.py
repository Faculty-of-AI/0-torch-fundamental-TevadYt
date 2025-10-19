"""
Exercise 8 — Max and Min of the output of (7).
We recompute the same matmul as in (7) to keep this file standalone.
"""
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.rand(2, 3, device=device)
b = torch.rand(2, 3, device=device)
res = a @ b.T  # (2,2)
print("max:", torch.max(res).item(), "min:", torch.min(res).item())
