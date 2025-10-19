"""
Exercise 7 — Matrix multiply tensors from (6).
We'll recreate a (2,3) and (2,3), then transpose the second to (3,2):
(2,3) @ (3,2) -> (2,2)
"""
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.rand(2, 3, device=device)
b = torch.rand(2, 3, device=device)
res = a @ b.T
print("Result:\n", res)
print("Shape:", res.shape)
