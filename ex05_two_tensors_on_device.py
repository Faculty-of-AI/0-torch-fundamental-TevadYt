"""
Exercise 6 — Create two random tensors of shape (2,3) on the available device.
"""
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.rand(2, 3, device=device)
b = torch.rand(2, 3, device=device)
print("Device:", device)
print("a:\n", a, "\nshape:", a.shape)
print("b:\n", b, "\nshape:", b.shape)
