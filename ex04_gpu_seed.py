"""
Exercise 5 — GPU random seed.
If CUDA is available, set cuda seed to 1234 and print confirmation.
"""
import torch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    print("CUDA is available. Set CUDA seed to 1234.")
else:
    print("CUDA not available on this system; nothing to do.")
