import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    z = x @ y
    print(f"Test matmul on MPS succeeded. Result shape: {z.shape}")
else:
    print("MPS not available — will fall back to CPU.")
