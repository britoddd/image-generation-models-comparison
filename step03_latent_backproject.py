"""
Tahap 03: Proyeksi Balik dari Latent Features Citra Uji (Latent Transpose Projection)

Fitur laten z_MLP dari citra uji riil diproyeksikan langsung ke ruang piksel
menggunakan transpos bobot ekstraktor fitur: x̂ = Sigmoid(W1ᵀ ReLU(W2ᵀ z))
z (64) ──W2ᵀ──> h' (256) ──W1ᵀ──> x̂ (784)

"""

import torch
import matplotlib.pyplot as plt

# Load bobot dan representasi
W = torch.load("weights/weights_W1_W2_W3.pt")
W1, W2 = W["W1"], W["W2"]
d = torch.load("latents/z_mlp_test.pt")
z, x, y = d["z"], d["x"], d["y"]

# Back-projection
x_hat = torch.sigmoid(torch.relu(z @ W2) @ W1)

mse = ((x_hat - x.flatten(1)) ** 2).mean()
print(f"Back-projection MSE: {mse:.4f}")

# Proof that Wᵀ ≠ W⁻¹
I = torch.eye(784)
print("‖ W1ᵀW1 − I ‖ =", torch.dist(W1.T @ W1, I).item())
print("rank(W1ᵀW1) =", torch.linalg.matrix_rank(W1.T @ W1).item(), "/ 784")

# Visual: originals (top) vs back-projections (bottom)
fig, ax = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    ax[0, i].imshow(x[i, 0], cmap="gray");                ax[0, i].axis("off")
    ax[1, i].imshow(x_hat[i].view(28, 28), cmap="gray");  ax[1, i].axis("off")
plt.savefig("figures/backprojection.png", bbox_inches="tight")