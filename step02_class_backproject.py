"""
Tahap 02: Proyeksi Balik Bobot Langsung dari Vektor Kelas (Top-Down Weight Transposition)

Vektor one-hot kelas y_c diproyeksikan balik ke ruang piksel tanpa pelatihan tambahan:
ẑ = ReLU(W3ᵀ y_c),  ĥ1 = ReLU(W2ᵀ ẑ),  x̂ = Sigmoid(W1ᵀ ĥ1)
y_c (10) ──W3ᵀ──> ẑ (64) ──W2ᵀ──> ĥ1 (256) ──W1ᵀ──> x̂ (784)
"""

import torch
import matplotlib.pyplot as plt

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

W = torch.load("weights/weights_W1_W2_W3.pt")
W1, W2, W3 = W["W1"], W["W2"], W["W3"]      # (256,784), (64,256), (10,64)

# One-hot vectors for all 10 classes at once: row c = y_c
y_c = torch.eye(10)                          # (10,10)

z_hat = torch.relu(y_c @ W3)                 # (10,64)
h_hat = torch.relu(z_hat @ W2)               # (10,256)
x_hat = torch.sigmoid(h_hat @ W1)            # (10,784)

print(f"x̂ range: {x_hat.min():.3f} – {x_hat.max():.3f}")

fig, ax = plt.subplots(1, 10, figsize=(15, 2))
for c in range(10):
    ax[c].imshow(x_hat[c].view(28, 28), cmap="gray")
    ax[c].set_title(CLASSES[c], fontsize=8)
    ax[c].axis("off")
plt.savefig("figures/class_backprojection.png", bbox_inches="tight", dpi=200)
