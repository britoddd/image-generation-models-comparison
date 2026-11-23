"""
Tahap 05: Uji Sampling Generatif pada Ruang Laten MLP Classifier

Hitung μ_z dan σ_z dari z_MLP pada test set, tarik sampel acak
z_rand ~ N(μ_z, diag(σ_z²)), lalu dekodekan dengan Decoder dari Tahap 4.

"""

import torch
import matplotlib.pyplot as plt
from common import Decoder          # importing common also sets the seed

N_SAMPLES = 16

# 1. Latent statistics on the test set
te = torch.load("latents/z_mlp_test.pt")
z, x = te["z"], te["x"]                           # (7000, 64), (7000, 1, 28, 28)
mu = z.mean(dim=0)                                # (64,)
sigma = z.std(dim=0)                              # (64,)

# 2. Sample z_rand ~ N(mu, diag(sigma²))
eps = torch.randn(N_SAMPLES, 64)                  # ε ~ N(0, I)
z_rand = mu + sigma * eps                         # (16, 64)

# Evidence: real z vs sampled z
print(f"Real z   : {(z == 0).float().mean():.1%} exact zeros, min {z.min():.2f}")
print(f"Sampled z: {(z_rand < 0).float().mean():.1%} negative values")

# 3. Decode real z and sampled z with the same Tahap 4 decoder
dec = Decoder()
dec.load_state_dict(torch.load("models/decoder_frozen_mlp.pt"))
dec.eval()
with torch.no_grad():
    x_real = dec(z[:8])                           # (8, 1, 28, 28)  from real z
    x_gen = dec(z_rand)                           # (16, 1, 28, 28) from sampled z

rows = [x[:8], x_real, x_gen[:8]]
labels = ["Original x", "Decoder(z real)", "Decoder(z_rand)"]
fig, ax = plt.subplots(3, 8, figsize=(10, 4))
for r, (imgs, label) in enumerate(zip(rows, labels)):
    for i in range(8):
        ax[r, i].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[r, i].axis("off")
    ax[r, 0].set_title(label, loc="left", fontsize=9)
# fig.suptitle("Real z vs z_rand ~ N(μ_z, diag(σ_z²)), same Decoder")
plt.savefig("figures/prior_sampling.png", bbox_inches="tight", dpi=200)