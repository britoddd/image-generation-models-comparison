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
z = torch.load("latents/z_mlp_test.pt")["z"]      # (7000, 64)
mu = z.mean(dim=0)                                # (64,)
sigma = z.std(dim=0)                              # (64,)

# 2. Sample z_rand ~ N(mu, diag(sigma²))
eps = torch.randn(N_SAMPLES, 64)                  # ε ~ N(0, I)
z_rand = mu + sigma * eps                         # (16, 64)

# Evidence: real z vs sampled z
print(f"Real z   : {(z == 0).float().mean():.1%} exact zeros, min {z.min():.2f}")
print(f"Sampled z: {(z_rand < 0).float().mean():.1%} negative values")

# 3. Decode with the Tahap 4 decoder
dec = Decoder()
dec.load_state_dict(torch.load("models/decoder_frozen_mlp.pt"))
dec.eval()
with torch.no_grad():
    x_gen = dec(z_rand)                           # (16, 1, 28, 28)

fig, ax = plt.subplots(4, 4, figsize=(6, 6))
for i, a in enumerate(ax.flat):
    a.imshow(x_gen[i, 0], cmap="gray")
    a.axis("off")
fig.suptitle("Prior sampling: z_rand ~ N(μ_z, diag(σ_z²)) → Decoder")
plt.savefig("figures/prior_sampling.png", bbox_inches="tight", dpi=200)
