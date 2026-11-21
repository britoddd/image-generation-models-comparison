"""
Tahap 07: Variational Autoencoder (VAE)

Encoder: x → (μ, log σ²) ∈ ℝ64,  z = μ + σ·ε  (reparameterization trick)
Loss   : L = BCE(x, x̂) + KL( N(μ, σ²) ‖ N(0, I) )

"""

import torch
from torch import nn
from torchvision.utils import save_image
from common import Decoder, fit, evaluate, device

LATENT = 64


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, LATENT)        # no ReLU: μ must be allowed to be negative
        self.fc_logvar = nn.Linear(256, LATENT)
        self.decoder = Decoder()

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        self.mu, self.logvar = self.encode(x)
        if self.training:                          # sample while training
            z = self.mu + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mu)
        else:                                      # use the mean for reconstruction
            z = self.mu
        return self.decoder(z)


def vae_loss(model, x, target):
    x_hat = model(x)
    B = x.size(0)
    recon = nn.functional.binary_cross_entropy(x_hat, target, reduction="sum") / B
    kl = -0.5 * torch.sum(1 + model.logvar - model.mu.pow(2) - model.logvar.exp()) / B
    return recon + kl


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    te = torch.load("latents/z_mlp_test.pt")

    model, log = fit(VAE(), tr["x"], tr["x"], loss_fn=vae_loss)
    _, mse, psnr, ssim = evaluate(model, te["x"], te["x"])
    print(f"vae | MSE {mse:.4f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f}")

    torch.save(model.state_dict(), "models/vae.pt")
    torch.save(log, "logs/vae.pt")

    with torch.no_grad():                          # preview: sample from the prior
        samples = model.decoder(torch.randn(16, LATENT, device=device))
    save_image(samples, "figures/preview_vae.png", nrow=4)


if __name__ == "__main__":
    main()
