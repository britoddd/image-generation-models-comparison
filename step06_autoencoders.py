"""
Tahap 06: Vanilla Autoencoder dan Sparse Autoencoder

Vanilla AE : L = MSE(x, x̂)
Sparse AE  : L = MSE(x, x̂) + λ‖h‖₁   (penalti L1 pada representasi laten h ∈ ℝ64)
Arsitektur keduanya identik: 784 → 256 → 64 → 256 → 784

"""

import torch
from torch import nn
from common import AE, fit, evaluate, device

LAMBDA = 1e-4   # sparsity strength λ


class SparseAE(AE):
    """Same layers as AE; forward also keeps h so the loss can penalise it."""

    def forward(self, x):
        self.h = self.encoder(x)
        return self.decoder(self.h)


def sparse_loss(model, x, target):
    x_hat = model(x)
    mse = nn.functional.mse_loss(x_hat, target)
    l1 = model.h.abs().sum(dim=1).mean()      # ‖h‖₁ per image, averaged over batch
    return mse + LAMBDA * l1


@torch.no_grad()
def active_fraction(model, x):
    """Share of latent units that are non-zero (> 0) on the test set."""
    model.eval()
    h = model.encoder(x.to(device))
    return (h > 0).float().mean().item()


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    te = torch.load("latents/z_mlp_test.pt")

    for name, model, loss_fn in [
        ("ae", AE(), None),
        ("sparse_ae", SparseAE(), sparse_loss),
    ]:
        print(f"== {name} ==")
        model, log = fit(model, tr["x"], tr["x"], loss_fn=loss_fn)
        _, mse, psnr, ssim = evaluate(model, te["x"], te["x"])
        active = active_fraction(model, te["x"])
        print(f"{name} | MSE {mse:.4f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | active h {active:.1%}")

        torch.save(model.state_dict(), f"models/{name}.pt")
        torch.save(log, f"logs/{name}.pt")


if __name__ == "__main__":
    main()
