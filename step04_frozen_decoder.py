"""
Tahap 04: Evaluasi Kandungan Informasi Latent Features (Frozen Latent Decoding)

Seluruh bobot MLP Classifier dibekukan (requires_grad=False). Hanya Decoder MLP
(64 → 256 → 784) yang dilatih dengan MSE Loss untuk merekonstruksi citra dari z_MLP,
lalu dibandingkan (MSE, PSNR, SSIM) dengan Vanilla AE berdimensi laten sama (z_AE ∈ ℝ64).

"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from common import Decoder, AE, fit, evaluate
from step01_train_classifier import MLPClassifier


class FrozenMLPDecoder(nn.Module):
    """x → frozen MLP encode() → z_MLP (64) → trainable Decoder → x̂"""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.mlp.encode(x))


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    te = torch.load("latents/z_mlp_test.pt")

    mlp = MLPClassifier()  # encoder from Tahap 1, frozen below
    mlp.load_state_dict(torch.load("models/mlp_classifier.pt"))
    mlp.requires_grad_(False)  # freeze every MLP weight
    W1_before = mlp.linear1.weight.clone()

    print("== Decoder on frozen Z_MLP ==")
    frozen, _ = fit(FrozenMLPDecoder(mlp), tr["x"], tr["x"])  # added decoder
    print("== Autoencoder (Z_AE) ==")
    ae, _ = fit(AE(), tr["x"], tr["x"])

    # Proof that the MLP weights never changed
    assert torch.equal(mlp.linear1.weight.cpu(), W1_before), "MLP was updated!"

    out_mlp, mse_mlp, psnr_mlp, ssim_mlp = evaluate(frozen, te["x"], te["x"])
    out_ae, mse_ae, psnr_ae, ssim_ae = evaluate(ae, te["x"], te["x"])
    print(f"Z_MLP | MSE {mse_mlp:.4f} | PSNR {psnr_mlp:.2f} dB | SSIM {ssim_mlp:.4f}")
    print(f"Z_AE  | MSE {mse_ae:.4f} | PSNR {psnr_ae:.2f} dB | SSIM {ssim_ae:.4f}")

    torch.save(frozen.decoder.state_dict(), "models/decoder_frozen_mlp.pt")
    torch.save(ae.state_dict(), "models/ae.pt")

    fig, ax = plt.subplots(3, 8, figsize=(12, 4.5))
    for i in range(8):
        for r, img in enumerate([te["x"], out_mlp, out_ae]):
            ax[r, i].imshow(img[i, 0], cmap="gray")
            ax[r, i].axis("off")
    for r, name in enumerate(["Original", "Z_MLP", "Z_AE"]):
        ax[r, 0].set_title(name, loc="left", fontsize=9)
    plt.savefig("figures/frozen_decoder.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
