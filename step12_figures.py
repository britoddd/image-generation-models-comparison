"""
Tahap 12: Kompilasi Lima Visualisasi Komparatif Utama (Bagian V)

Fig 1: rekonstruksi AE / Sparse AE / VAE      Fig 2: random sampling 4x4 per model
Fig 3: batasan MLP Classifier (Tahap 2–5)     Fig 4: interpolasi laten linear
Fig 5: training loss curves + DDPM denoising trajectory
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from common import device
from step08_gan import Z_DIM
from step11_evaluate import load_models, generators

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_A, CLASS_B = 1, 7          # Trouser vs Sneaker: two contrasting classes


def show(ax, img, title=None):
    ax.imshow(img.cpu().squeeze(), cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1(m, x):
    x8 = x[:8].to(device)
    rows = {"Original": x8, "Vanilla AE": m["ae"](x8),
            "Sparse AE": m["sparse_ae"](x8), "VAE": m["vae"](x8)}
    fig, ax = plt.subplots(4, 8, figsize=(12, 6.5))
    for r, (name, imgs) in enumerate(rows.items()):
        for i in range(8):
            show(ax[r, i], imgs[i])
        ax[r, 0].set_title(name, loc="left", fontsize=10)
    save(fig, "fig1_reconstruction")


def figure2(gens):
    panels = [("vae", "A. VAE"), ("gan", "B. Vanilla GAN"), ("dcgan", "C. DCGAN"), ("ddpm", "D. DDPM")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    for a, (name, title) in zip(ax, panels):
        show(a, make_grid(gens[name](16), nrow=4, pad_value=1)[0], title)
    save(fig, "fig2_random_sampling")


def figure3():
    panels = [
        ("figures/class_backprojection.png", "(2a) Wᵀ projection from one-hot y_c (Tahap 2)"),
        ("figures/backprojection.png", "(1)+(2b) Original vs Wᵀ projection from z_MLP (Tahap 3)"),
        ("figures/frozen_decoder.png", "(3) Frozen decoder z_MLP vs Vanilla AE z_AE (Tahap 4)"),
        ("figures/prior_sampling.png", "(4) Prior sampling z_rand ~ N(μ_z, diag(σ_z²)) (Tahap 5)"),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for a, (path, title) in zip(ax.flat, panels):
        a.imshow(plt.imread(path))
        a.set_title(title, fontsize=10)
        a.axis("off")
    save(fig, "fig3_mlp_limitations")


def figure4(m, x, y):
    xa = x[y == CLASS_A][:1].to(device)
    xb = x[y == CLASS_B][:1].to(device)
    za, zb = torch.randn(2, 1, Z_DIM, device=device)   # GANs have no encoder → random endpoints
    ends = {
        "Vanilla AE": (m["ae"].encoder(xa), m["ae"].encoder(xb), m["ae"].decoder),
        "Sparse AE": (m["sparse_ae"].encoder(xa), m["sparse_ae"].encoder(xb), m["sparse_ae"].decoder),
        "VAE": (m["vae"].encode(xa)[0], m["vae"].encode(xb)[0], m["vae"].decoder),
        "Vanilla GAN": (za, zb, lambda z: (m["gan"](z) + 1) / 2),
        "DCGAN": (za, zb, lambda z: (m["dcgan"](z) + 1) / 2),
    }
    fig, ax = plt.subplots(5, 6, figsize=(9, 8))
    for r, (name, (z_a, z_b, decode)) in enumerate(ends.items()):
        for c, a in enumerate(ALPHAS):
            show(ax[r, c], decode((1 - a) * z_a + a * z_b)[0], f"α = {a}" if r == 0 else None)
        ax[r, 0].text(-4, 14, name, ha="right", va="center", fontsize=10)
    save(fig, "fig4_interpolation")


def figure5():
    fig, ax = plt.subplot_mosaic(
        [["ae", "sparse_ae", "vae"], ["gan", "dcgan", "ddpm"], ["traj", "traj", "traj"]],
        figsize=(13, 11), height_ratios=[1, 1, 0.9],
    )
    titles = {"ae": "Vanilla AE (MSE)", "sparse_ae": "Sparse AE (MSE + λ‖h‖₁)", "vae": "VAE (BCE + KL)",
              "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM (noise MSE)"}
    for name, title in titles.items():
        log = torch.load(f"logs/{name}.pt")
        if "d_loss" in log:                       # GANs: two curves
            epochs = range(1, len(log["d_loss"]) + 1)
            ax[name].plot(epochs, log["d_loss"], label="Discriminator")
            ax[name].plot(epochs, log["g_loss"], label="Generator")
            ax[name].legend(fontsize=8)
        else:
            ax[name].plot(range(1, len(log["loss"]) + 1), log["loss"])
        ax[name].set(title=title, xlabel="Epoch", ylabel="Loss")

    traj = torch.load("logs/ddpm_trajectory.pt")  # 6 snapshots, each (16,1,28,28)
    strip = torch.cat([torch.cat([s[i, 0] for s in traj], dim=1) for i in range(4)], dim=0)
    ax["traj"].imshow(strip, cmap="gray")
    ax["traj"].axis("off")
    ax["traj"].set_title("DDPM reverse diffusion  x_T → x_0   (t = 999 → 750 → 500 → 250 → 100 → 0)")
    save(fig, "fig5_loss_and_trajectory")


@torch.no_grad()
def main():
    m = load_models()
    te = torch.load("latents/z_mlp_test.pt")
    figure1(m, te["x"])
    figure2(generators(m))
    figure3()
    figure4(m, te["x"], te["y"])
    figure5()


if __name__ == "__main__":
    main()