"""
Tahap 08: Vanilla GAN (Fully-Connected / MLP)

G: z ∈ ℝ100 ~ N(0, I) → 784 (Tanh)        D: 784 → 1 (logit real/fake)
min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]

"""

import time
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from common import device

Z_DIM = 100
EPOCHS = 30


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh(),        # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),                     # raw logit, no Sigmoid
        )

    def forward(self, x):
        return self.net(x)


def train_gan(G, D, x, epochs=EPOCHS):
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    G.to(device)
    D.to(device)
    log = {"d_loss": [], "g_loss": [], "epoch_time": []}

    for epoch in range(epochs):
        G.train()
        D.train()
        start = time.perf_counter()
        d_total, g_total = 0, 0
        for (real,) in loader:
            real = real.to(device) * 2 - 1        # [0,1] → [-1,1] to match Tanh
            B = real.size(0)
            ones = torch.ones(B, 1, device=device)
            zeros = torch.zeros(B, 1, device=device)
            fake = G(torch.randn(B, Z_DIM, device=device))

            # 1. Discriminator: real → 1, fake → 0
            d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2. Generator: make D call the fakes "real"
            g_loss = bce(D(fake), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_total += d_loss.item()
            g_total += g_loss.item()

        log["d_loss"].append(d_total / len(loader))
        log["g_loss"].append(g_total / len(loader))
        log["epoch_time"].append(time.perf_counter() - start)
        print(f"Epoch {epoch+1}/{epochs} | D: {log['d_loss'][-1]:.4f} | G: {log['g_loss'][-1]:.4f} | {log['epoch_time'][-1]:.1f}s")
    return G, D, log


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    G, D, log = train_gan(Generator(), Discriminator(), tr["x"])

    torch.save(G.state_dict(), "models/gan_G.pt")
    torch.save(D.state_dict(), "models/gan_D.pt")
    torch.save(log, "logs/gan.pt")

    G.eval()
    with torch.no_grad():
        samples = (G(torch.randn(16, Z_DIM, device=device)) + 1) / 2   # back to [0,1]
    save_image(samples, "figures/preview_gan.png", nrow=4)


if __name__ == "__main__":
    main()
