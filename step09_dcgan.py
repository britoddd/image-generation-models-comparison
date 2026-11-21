"""
Tahap 09: Deep Convolutional GAN (DCGAN)

Prinsip DCGAN: strided (transposed) convolution tanpa pooling, tanpa lapisan FC,
BatchNorm, ReLU di G / LeakyReLU di D, Tanh di output, inisialisasi bobot N(0, 0.02).
G: z (100×1×1) → 7×7 → 14×14 → 28×28        D: 28×28 → 14×14 → 7×7 → 1

"""

import torch
from torch import nn
from torchvision.utils import save_image
from common import device
from step08_gan import train_gan, Z_DIM


class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 128, 7, 1, 0), nn.BatchNorm2d(128), nn.ReLU(),  # 1 → 7
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),      # 7 → 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),                            # 14 → 28
        )

    def forward(self, z):
        return self.net(z.view(-1, Z_DIM, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),                             # 28 → 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),      # 14 → 7
            nn.Conv2d(128, 1, 7, 1, 0), nn.Flatten(),                                 # 7 → 1
        )

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    G, D = DCGenerator(), DCDiscriminator()
    G.apply(init_weights)
    D.apply(init_weights)
    G, D, log = train_gan(G, D, tr["x"], epochs=20)

    torch.save(G.state_dict(), "models/dcgan_G.pt")
    torch.save(D.state_dict(), "models/dcgan_D.pt")
    torch.save(log, "logs/dcgan.pt")

    G.eval()
    with torch.no_grad():
        samples = (G(torch.randn(16, Z_DIM, device=device)) + 1) / 2
    save_image(samples, "figures/preview_dcgan.png", nrow=4)


if __name__ == "__main__":
    main()
