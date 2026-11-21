"""
Tahap 10: Denoising Diffusion Probabilistic Model (DDPM)

Forward : x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)
Loss    : L = ‖ε − ε_θ(x_t, t)‖²   (supervised noise prediction)
Reverse : x_{t−1} = (x_t − β_t/√(1−ᾱ_t) · ε_θ) / √α_t + √β_t · z

"""

import torch
from torch import nn
from torchvision.utils import save_image
from common import fit, device

T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class UNet(nn.Module):
    """Small U-Net that predicts the noise ε from (x_t, t)."""

    def __init__(self, ch=64):
        super().__init__()
        self.t1 = nn.Embedding(T, ch)              # learned timestep embedding
        self.t2 = nn.Embedding(T, ch * 2)
        self.down1 = block(1, ch)                  # 28×28
        self.down2 = block(ch, ch * 2)             # 14×14
        self.mid = block(ch * 2, ch * 2)           # 7×7
        self.up2 = block(ch * 4, ch)               # 14×14 (concat skip from down2)
        self.up1 = block(ch * 2, ch)               # 28×28 (concat skip from down1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, t):
        d1 = self.down1(x) + self.t1(t)[:, :, None, None]
        d2 = self.down2(self.pool(d1)) + self.t2(t)[:, :, None, None]
        m = self.mid(self.pool(d2))
        u2 = self.up2(torch.cat([self.upsample(m), d2], dim=1))
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1))
        return self.out(u1)


def ddpm_loss(model, x, _):
    x0 = x * 2 - 1                                  # [0,1] → [-1,1]
    t = torch.randint(0, T, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise   # jump straight to step t
    return nn.functional.mse_loss(model(xt, t), noise)


@torch.no_grad()
def sample(model, n, keep=()):
    """Reverse process x_T → x_0. Returns final images and snapshots at steps in `keep`."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)   # x_T ~ N(0, I)
    trajectory = []
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
        if t in keep:
            trajectory.append(((x.clamp(-1, 1) + 1) / 2).cpu())
    return (x.clamp(-1, 1) + 1) / 2, trajectory


def main():
    tr = torch.load("latents/z_mlp_train.pt")
    model, log = fit(UNet(), tr["x"], tr["x"], loss_fn=ddpm_loss, lr=2e-4)

    torch.save(model.state_dict(), "models/ddpm.pt")
    torch.save(log, "logs/ddpm.pt")

    samples, traj = sample(model, 16, keep={999, 750, 500, 250, 100, 0})
    save_image(samples, "figures/preview_ddpm.png", nrow=4)
    torch.save(traj, "logs/ddpm_trajectory.pt")     # for Figure 5


if __name__ == "__main__":
    main()
