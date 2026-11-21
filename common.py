"""
Shared helpers for every step: seed, device, Decoder/AE models, training and evaluation.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Fixed seed (instruction: torch.manual_seed(42) and np.random.seed(42))
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),  # pixels are in [0,1]
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # mirrors MLP encode()
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit(model, inp, target, epochs=20):
    loader = DataLoader(TensorDataset(inp, target), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total = 0
        for a, b in loader:
            a, b = a.to(device), b.to(device)
            loss = nn.functional.mse_loss(model(a), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | MSE: {total/len(loader):.5f}")
    return model


@torch.no_grad()
def evaluate(model, inp, target):
    model.eval()
    out = model(inp.to(device)).cpu()
    mse = ((out - target) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    return out, mse, psnr
