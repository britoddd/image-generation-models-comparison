'''
Tahap 00: Data splitting

Pembagian dataset menjadi rasio 80/10/10 untuk train/test/split. 
Data yang di-return berupa DataLoader yang siap digunakan untuk melatih model

'''

import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, ConcatDataset

BATCH = 64

def get_data_loaders():
    train_data = FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    )

    test_data = FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    )

    generator = torch.Generator().manual_seed(42)

    full_data = ConcatDataset([train_data, test_data])
    train_subset, val_subset, test_subset = random_split(full_data, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH, shuffle=False)

    return train_loader, val_loader, test_loader