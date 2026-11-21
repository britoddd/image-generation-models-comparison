"""
Tahap 11: Kompilasi Kuantitatif (Tabel Bagian IV)

Untuk keenam model: dimensi laten, training time/epoch, latency (ms / batch 64),
Test MSE / PSNR / SSIM (model rekonstruksi), dan FID (model generatif).

"""

import json
import time
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from common import AE, evaluate, device
from step06_autoencoders import SparseAE
from step07_vae import VAE, LATENT
from step08_gan import Generator, Z_DIM
from step09_dcgan import DCGenerator
from step10_ddpm import UNet, sample

N_FID = 5000        # same number of samples for every model → fair comparison

NAMES = {
    "ae": "Vanilla Autoencoder", "sparse_ae": "Sparse Autoencoder", "vae": "VAE",
    "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM",
}
LATENT_DIM = {"ae": 64, "sparse_ae": 64, "vae": LATENT, "gan": Z_DIM, "dcgan": Z_DIM, "ddpm": "28x28"}


def load(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def load_models():
    return {
        "ae": load(AE(), "models/ae.pt"),
        "sparse_ae": load(SparseAE(), "models/sparse_ae.pt"),
        "vae": load(VAE(), "models/vae.pt"),
        "gan": load(Generator(), "models/gan_G.pt"),
        "dcgan": load(DCGenerator(), "models/dcgan_G.pt"),
        "ddpm": load(UNet(), "models/ddpm.pt"),
    }


def generators(m):
    """For each generative model: n → n new images in [0,1], sampled from noise."""
    return {
        "vae": lambda n: m["vae"].decoder(torch.randn(n, LATENT, device=device)),
        "gan": lambda n: (m["gan"](torch.randn(n, Z_DIM, device=device)) + 1) / 2,
        "dcgan": lambda n: (m["dcgan"](torch.randn(n, Z_DIM, device=device)) + 1) / 2,
        "ddpm": lambda n: sample(m["ddpm"], n)[0],
    }


def latency_ms(fn, reps=20):
    fn()                                        # warm-up (first call is always slow)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()                # wait until the GPU has really finished
    return (time.perf_counter() - start) / reps * 1000


def fid_score(gen, real, n=N_FID, bs=500):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for i in range(0, n, bs):
        k = min(bs, n - i)
        fid.update(real[i:i + k].to(device).repeat(1, 3, 1, 1), real=True)   # grey → 3 channels
        fid.update(gen(k).clamp(0, 1).repeat(1, 3, 1, 1), real=False)
    return fid.compute().item()


@torch.no_grad()
def main():
    m = load_models()
    gens = generators(m)
    te = torch.load("latents/z_mlp_test.pt")
    x64 = te["x"][:64].to(device)

    results = {}
    for name, model in m.items():
        print(f"== {name} ==")
        log = torch.load(f"logs/{name}.pt")
        r = {
            "latent": LATENT_DIM[name],
            "epoch_time_s": sum(log["epoch_time"]) / len(log["epoch_time"]),
        }
        if name in gens:   # generative: time to create 64 new images, + FID
            r["latency_ms"] = latency_ms(lambda: gens[name](64), reps=3 if name == "ddpm" else 20)
            r["fid"] = fid_score(gens[name], te["x"])
        else:              # AE / Sparse AE: time to reconstruct 64 images
            r["latency_ms"] = latency_ms(lambda: model(x64))
        if name in ("ae", "sparse_ae", "vae"):
            _, r["mse"], r["psnr"], r["ssim"] = evaluate(model, te["x"], te["x"])
        results[name] = r
        print(r)

    with open("logs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    def fmt(v, spec):
        return "N/A" if v is None else format(v, spec)

    print("\n% LaTeX rows for report.tex (Bagian IV)")
    for name, r in results.items():
        print(f"{NAMES[name]} & {r['latent']} & {r['epoch_time_s']:.2f} & {r['latency_ms']:.2f} & "
              f"{fmt(r.get('mse'), '.4f')} & {fmt(r.get('psnr'), '.2f')} & "
              f"{fmt(r.get('ssim'), '.4f')} & {fmt(r.get('fid'), '.2f')} \\\\")


if __name__ == "__main__":
    main()
