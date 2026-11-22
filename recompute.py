import torch
from common import evaluate, device
from step01_train_classifier import MLPClassifier
from step04_frozen_decoder import FrozenMLPDecoder
from step06_autoencoders import SparseAE, active_fraction
from common import AE

te = torch.load("latents/z_mlp_test.pt")

# step01: test accuracy
mlp = MLPClassifier()
mlp.load_state_dict(torch.load("models/mlp_classifier.pt"))
mlp.to(device).eval()
with torch.no_grad():
    pred = mlp(te["x"].to(device)).argmax(1).cpu()
print(f"Test accuracy: {(pred == te['y']).float().mean():.2%}")

# step04: frozen MLP encoder + saved decoder
frozen = FrozenMLPDecoder(mlp)
frozen.decoder.load_state_dict(torch.load("models/decoder_frozen_mlp.pt"))
frozen.to(device)
_, mse, psnr, ssim = evaluate(frozen, te["x"], te["x"])
print(f"Z_MLP | MSE {mse:.4f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f}")

# step06: active latent units
for name, model in [("ae", AE()), ("sparse_ae", SparseAE())]:
    model.load_state_dict(torch.load(f"models/{name}.pt"))
    model.to(device)
    print(f"{name} | active h {active_fraction(model, te['x']):.1%}")