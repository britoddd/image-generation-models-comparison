"""
Tahap 01: Pelatihan MLP Classifier berjenjang

Model yang dibuat memiliki dimensi 28 * 28 (784) -> 256 -> 64 -> 10
Loss function yang digunakan: Cross-Entropy loss

"""

import torch
from datasplit import get_data_loaders
from common import device

EPOCHS = 10


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(28 * 28, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 10)
        self.activation = torch.nn.ReLU()

    def encode(self, x):
        """Return z representation (64 dimensions), the features before the final classifier."""

        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        z = self.activation(x)
        return z

    def forward(self, x):
        z = self.linear3(self.encode(x))
        return z


@torch.no_grad()
def extract_z(model, loader):
    model.eval()
    zs, ys, xs = [], [], []
    for images, labels in loader:
        zs.append(model.encode(images.to(device)).cpu())
        ys.append(labels)
        xs.append(images)
    return torch.cat(zs), torch.cat(ys), torch.cat(xs)


def main():
    model = MLPClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    train_loader, val_loader, test_loader = get_data_loaders()

    losses = []
    model.to(device)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Calculate model prediction loss
            logits = model(images)
            loss = loss_fn(logits, labels)

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            losses.append(loss.item())
        epoch_loss = sum(losses[-len(train_loader) :]) / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/mlp_classifier.pt")

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.6f} | Val acc: {val_acc:.2f}%")

    model.load_state_dict(torch.load("models/mlp_classifier.pt"))

    # Extract Parameters
    weights = {
        "W1": model.linear1.weight.detach().cpu(),
        "W2": model.linear2.weight.detach().cpu(),
        "W3": model.linear3.weight.detach().cpu(),
    }

    torch.save(weights, "weights/weights_W1_W2_W3.pt")

    model.eval()
    correct, total_pred = 0, 0

    with torch.no_grad():
        # Accuracy calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_pred += labels.size(0)

        print(f"Accuracy: {correct/total_pred * 100}%")

    # z value extraction
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        z, y, x = extract_z(model, loader)
        torch.save({"z": z, "y": y, "x": x}, f"latents/z_mlp_{name}.pt")


if __name__ == "__main__":
    main()
