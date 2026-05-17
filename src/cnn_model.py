"""CNN training/inference utilities for mitochondria segmentation and health classification."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def _require_torch_classifier_deps():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except ImportError as exc:
        raise ImportError(
            "PyTorch + torchvision are required for classifier training/inference."
        ) from exc
    return torch, nn, DataLoader, models, transforms


def _require_unet_deps():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch is required for U-Net. Install requirements.txt"
        ) from exc
    return smp


# --- Segmentation U-Net ---

def build_unet(encoder: str = "resnet34", pretrained: bool = True):
    """
    Build a U-Net with ImageNet-pretrained encoder.
    Input: single-channel grayscale EM tiles (1, H, W)
    Output: binary segmentation mask (1, H, W)
    """
    smp = _require_unet_deps()
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )
    return model


# --- Health Classification CNN ---

class MitoHealthDataset:
    """Dataset of mitochondria crops with binary health labels."""

    def __init__(self, crops: list[Path], labels: list[int], transform=None):
        torch, _, _, _, transforms = _require_torch_classifier_deps()
        self._torch = torch
        self.crops = [Path(p) for p in crops]
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = Image.open(self.crops[idx]).convert("L")
        x = self.transform(img)
        y = self._torch.tensor(self.labels[idx], dtype=self._torch.long)
        return x, y


def build_resnet_classifier(num_classes: int = 2, pretrained: bool = True):
    """Fine-tune ResNet-50 for binary healthy/unhealthy classification."""
    _, nn, _, models, _ = _require_torch_classifier_deps()
    model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def build_labeled_samples(data_dir: Path) -> tuple[list[Path], list[int]]:
    """
    Build sample list from a folder tree:
      data_dir/
        HEALTHY/*.png
        UNHEALTHY/*.png
    Also accepts lowercase folder names.
    """
    healthy_dirs = [data_dir / "HEALTHY", data_dir / "healthy"]
    unhealthy_dirs = [data_dir / "UNHEALTHY", data_dir / "unhealthy"]
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    crops: list[Path] = []
    labels: list[int] = []

    for hdir in healthy_dirs:
        if hdir.exists():
            for p in sorted(hdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    crops.append(p)
                    labels.append(0)

    for udir in unhealthy_dirs:
        if udir.exists():
            for p in sorted(udir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    crops.append(p)
                    labels.append(1)

    return crops, labels


def save_split_manifest(output_dir: Path, train_paths, train_labels, val_paths, val_labels):
    rows = []
    for p, y in zip(train_paths, train_labels):
        rows.append({"split": "train", "image_path": str(p), "label": int(y)})
    for p, y in zip(val_paths, val_labels):
        rows.append({"split": "val", "image_path": str(p), "label": int(y)})
    pd.DataFrame(rows).to_csv(output_dir / "dataset_split.csv", index=False)


def train_classifier(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 25,
    lr: float = 1e-4,
    batch_size: int = 16,
    val_ratio: float = 0.2,
):
    torch, nn, DataLoader, _, transforms = _require_torch_classifier_deps()

    crops, labels = build_labeled_samples(data_dir)
    if len(crops) < 10:
        raise ValueError("Need at least 10 labeled crop images for training.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        crops,
        labels,
        test_size=val_ratio,
        random_state=42,
        stratify=labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(output_dir, train_paths, train_labels, val_paths, val_labels)

    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    train_ds = MitoHealthDataset(train_paths, train_labels, transform=train_tf)
    val_ds = MitoHealthDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet_classifier(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc = -1.0
    best_path = output_dir / "resnet50_best.pt"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "resnet50_last.pt")
    print(f"Saved best model -> {best_path}")
    print(f"Saved history -> {output_dir / 'training_history.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train health classifier")
    parser.add_argument("--data", required=True, help="Labeled data directory with HEALTHY/UNHEALTHY subfolders")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    train_classifier(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )
