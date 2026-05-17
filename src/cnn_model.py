"""CNN model for mitochondria segmentation and health classification."""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp


# --- Segmentation U-Net ---

def build_unet(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """
    Build a U-Net with ImageNet-pretrained encoder.
    Input: single-channel grayscale EM tiles (1, H, W)
    Output: binary segmentation mask (1, H, W)
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )
    return model


# --- Health Classification CNN ---

class MitoHealthDataset(Dataset):
    """Dataset of cropped mitochondrion patches with binary health labels."""

    def __init__(self, crops: list, labels: list, transform=None):
        self.crops = crops
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
        img = Image.fromarray((self.crops[idx] * 255).astype(np.uint8))
        x = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def build_resnet_classifier(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Fine-tune ResNet-50 for binary healthy/unhealthy classification."""
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


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train health classifier")
    parser.add_argument("--data", required=True, help="Labeled data directory")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load labeled data
    data_dir = Path(args.data)
    labels_csv = data_dir / "final_labels.csv"
    if not labels_csv.exists():
        print(f"ERROR: {labels_csv} not found. Run labeling pipeline first.")
        exit(1)

    # TODO: Load actual crops based on labels CSV
    # This is a skeleton — real implementation loads image crops
    print(f"Would train ResNet-50 for {args.epochs} epochs, lr={args.lr}")
    print(f"Output → {args.output}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
