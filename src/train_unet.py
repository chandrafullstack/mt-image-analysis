"""Train U-Net segmentation model on paired EM images and annotated masks."""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

try:
    from src.cnn_model import build_unet
except ModuleNotFoundError:
    from cnn_model import build_unet

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(str(path))
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32)
    else:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)

    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def _resize(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    out = pil.resize((size, size), resample=resample)
    out_arr = np.array(out, dtype=np.float32) / 255.0
    if is_mask:
        out_arr = (out_arr > 0.5).astype(np.float32)
    return out_arr


def build_pairs(images_dir: Path, masks_dir: Path, mask_suffix: str = "_mask") -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        stem = img_path.stem
        candidates = []
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            candidates.append(masks_dir / f"{stem}{mask_suffix}{ext}")
            candidates.append(masks_dir / f"{stem}{ext}")

        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


class EMMaskDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int = 512, augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = _load_gray(img_path)
        mask = _load_gray(mask_path)
        mask = (mask > 0.5).astype(np.float32)

        img = _resize(img, self.image_size, is_mask=False)
        mask = _resize(mask, self.image_size, is_mask=True)

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        x = torch.from_numpy(img).unsqueeze(0).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        return x, y


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    total = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (total + eps)
    return float(iou.mean().item())


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_dice += dice_score(pred.detach(), y)
        total_iou += iou_score(pred.detach(), y)
        batches += 1

    return {
        "loss": total_loss / max(1, batches),
        "dice": total_dice / max(1, batches),
        "iou": total_iou / max(1, batches),
    }


def train_unet(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    mask_suffix: str = "_mask",
    image_size: int = 512,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_ratio: float = 0.2,
):
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is required for U-Net training. Install dependencies from requirements.txt")

    pairs = build_pairs(images_dir, masks_dir, mask_suffix=mask_suffix)
    if len(pairs) < 8:
        raise ValueError("Need at least 8 paired image/mask samples for training.")

    train_pairs, val_pairs = train_test_split(pairs, test_size=val_ratio, random_state=42)

    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for p in train_pairs:
        split_rows.append({"split": "train", "image": str(p[0]), "mask": str(p[1])})
    for p in val_pairs:
        split_rows.append({"split": "val", "image": str(p[0]), "mask": str(p[1])})
    pd.DataFrame(split_rows).to_csv(output_dir / "unet_split.csv", index=False)

    train_ds = EMMaskDataset(train_pairs, image_size=image_size, augment=True)
    val_ds = EMMaskDataset(val_pairs, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(pretrained=True).to(device)

    # build_unet returns sigmoid output, so BCE is appropriate.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_dice": tr["dice"],
            "train_iou": tr["iou"],
            "val_loss": va["loss"],
            "val_dice": va["dice"],
            "val_iou": va["iou"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={tr['loss']:.4f} train_dice={tr['dice']:.3f} | "
            f"val_loss={va['loss']:.4f} val_dice={va['dice']:.3f}"
        )

        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save(model.state_dict(), output_dir / "unet_best.pt")

    torch.save(model.state_dict(), output_dir / "unet_last.pt")
    pd.DataFrame(history).to_csv(output_dir / "unet_training_history.csv", index=False)

    print(f"Saved best model -> {output_dir / 'unet_best.pt'}")
    print(f"Saved last model -> {output_dir / 'unet_last.pt'}")
    print(f"Saved history -> {output_dir / 'unet_training_history.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on annotated EM masks")
    parser.add_argument("--images-dir", required=True, help="Directory containing input EM images")
    parser.add_argument("--masks-dir", required=True, help="Directory containing binary masks")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--mask-suffix", default="_mask", help="Mask filename suffix (default: _mask)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    train_unet(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        output_dir=Path(args.output),
        mask_suffix=args.mask_suffix,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
    )
