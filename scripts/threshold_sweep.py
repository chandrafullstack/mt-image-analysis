"""
Threshold sweep on val + test for the v2 ResNet model.

Computes P(UNHEALTHY) for every val and test crop, then sweeps the
decision threshold from 0.05 to 0.95 in 0.05 increments and reports
precision / recall / F1 + confusion matrix for the UNHEALTHY class.
Writes a CSV per split plus a combined PNG plot.

Usage:
  python -m scripts.threshold_sweep \\
      --model-dir outputs/models_v2_round1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _torch_imports():
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    return torch, DataLoader, Dataset, models, transforms


def load_model(model_dir: Path, device):
    torch, _, _, models, _ = _torch_imports()
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_dir / "resnet50_best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_proba(model, manifest: pd.DataFrame, device, batch_size: int = 32) -> np.ndarray:
    """Return P(UNHEALTHY) per row in the manifest order."""
    torch, DataLoader, Dataset, _, transforms = _torch_imports()
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    class _DS(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return tfm(Image.open(self.paths[i]).convert("L"))

    ds = _DS(manifest["crop_path"].tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # index 1 = UNHEALTHY
            probs.append(p)
    return np.concatenate(probs)


def sweep(y_true: np.ndarray, p_unhealthy: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (p_unhealthy >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0  # = sensitivity = TPR
        specificity = tn / (tn + fp) if (tn + fp) else 0.0  # = TNR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        bal_acc = 0.5 * (recall + specificity)
        # MCC -- imbalance-safe single number
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
        rows.append({
            "threshold": round(float(t), 3),
            "precision_U": round(precision, 3),
            "recall_U": round(recall, 3),       # sensitivity
            "specificity_H": round(specificity, 3),
            "f1_U": round(f1, 3),
            "accuracy": round(acc, 3),
            "balanced_acc": round(bal_acc, 3),
            "mcc": round(mcc, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--start", type=float, default=0.05)
    ap.add_argument("--stop", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(args.model_dir, device)
    thresholds = np.arange(args.start, args.stop + 1e-9, args.step)

    out_rows = []
    for split in ("val", "test"):
        manifest = pd.read_csv(args.model_dir / f"manifest_{split}.csv")
        manifest["crop_path"] = manifest["crop_path"].apply(
            lambda p: str(Path(p)) if Path(p).is_absolute() else
            str(Path("c:/Chandra/MT/mito_classifier") / p)
        )
        y_true = manifest["y"].astype(int).values  # 1 = UNHEALTHY, 0 = HEALTHY
        probs = predict_proba(model, manifest, device)
        df = sweep(y_true, probs, thresholds)
        df["split"] = split
        out_path = args.model_dir / f"threshold_sweep_{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n=== {split} (n={len(y_true)}, UNHEALTHY={int(y_true.sum())}) ===")
        print(df.to_string(index=False))
        print(f"wrote {out_path}")
        out_rows.append(df)

        # Save raw probabilities for later analysis
        manifest_out = manifest.copy()
        manifest_out["p_unhealthy"] = probs
        manifest_out.to_csv(args.model_dir / f"probs_{split}.csv", index=False)

    combined = pd.concat(out_rows, ignore_index=True)
    combined.to_csv(args.model_dir / "threshold_sweep_all.csv", index=False)

    # Best operating points by multiple criteria
    print("\n=== Best operating points (multiple criteria) ===")
    for split in ("val", "test"):
        d = combined[combined["split"] == split].reset_index(drop=True)
        for crit in ("f1_U", "balanced_acc", "mcc"):
            best = d.loc[d[crit].idxmax()]
            print(f"  {split:<5} by {crit:<12} thr={best['threshold']:.3f}  "
                  f"P={best['precision_U']:.2f}  R={best['recall_U']:.2f}  "
                  f"Spec={best['specificity_H']:.2f}  F1={best['f1_U']:.2f}  "
                  f"bal_acc={best['balanced_acc']:.2f}  mcc={best['mcc']:.2f}")


if __name__ == "__main__":
    main()
