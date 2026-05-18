"""
Rigorous ResNet-50 trainer for pseudo-labelled mitochondria crops.

Key differences from src/cnn_model.py train_classifier:
  1) Loads labels from one or more Claude consensus CSVs (not folders),
     joining each crop to its source TIF via the metrics CSV.
  2) Splits crops **by source image** (GroupShuffleSplit) so the same
     parent TIF never appears in both train and test. This is the real
     test of generalisation -- random crop-level splits leak shared
     acquisition noise / staining intensity.
  3) Holds out a real test set (~20% of source images) that the model
     never sees during training or validation.
  4) Class-weighted CrossEntropyLoss + optional WeightedRandomSampler
     to counter the 4-5:1 HEALTHY:UNHEALTHY imbalance.
  5) Reports per-class precision/recall/F1 and confusion matrix on
     both val and test sets, plus train_acc, so overfitting is visible.

Example:
  python -m src.train_pseudo_labels \\
      --consensus-csv outputs/predictions/round0_200nm_consensus.csv \\
      --metrics-csv  outputs/metrics/round0_200nm.csv \\
      --crops-dir    outputs/crops/round0_200nm \\
      --output-dir   outputs/models_v2 \\
      --epochs 25 --batch-size 16 --min-agreement 0.67 --min-calibrated 0.55
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


CROP_ID_RE = re.compile(r"mito_(\d+)\.png", re.IGNORECASE)


def _require_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import models, transforms
    return torch, nn, DataLoader, Dataset, WeightedRandomSampler, models, transforms


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def assemble_dataset(
    consensus_csvs: list[Path],
    metrics_csvs: list[Path],
    crops_dir: Path,
    min_agreement: float,
    min_calibrated: float,
) -> pd.DataFrame:
    """Join consensus + metrics, filter to confident HEALTHY/UNHEALTHY, return DataFrame."""
    # Load metrics: label (instance id) -> source_file
    metric_frames = []
    for mc in metrics_csvs:
        df = pd.read_csv(mc)
        if "label" not in df.columns or "source_file" not in df.columns:
            raise ValueError(f"{mc} missing required columns label/source_file")
        metric_frames.append(df[["label", "source_file"]])
    metrics = pd.concat(metric_frames, ignore_index=True).drop_duplicates("label")

    # Load consensus CSVs
    cons_frames = []
    for cc in consensus_csvs:
        df = pd.read_csv(cc)
        cons_frames.append(df)
    cons = pd.concat(cons_frames, ignore_index=True).drop_duplicates("crop_file", keep="last")

    # Derive label column (handle single-pass + consensus schemas)
    if "final_classification" in cons.columns:
        cons["label_str"] = cons["final_classification"]
    elif "classification" in cons.columns:
        cons["label_str"] = cons["classification"]
    else:
        raise ValueError("Consensus CSV missing final_classification / classification column")

    if "label_agreement" not in cons.columns:
        cons["label_agreement"] = 1.0
    if "calibrated_confidence" not in cons.columns:
        cons["calibrated_confidence"] = cons.get("confidence", 1.0)

    # Filter to high-confidence HEALTHY/UNHEALTHY
    mask = (
        cons["label_str"].isin(["HEALTHY", "UNHEALTHY"])
        & (cons["label_agreement"] >= min_agreement)
        & (cons["calibrated_confidence"] >= min_calibrated)
    )
    cons = cons.loc[mask, ["crop_file", "label_str", "label_agreement", "calibrated_confidence"]].copy()
    if cons.empty:
        raise SystemExit("No crops pass the agreement/confidence thresholds.")

    # Extract instance id from crop filename and join to metrics
    cons["instance_id"] = cons["crop_file"].str.extract(CROP_ID_RE)[0]
    n_matched = cons["instance_id"].notna().sum()
    if n_matched < 0.9 * len(cons):
        raise SystemExit(
            f"CROP_ID_RE matched only {n_matched}/{len(cons)} crop filenames. "
            f"Check that crop files are named like 'mito_<int>.png' or update "
            f"CROP_ID_RE in train_pseudo_labels.py."
        )
    cons = cons.dropna(subset=["instance_id"]).copy()
    cons["instance_id"] = cons["instance_id"].astype(int)
    joined = cons.merge(metrics, left_on="instance_id", right_on="label", how="left")

    missing_src = joined["source_file"].isna().sum()
    if missing_src:
        print(f"[warn] {missing_src} crops could not be matched to a source_file; dropping them.")
        joined = joined.dropna(subset=["source_file"])

    # Confirm crop files exist on disk
    joined["crop_path"] = joined["crop_file"].apply(lambda name: crops_dir / name)
    exists_mask = joined["crop_path"].apply(lambda p: p.exists())
    if (~exists_mask).any():
        missing = (~exists_mask).sum()
        print(f"[warn] {missing} crops listed in CSV not found in {crops_dir}; dropping them.")
        joined = joined[exists_mask]

    joined["y"] = (joined["label_str"] == "UNHEALTHY").astype(int)  # 0=HEALTHY, 1=UNHEALTHY
    return joined.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int):
    """
    Group-by-source-file holdout for test, then group-aware split of the
    remainder into train/val.  When too few groups exist for a group-aware
    val split (common in pilot runs), falls back to stratified crop split.
    """
    groups = df["source_file"].values
    y = df["y"].values
    n_groups = df["source_file"].nunique()
    print(f"  groups (source images): {n_groups}")

    # ---- Test holdout
    if n_groups >= 3 and test_frac > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(gss.split(df, y, groups))
    else:
        print("  [warn] too few groups for grouped test holdout; using stratified.")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(sss.split(df, y))
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ---- Train / val on the remainder
    groups_tv = df_trainval["source_file"].values
    y_tv = df_trainval["y"].values
    n_groups_tv = df_trainval["source_file"].nunique()
    val_size = max(val_frac, 1.0 / max(n_groups_tv, 2))
    try:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
        tr_idx, va_idx = next(gss2.split(df_trainval, y_tv, groups_tv))
        # Ensure val has both classes; if not, fall back.
        if len(set(y_tv[va_idx])) < 2 and len(set(y_tv)) > 1:
            raise ValueError("group split produced single-class val set")
        df_train = df_trainval.iloc[tr_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[va_idx].reset_index(drop=True)
        val_kind = "grouped"
    except Exception as exc:
        print(f"  [warn] grouped train/val split failed ({exc}); using stratified crop split.")
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
        tr_idx, va_idx = next(sss2.split(df_trainval, y_tv))
        df_train = df_trainval.iloc[tr_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[va_idx].reset_index(drop=True)
        val_kind = "stratified-crop"

    print(f"  split sizes -> train={len(df_train)}  val={len(df_val)} ({val_kind})  test={len(df_test)}")
    print(f"  train sources: {sorted(df_train['source_file'].unique())}")
    print(f"  val   sources: {sorted(df_val['source_file'].unique())}")
    print(f"  test  sources: {sorted(df_test['source_file'].unique())}")
    return df_train, df_val, df_test


# --------------------------------------------------------------------------- #
# Torch dataset / training
# --------------------------------------------------------------------------- #
def _make_dataset_class(Dataset, Image):
    class _MitoCropDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = list(paths)
            self.labels = list(labels)
            self.tf = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("L")
            x = self.tf(img)
            return x, int(self.labels[idx])

    return _MitoCropDataset


def _eval_split(model, loader, device, torch):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(preds)


def _print_metrics(name, y_true, y_pred):
    if len(y_true) == 0:
        print(f"  [{name}] empty split")
        return {}
    acc = float((y_true == y_pred).mean())
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"  [{name}] acc={acc:.3f}  n={len(y_true)}")
    print(f"    HEALTHY   (n={support[0]}): P={prec[0]:.3f} R={rec[0]:.3f} F1={f1[0]:.3f}")
    print(f"    UNHEALTHY (n={support[1]}): P={prec[1]:.3f} R={rec[1]:.3f} F1={f1[1]:.3f}")
    print(f"    macro F1 = {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print(f"    confusion matrix (rows=true [H,U], cols=pred [H,U]):\n      {cm.tolist()}")
    return {
        "acc": acc,
        "n": int(len(y_true)),
        "p_healthy": float(prec[0]), "r_healthy": float(rec[0]), "f1_healthy": float(f1[0]),
        "p_unhealthy": float(prec[1]), "r_unhealthy": float(rec[1]), "f1_unhealthy": float(f1[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": cm.tolist(),
    }


def train(args):
    torch, nn, DataLoader, Dataset, WeightedRandomSampler, tv_models, transforms = _require_torch()

    print("=== assembling dataset ===")
    df = assemble_dataset(
        consensus_csvs=args.consensus_csv,
        metrics_csvs=args.metrics_csv,
        crops_dir=args.crops_dir,
        min_agreement=args.min_agreement,
        min_calibrated=args.min_calibrated,
    )
    print(f"  usable labelled crops: {len(df)}")
    print(f"  class counts: {df['label_str'].value_counts().to_dict()}")

    print("=== splitting ===")
    df_train, df_val, df_test = make_splits(
        df, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.assign(split="all").to_csv(output_dir / "manifest_all.csv", index=False)
    df_train.assign(split="train").to_csv(output_dir / "manifest_train.csv", index=False)
    df_val.assign(split="val").to_csv(output_dir / "manifest_val.csv", index=False)
    df_test.assign(split="test").to_csv(output_dir / "manifest_test.csv", index=False)

    # ---- transforms
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    DS = _make_dataset_class(Dataset, Image)
    train_ds = DS(df_train["crop_path"].tolist(), df_train["y"].tolist(), train_tf)
    val_ds   = DS(df_val["crop_path"].tolist(),   df_val["y"].tolist(),   eval_tf)
    test_ds  = DS(df_test["crop_path"].tolist(),  df_test["y"].tolist(),  eval_tf)

    # ---- class weights
    counts = np.bincount(df_train["y"].values, minlength=2).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    class_weights_t = torch.tensor(inv / inv.sum() * 2.0, dtype=torch.float32)
    print(f"  train class counts: HEALTHY={int(counts[0])}, UNHEALTHY={int(counts[1])}")
    print(f"  class weights: {class_weights_t.tolist()}")

    # ---- sampler: each crop gets weight 1/count(class)
    sample_w = np.array([inv[y] for y in df_train["y"].values], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=sample_w.tolist(), num_samples=len(sample_w), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tv_models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("=== training ===")
    history = []
    best_val_macro_f1 = -1.0
    best_path = output_dir / "resnet50_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_correct, tr_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_n += x.size(0)
        tr_loss = tr_loss_sum / max(tr_n, 1)
        tr_acc  = tr_correct / max(tr_n, 1)

        y_val_true, y_val_pred = _eval_split(model, val_loader, device, torch)
        val_acc = float((y_val_true == y_val_pred).mean()) if len(y_val_true) else float("nan")
        val_f1m = float(
            f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)
        ) if len(y_val_true) else float("nan")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1m,
        })
        print(
            f"  epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
            f"val_acc={val_acc:.3f} val_macroF1={val_f1m:.3f}"
        )

        if val_f1m > best_val_macro_f1:
            best_val_macro_f1 = val_f1m
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), output_dir / "resnet50_last.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"  best val macro F1 = {best_val_macro_f1:.3f} -> {best_path}")

    # ---- final eval with best weights
    print("\n=== final evaluation (best-by-val-macro-F1 weights) ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_t, p_t = _eval_split(model, train_loader, device, torch)
    train_report = _print_metrics("train", y_t, p_t)
    y_v, p_v = _eval_split(model, val_loader, device, torch)
    val_report   = _print_metrics("val",   y_v, p_v)
    y_te, p_te = _eval_split(model, test_loader, device, torch)
    test_report  = _print_metrics("test",  y_te, p_te)

    summary = {
        "n_total": int(len(df)),
        "class_counts": df["label_str"].value_counts().to_dict(),
        "split_sizes": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "train_metrics": train_report,
        "val_metrics": val_report,
        "test_metrics": test_report,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {output_dir / 'evaluation_summary.json'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--consensus-csv", type=Path, action="append", required=True,
                   help="Repeatable: one or more Claude consensus CSVs.")
    p.add_argument("--metrics-csv",   type=Path, action="append", required=True,
                   help="Repeatable: one or more segmentation metrics CSVs (must contain label, source_file).")
    p.add_argument("--crops-dir",     type=Path, required=True,
                   help="Directory containing the crop PNGs.")
    p.add_argument("--output-dir",    type=Path, required=True)
    p.add_argument("--min-agreement",   type=float, default=0.67)
    p.add_argument("--min-calibrated",  type=float, default=0.55)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--val-frac",  type=float, default=0.20)
    p.add_argument("--epochs",       type=int, default=25)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
