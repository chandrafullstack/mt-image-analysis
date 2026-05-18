"""Step 1 of the 70% precision staircase: targeted hard-negative selection.

Pipeline:
  1. Score every crop in outputs/crops/round0_200nm/ with the v2 ResNet
     (outputs/models_v2_round1/resnet50_best.pt) -> P(UNHEALTHY).
  2. Exclude crops already labelled in R0 + R1 consensus CSVs.
  3. Select N targets from the model's *informative* zones:
        - "borderline":  0.30 <= P(U) < 0.70   (model is confused -> teach it)
        - "confident_U": 0.70 <= P(U)          (model says U; if Claude
                                                says H these become hard
                                                false positives we can fix)
  4. Copy those PNGs into outputs/crops/round2_hardneg/ and write a
     selection manifest.
  5. Print the exact Claude consensus command to label them.

Usage:
  python scripts/round2_hardneg_select.py \
      --crops-dir outputs/crops/round0_200nm \
      --model-dir outputs/models_v2_round1 \
      --exclude-csv outputs/predictions/round0_200nm_consensus.csv \
      --exclude-csv outputs/predictions/round1_200nm_consensus.csv \
      --out-dir outputs/crops/round2_hardneg \
      --select-csv outputs/predictions/round2_hardneg_selection.csv \
      --n-borderline 150 \
      --n-confident-u 100

Defaults are tuned for a ~$10 Claude budget (250 crops * 3 passes *
~$0.014/pass with claude-opus-4-6 = ~$10.5).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
def _torch():
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    return torch, DataLoader, Dataset, models, transforms


def load_model(model_path: Path, device):
    torch, _, _, models, _ = _torch()
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device).eval()
    return m


def score_all_crops(crops_dir: Path, model_path: Path, batch_size: int = 64) -> pd.DataFrame:
    """Return a DataFrame [crop_file, p_unhealthy] for every PNG in crops_dir."""
    torch, DataLoader, Dataset, _, transforms = _torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    crop_paths = sorted(crops_dir.glob("*.png"))
    if not crop_paths:
        raise SystemExit(f"No PNG crops in {crops_dir}")
    print(f"Found {len(crop_paths):,} crops in {crops_dir}")

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

    model = load_model(model_path, device)
    ds = _DS(crop_paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    probs: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring", unit="batch"):
            logits = model(batch.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())

    return pd.DataFrame({
        "crop_file": [p.name for p in crop_paths],
        "p_unhealthy": np.round(probs, 4),
    })


def load_excluded(exclude_csvs: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in exclude_csvs:
        if not path.exists():
            print(f"[warn] exclude-csv not found: {path}")
            continue
        df = pd.read_csv(path, usecols=["crop_file"])
        excluded.update(df["crop_file"].dropna().astype(str).tolist())
        print(f"  excluded {len(df):,} from {path.name}")
    return excluded


def select_targets(
    scored: pd.DataFrame,
    excluded: set[str],
    n_borderline: int,
    n_confident_u: int,
    border_lo: float,
    border_hi: float,
    conf_u_thr: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = scored[~scored["crop_file"].isin(excluded)].copy()
    print(f"After excluding labelled: {len(pool):,} candidates")

    border = pool[(pool["p_unhealthy"] >= border_lo) & (pool["p_unhealthy"] < border_hi)]
    conf_u = pool[pool["p_unhealthy"] >= conf_u_thr]

    print(f"  borderline pool  [{border_lo:.2f},{border_hi:.2f}): {len(border):,}")
    print(f"  confident-U pool [>={conf_u_thr:.2f}]:           {len(conf_u):,}")

    def _sample(df: pd.DataFrame, n: int, tag: str) -> pd.DataFrame:
        if len(df) == 0:
            return df.assign(tier=tag)
        take = min(n, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        return df.iloc[idx].assign(tier=tag)

    chosen = pd.concat([
        _sample(border, n_borderline, "borderline"),
        _sample(conf_u, n_confident_u, "confident_u"),
    ], ignore_index=True)

    # Drop dupes if a row happened to fall in both bands.
    chosen = chosen.drop_duplicates("crop_file").reset_index(drop=True)
    print(f"Selected {len(chosen):,} targets")
    print(chosen["tier"].value_counts().to_string())
    return chosen


def copy_into_workdir(selection: pd.DataFrame, src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in tqdm(selection["crop_file"], desc="copying", unit="file"):
        src = src_dir / name
        dst = dst_dir / name
        if not dst.exists():
            shutil.copyfile(src, dst)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", type=Path,
                    default=Path("outputs/crops/round0_200nm"))
    ap.add_argument("--model-dir", type=Path,
                    default=Path("outputs/models_v2_round1"))
    ap.add_argument("--exclude-csv", type=Path, action="append", required=True,
                    help="Repeatable. Prior consensus CSVs (crop_file col).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/crops/round2_hardneg"))
    ap.add_argument("--select-csv", type=Path,
                    default=Path("outputs/predictions/round2_hardneg_selection.csv"))
    ap.add_argument("--scored-csv", type=Path,
                    default=Path("outputs/predictions/all_crops_scored_v2.csv"),
                    help="Cached per-crop P(U) so we don't rescore every run.")
    ap.add_argument("--rescore", action="store_true",
                    help="Force rescoring even if scored-csv exists.")
    ap.add_argument("--n-borderline", type=int, default=150)
    ap.add_argument("--n-confident-u", type=int, default=100)
    ap.add_argument("--border-lo", type=float, default=0.30)
    ap.add_argument("--border-hi", type=float, default=0.70)
    ap.add_argument("--conf-u-thr", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- step 1: score (or load cached) --------------------------------- #
    if args.scored_csv.exists() and not args.rescore:
        print(f"Loading cached scores from {args.scored_csv}")
        scored = pd.read_csv(args.scored_csv)
    else:
        scored = score_all_crops(args.crops_dir, args.model_dir / "resnet50_best.pt")
        args.scored_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.scored_csv, index=False)
        print(f"wrote {args.scored_csv}")

    # Quick distribution print
    print("\nP(U) distribution over all crops:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    print(scored["p_unhealthy"].value_counts(bins=bins, sort=False).to_string())

    # --- step 2: exclude already-labelled ------------------------------- #
    excluded = load_excluded(args.exclude_csv)

    # --- step 3: select ------------------------------------------------- #
    selection = select_targets(
        scored, excluded,
        args.n_borderline, args.n_confident_u,
        args.border_lo, args.border_hi, args.conf_u_thr,
        args.seed,
    )

    args.select_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.select_csv, index=False)
    print(f"wrote {args.select_csv}")

    # --- step 4: copy --------------------------------------------------- #
    copy_into_workdir(selection, args.crops_dir, args.out_dir)
    print(f"copied {len(selection):,} crops into {args.out_dir}")

    # --- step 5: print Claude command ----------------------------------- #
    print("\n" + "=" * 70)
    print("Next: run Claude 3-pass consensus on these hard negatives")
    print("=" * 70)
    print("  python -m src.claude_score_crops \\")
    print(f"      --crops-dir {args.out_dir.as_posix()} \\")
    print(f"      --output-csv outputs/predictions/round2_hardneg_consensus.csv \\")
    print(f"      --sample {len(selection)} \\")
    print(f"      --model claude-opus-4-6 \\")
    print(f"      --budget-usd 12 \\")
    print(f"      --passes 3")
    print("\nEstimated cost: ~$%.2f  (3 passes x %d crops x ~$0.014/pass)" %
          (len(selection) * 3 * 0.014, len(selection)))


if __name__ == "__main__":
    main()
