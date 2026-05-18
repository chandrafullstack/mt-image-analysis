"""Concatenate per-resolution inference CSVs into the dashboard CSV.

Reads:
  outputs/metrics/features_200nm.csv  (pixel_size_um=0.2)
  outputs/metrics/features_400nm.csv  (pixel_size_um=0.4)
  outputs/metrics/features_800nm.csv  (pixel_size_um=0.8)

Writes:
  outputs/metrics/features_with_gratio.csv  (concatenated, instance ids renumbered)

Missing files are skipped silently so this is safe to run mid-pipeline.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

OUT = Path("outputs/metrics")
PARTS = [
    ("features_200nm.csv", 0.2),
    ("features_400nm.csv", 0.4),
    ("features_800nm.csv", 0.8),
]


def main() -> None:
    frames = []
    for fname, px in PARTS:
        p = OUT / fname
        if not p.exists():
            print(f"[skip] {fname} not present yet")
            continue
        df = pd.read_csv(p)
        df["pixel_size_um"] = px
        df["resolution_group"] = fname.replace("features_", "").replace(".csv", "")
        print(f"[load] {fname}: {len(df):,} rows")
        frames.append(df)

    if not frames:
        print("Nothing to merge.")
        return

    merged = pd.concat(frames, ignore_index=True)
    # Renumber instance ids so dashboard hover/crops still resolve uniquely.
    merged["label"] = range(1, len(merged) + 1)

    target = OUT / "features_with_gratio.csv"
    merged.to_csv(target, index=False)
    print(f"[ok] wrote {target} with {len(merged):,} total rows from {len(frames)} parts")


if __name__ == "__main__":
    main()
