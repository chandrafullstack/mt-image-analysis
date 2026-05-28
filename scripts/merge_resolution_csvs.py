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

    # Preserve the per-resolution `label` as `local_label` so the dashboard
    # can resolve crop PNGs that live in per-resolution subfolders
    # (outputs/crops/<resolution_group>/mito_{local_label:04d}.png) for
    # resolutions whose inference wrote into a subdir (e.g. 800nm). The
    # top-level `label` is then renumbered globally so legacy crops that
    # were written directly into outputs/crops/ (200nm + 400nm rounds)
    # still resolve via mito_{label:04d}.png.
    for f in frames:
        f["local_label"] = f["label"]
    merged = pd.concat(frames, ignore_index=True)
    merged["label"] = range(1, len(merged) + 1)

    target = OUT / "features_with_gratio.csv"
    merged.to_csv(target, index=False)
    print(f"[ok] wrote {target} with {len(merged):,} total rows from {len(frames)} parts")


if __name__ == "__main__":
    main()
