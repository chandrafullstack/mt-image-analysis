"""Detect the scale bar in TEM TIFFs and compute the real pixel size.

The scale bar is a solid black horizontal line in the bottom-right corner,
with a label like "200 nm" beneath it. We crop the bottom-right region,
threshold it, and find the longest connected horizontal black run.

Usage:
    python scripts/detect_pixel_size.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path("data/raw/confocal")
# Folder name -> nominal scale bar length in nm
FOLDERS = {"200nm": 200, "400nm": 400, "800nm": 800}


def find_scale_bar_px(img: np.ndarray) -> int | None:
    """Return the length of the scale bar in pixels, or None if not found."""
    h, w = img.shape
    # Bottom-right quadrant where the scale bar lives
    roi = img[int(h * 0.85):, int(w * 0.6):]
    # Scale bar is solid black on light background
    black = roi < 50
    # Find the longest run of black pixels in any single row
    best = 0
    for row in black:
        runs = np.diff(np.where(np.concatenate(([0], row.view(np.int8), [0])) == 0)[0]) - 1
        if len(runs):
            best = max(best, int(runs.max()))
    return best if best > 20 else None


def main() -> None:
    print(f"{'folder':<8} {'file':<45} {'bar_px':>7}  {'nm/px':>8}")
    print("-" * 75)
    for folder, bar_nm in FOLDERS.items():
        d = ROOT / folder
        files = sorted(d.glob("*.tif"))[:3]
        for f in files:
            img = tifffile.imread(str(f))
            if img.ndim == 3:
                img = img[0]
            bar = find_scale_bar_px(img)
            if bar:
                px_nm = bar_nm / bar
                print(f"{folder:<8} {f.name:<45} {bar:>7}  {px_nm:>8.2f}")
            else:
                print(f"{folder:<8} {f.name:<45} {'?':>7}  {'?':>8}")


if __name__ == "__main__":
    main()
