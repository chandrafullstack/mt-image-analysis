"""G-ratio calculation for mitochondria instances."""
import numpy as np
import pandas as pd
from skimage import measure, morphology
from pathlib import Path
import tifffile
from tqdm import tqdm


def estimate_gratio(instance_mask: np.ndarray,
                    label_id: int,
                    pixel_size_um: float = 0.008) -> float:
    """
    Estimate the G-ratio for a single mitochondrion instance.

    Strategy:
      1. Extract binary mask for this instance
      2. Erode to approximate inner membrane boundary
      3. Compute equivalent diameters of outer (OMM) and inner (IMM) regions
      4. G-ratio = d_inner / d_outer
    """
    binary = (instance_mask == label_id).astype(np.uint8)
    props_outer = measure.regionprops(binary)
    if not props_outer:
        return float("nan")

    area_outer = props_outer[0].area
    d_outer = 2 * np.sqrt(area_outer / np.pi) * pixel_size_um

    # Approximate IMM as the eroded region (membrane thickness ~5–10 nm = 1 pixel at 8nm/px)
    membrane_px = max(1, int(round(0.01 / pixel_size_um)))  # 10 nm membrane
    inner = morphology.erosion(binary, morphology.disk(membrane_px))
    props_inner = measure.regionprops(inner.astype(np.uint8))
    if not props_inner:
        return float("nan")

    area_inner = props_inner[0].area
    d_inner = 2 * np.sqrt(area_inner / np.pi) * pixel_size_um

    return d_inner / (d_outer + 1e-8)


def compute_gratio_for_all(instance_mask: np.ndarray,
                           pixel_size_um: float = 0.008) -> dict:
    """Returns {label_id: g_ratio} for all instances."""
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]
    return {
        int(lbl): estimate_gratio(instance_mask, lbl, pixel_size_um)
        for lbl in labels
    }


def add_gratio_to_features(features_csv: Path, masks_dir: Path,
                           output_csv: Path, pixel_size_um: float = 0.008):
    """Add G-ratio column to an existing features CSV."""
    df = pd.read_csv(features_csv)
    gratios = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing G-ratios"):
        source = row["source_file"]
        mask_path = masks_dir / f"{source}.tif"
        if mask_path.exists():
            mask = tifffile.imread(str(mask_path)).astype(np.int32)
            g = estimate_gratio(mask, int(row["label"]), pixel_size_um)
        else:
            g = float("nan")
        gratios.append(g)

    df["g_ratio"] = gratios
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Added G-ratio to {len(df)} instances → {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute G-ratios")
    parser.add_argument("--features", required=True, help="Input features CSV")
    parser.add_argument("--masks", required=True, help="Masks directory")
    parser.add_argument("--output", required=True, help="Output CSV with G-ratio")
    parser.add_argument("--pixel-size", type=float, default=0.008)
    args = parser.parse_args()

    add_gratio_to_features(
        Path(args.features), Path(args.masks), Path(args.output), args.pixel_size
    )
