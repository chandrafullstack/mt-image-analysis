"""Morphological feature extraction for segmented mitochondria."""
import argparse
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
from pathlib import Path
from tqdm import tqdm


def extract_instance_features(instance_mask: np.ndarray,
                              pixel_size_um: float = 0.008) -> pd.DataFrame:
    """
    Extract morphological features for each labelled mitochondrion instance.

    Args:
        instance_mask: int32 label map, 0=background, >0 = instance ID
        pixel_size_um: physical size of one pixel in micrometres (default 8 nm = 0.008 µm)

    Returns:
        DataFrame with one row per mitochondrion instance
    """
    props = measure.regionprops_table(
        instance_mask,
        properties=[
            "label", "area", "perimeter", "major_axis_length",
            "minor_axis_length", "eccentricity", "solidity",
            "orientation", "centroid",
        ],
    )
    df = pd.DataFrame(props)
    if df.empty:
        return df

    s = pixel_size_um
    df["area_um2"] = df["area"] * (s ** 2)
    df["perimeter_um"] = df["perimeter"] * s
    df["major_axis_um"] = df["major_axis_length"] * s
    df["minor_axis_um"] = df["minor_axis_length"] * s
    df["aspect_ratio"] = df["major_axis_um"] / (df["minor_axis_um"] + 1e-8)
    df["form_factor"] = (4 * np.pi * df["area_um2"]) / (df["perimeter_um"] ** 2 + 1e-8)
    df["roundness"] = (4 * df["area_um2"]) / (np.pi * df["major_axis_um"] ** 2 + 1e-8)

    return df


def extract_features_from_directory(masks_dir: Path, output_path: Path,
                                    pixel_size_um: float = 0.008):
    """Extract features from all mask files in a directory."""
    masks_dir = Path(masks_dir)
    all_features = []

    mask_files = sorted(masks_dir.glob("*_mask.tif"))
    for mask_path in tqdm(mask_files, desc="Extracting features"):
        mask = tifffile.imread(str(mask_path)).astype(np.int32)
        df = extract_instance_features(mask, pixel_size_um)
        if not df.empty:
            df["source_file"] = mask_path.stem
            all_features.append(df)

    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"Extracted features for {len(combined)} instances → {output_path}")
    else:
        print("No instances found in any mask file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract morphological features")
    parser.add_argument("--masks", required=True, help="Directory of mask .tif files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--pixel-size", type=float, default=0.008)
    args = parser.parse_args()

    extract_features_from_directory(Path(args.masks), Path(args.output), args.pixel_size)
