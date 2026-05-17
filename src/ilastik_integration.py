"""Ilastik integration: export crops and import predictions."""
import argparse
import subprocess
import h5py
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm


def export_crops_for_ilastik(instance_mask: np.ndarray,
                             raw_image: np.ndarray,
                             output_dir: Path):
    """Export individual mitochondrion crops as .h5 files for Ilastik."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]

    for lbl in labels:
        mask = (instance_mask == lbl)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1

        crop = raw_image[r_min:r_max, c_min:c_max].copy()
        crop_mask = mask[r_min:r_max, c_min:c_max]

        with h5py.File(output_dir / f"mito_{lbl:04d}.h5", "w") as f:
            f.create_dataset("raw", data=crop, compression="gzip")
            f.create_dataset("mask", data=crop_mask.astype(np.uint8), compression="gzip")


def export_directory_for_ilastik(masks_dir: Path, processed_dir: Path, output_dir: Path):
    """Export all instances from a directory of masks."""
    mask_files = sorted(masks_dir.glob("*_mask.tif"))
    for mask_path in tqdm(mask_files, desc="Exporting for Ilastik"):
        mask = tifffile.imread(str(mask_path)).astype(np.int32)
        # Find corresponding raw image
        raw_name = mask_path.stem.replace("_mask", "") + ".tif"
        raw_path = processed_dir / raw_name
        if raw_path.exists():
            raw = tifffile.imread(str(raw_path)).astype(np.float32)
            raw = raw / (raw.max() + 1e-8)
            export_crops_for_ilastik(mask, raw, output_dir)


def run_ilastik_headless(project_path: str, input_dir: str, output_dir: str):
    """Run Ilastik object classification in headless (batch) mode."""
    input_files = sorted(Path(input_dir).glob("*.h5"))
    cmd = [
        "ilastik",
        "--headless",
        f"--project={project_path}",
        "--output_format=hdf5",
        f"--output_filename_format={output_dir}/{{nickname}}_predictions.h5",
        "--export_source=Object Predictions",
    ] + [str(f) for f in input_files]

    subprocess.run(cmd, check=True)


def import_ilastik_labels(predictions_dir: Path) -> dict:
    """Import Ilastik object classification predictions."""
    labels = {}
    for pred_file in sorted(predictions_dir.glob("*_predictions.h5")):
        stem_parts = pred_file.stem.split("_")
        try:
            instance_id = int(stem_parts[1])
        except (IndexError, ValueError):
            continue
        with h5py.File(pred_file, "r") as f:
            probs = f["exported_data"][:]
            predicted_class = int(np.argmax(probs))
            labels[instance_id] = "UNHEALTHY" if predicted_class == 1 else "HEALTHY"
    return labels


def save_ilastik_labels_csv(labels: dict, output_path: Path):
    """Save Ilastik labels as CSV."""
    df = pd.DataFrame([
        {"instance_id": k, "label": v} for k, v in labels.items()
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ilastik labels saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ilastik integration")
    parser.add_argument("--export", action="store_true", help="Export crops for Ilastik")
    parser.add_argument("--import-labels", action="store_true", help="Import Ilastik predictions")
    parser.add_argument("--masks", help="Masks directory")
    parser.add_argument("--processed", help="Processed images directory")
    parser.add_argument("--predictions", help="Ilastik predictions directory")
    parser.add_argument("--output", required=True, help="Output directory or CSV")
    args = parser.parse_args()

    if args.export:
        export_directory_for_ilastik(
            Path(args.masks), Path(args.processed), Path(args.output)
        )
    elif args.import_labels:
        labels = import_ilastik_labels(Path(args.predictions))
        save_ilastik_labels_csv(labels, Path(args.output))
