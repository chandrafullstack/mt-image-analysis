"""
Run the full pipeline on the EPFL dataset (subset) to populate the dashboard.
This processes real EM data end-to-end without needing empanada/MitoNet installed.
"""
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from PIL import Image
from skimage import measure, morphology, exposure

# Config
NUM_SLICES = 10  # Process first N slices for demo
PIXEL_SIZE_UM = 0.008  # 8 nm per pixel
OUTPUT_DIR = Path("outputs")
CROPS_DIR = OUTPUT_DIR / "crops"
METRICS_DIR = OUTPUT_DIR / "metrics"

CROPS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading EPFL dataset...")
raw_stack = tifffile.imread("data/raw/epfl_rat/training.tif")
gt_stack = tifffile.imread("data/raw/epfl_rat/training_groundtruth.tif")
print(f"  Stack: {raw_stack.shape}, GT: {gt_stack.shape}")

all_records = []
instance_counter = 0

for z in range(min(NUM_SLICES, raw_stack.shape[0])):
    print(f"  Processing slice {z+1}/{NUM_SLICES}...")
    raw = raw_stack[z].astype(np.float32) / 255.0
    gt = (gt_stack[z] > 128).astype(np.uint8)

    # Clean up GT mask
    gt_clean = morphology.remove_small_objects(gt.astype(bool), min_size=200)
    gt_clean = morphology.remove_small_holes(gt_clean, area_threshold=100)

    # Instance segmentation via connected components
    instance_mask = measure.label(gt_clean).astype(np.int32)
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]

    for lbl in labels:
        instance_counter += 1
        binary = (instance_mask == lbl).astype(np.uint8)
        props_list = measure.regionprops(binary)
        if not props_list:
            continue
        prop = props_list[0]

        # Skip very small instances
        if prop.area < 300:
            continue

        s = PIXEL_SIZE_UM
        area_um2 = prop.area * (s ** 2)
        perim_um = prop.perimeter * s
        major_um = prop.major_axis_length * s
        minor_um = prop.minor_axis_length * s
        aspect_ratio = major_um / (minor_um + 1e-8)
        form_factor = (4 * np.pi * area_um2) / (perim_um ** 2 + 1e-8)
        roundness = (4 * area_um2) / (np.pi * major_um ** 2 + 1e-8)

        # G-ratio estimation
        membrane_px = max(1, int(round(0.01 / PIXEL_SIZE_UM)))
        inner = morphology.erosion(binary, morphology.disk(membrane_px))
        inner_props = measure.regionprops(inner)
        if inner_props:
            area_inner = inner_props[0].area
            d_outer = 2 * np.sqrt(prop.area / np.pi) * s
            d_inner = 2 * np.sqrt(area_inner / np.pi) * s
            g_ratio = d_inner / (d_outer + 1e-8)
        else:
            g_ratio = float("nan")

        # Export crop as PNG
        rows, cols = np.where(binary)
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        crop = raw[r_min:r_max, c_min:c_max].copy()
        crop_mask = binary[r_min:r_max, c_min:c_max]
        # Darken outside mask slightly for context
        crop[~crop_mask.astype(bool)] *= 0.3

        img_uint8 = (crop * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(CROPS_DIR / f"mito_{instance_counter:04d}.png")

        all_records.append({
            "label": instance_counter,
            "source_file": f"slice_{z:04d}",
            "area": prop.area,
            "perimeter": prop.perimeter,
            "area_um2": round(area_um2, 6),
            "perimeter_um": round(perim_um, 4),
            "major_axis_um": round(major_um, 4),
            "minor_axis_um": round(minor_um, 4),
            "aspect_ratio": round(aspect_ratio, 3),
            "form_factor": round(form_factor, 4),
            "roundness": round(roundness, 4),
            "eccentricity": round(prop.eccentricity, 4),
            "solidity": round(prop.solidity, 4),
            "g_ratio": round(g_ratio, 4) if not np.isnan(g_ratio) else None,
        })

print(f"\nExtracted {len(all_records)} mitochondria instances")

# Create DataFrame and assign labels
df = pd.DataFrame(all_records)

# Rule-based labeling
df["unhealthy_score"] = 0
df.loc[df["aspect_ratio"] < 1.5, "unhealthy_score"] += 1
df.loc[(df["form_factor"] < 0.4) | (df["form_factor"] > 1.2), "unhealthy_score"] += 1
df.loc[df["roundness"] > 0.7, "unhealthy_score"] += 1
df.loc[(df["g_ratio"].notna()) & ((df["g_ratio"] < 0.5) | (df["g_ratio"] > 0.85)), "unhealthy_score"] += 1

df["label_rule_based"] = np.where(df["unhealthy_score"] >= 2, "UNHEALTHY", "HEALTHY")
df["label_final"] = df["label_rule_based"]

# Shape categorization
def categorize_shape(row):
    if row["aspect_ratio"] > 2.0:
        return "ELONGATED"
    elif row["roundness"] > 0.8 and row["aspect_ratio"] < 1.5:
        return "CIRCULAR"
    else:
        return "OTHER"

df["shape_category"] = df.apply(categorize_shape, axis=1)

# Fission/fusion detection (simplified: uses solidity + pinch heuristic)
from scipy import ndimage

def detect_fission_state(row):
    """Quick heuristic: low solidity + elongated = possible fission."""
    solidity = row.get("solidity", 1.0)
    eccentricity = row.get("eccentricity", 0)
    concavity = 1.0 - solidity

    if concavity > 0.15 and eccentricity > 0.85:
        return "FISSION"
    elif concavity > 0.2 and eccentricity > 0.9:
        return "FUSION"
    else:
        return "NORMAL"

df["fission_fusion_state"] = df.apply(detect_fission_state, axis=1)

# Myelin context: for EPFL demo, assign based on slice position (placeholder)
# In real usage, this comes from myelin_segmentation.py
df["myelin_context"] = np.where(
    df["g_ratio"].notna() & (df["g_ratio"] > 0.95),
    "POORLY_MYELINATED",
    np.where(df["g_ratio"].notna(), "WELL_MYELINATED", "UNASSIGNED")
)

# Save
output_csv = METRICS_DIR / "features_with_gratio.csv"
df.to_csv(output_csv, index=False)

healthy_count = (df["label_final"] == "HEALTHY").sum()
unhealthy_count = (df["label_final"] == "UNHEALTHY").sum()
fission_count = (df["fission_fusion_state"] == "FISSION").sum()
fusion_count = (df["fission_fusion_state"] == "FUSION").sum()
print(f"  Healthy: {healthy_count}, Unhealthy: {unhealthy_count}")
print(f"  Fission: {fission_count}, Fusion: {fusion_count}")
print(f"  Shapes: {df['shape_category'].value_counts().to_dict()}")
print(f"  Mean G-ratio: {df['g_ratio'].mean():.4f}")
print(f"  Saved -> {output_csv}")
print(f"  Crops -> {CROPS_DIR} ({len(list(CROPS_DIR.glob('*.png')))} images)")
print("\nDone! Ready to launch dashboard.")
