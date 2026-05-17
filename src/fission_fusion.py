"""
Fission/Fusion state detection for mitochondria.

Fission (dividing): constriction point visible, dumbbell/hourglass shape.
Fusion (merging): two membranes in close apposition, bridge structure.
Normal: single intact organelle, neither dividing nor fusing.

Detection approach:
- Skeleton analysis: fission shows a pinch-point (narrow cross-section)
- Concavity analysis: fission creates deep concavities
- Topology: fission candidates have convexity defects / multi-lobe structure
"""
import numpy as np
import pandas as pd
from skimage import measure, morphology
from scipy import ndimage


def detect_fission_fusion(instance_mask: np.ndarray,
                          pixel_size_um: float = 0.008) -> pd.DataFrame:
    """
    Detect fission/fusion state for each mitochondrion instance.

    Args:
        instance_mask: int32 instance label map (0=background)
        pixel_size_um: physical pixel size

    Returns:
        DataFrame with instance_id, state (FISSION/FUSION/NORMAL), and
        supporting measurements.
    """
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]
    records = []

    for lbl in labels:
        binary = (instance_mask == lbl).astype(np.uint8)
        props = measure.regionprops(binary)
        if not props:
            continue
        prop = props[0]

        # Skip very small instances
        if prop.area < 200:
            records.append({"instance_id": int(lbl), "fission_fusion_state": "NORMAL",
                           "confidence": 1.0, "concavity_ratio": 0, "pinch_ratio": 1.0})
            continue

        state, confidence, concavity, pinch = _classify_state(binary, prop)

        records.append({
            "instance_id": int(lbl),
            "fission_fusion_state": state,
            "confidence": round(confidence, 3),
            "concavity_ratio": round(concavity, 4),
            "pinch_ratio": round(pinch, 4),
        })

    return pd.DataFrame(records)


def _classify_state(binary: np.ndarray, prop) -> tuple:
    """
    Classify a single mitochondrion's fission/fusion state.

    Returns: (state, confidence, concavity_ratio, pinch_ratio)
    """
    # 1. Convexity defects: fission creates deep concavities
    convex_area = prop.convex_area if hasattr(prop, 'convex_area') else prop.area
    solidity = prop.solidity  # area / convex_area
    concavity_ratio = 1.0 - solidity  # higher = more concave

    # 2. Pinch ratio: minimum cross-section width / maximum width
    # For fission, there's a narrow "neck" connecting two lobes
    pinch_ratio = _compute_pinch_ratio(binary)

    # 3. Eccentricity: fission candidates are often elongated
    eccentricity = prop.eccentricity

    # Classification logic
    if pinch_ratio < 0.3 and concavity_ratio > 0.15:
        # Deep pinch + concavity = fission
        state = "FISSION"
        confidence = min(1.0, (0.3 - pinch_ratio) * 3 + concavity_ratio)
    elif concavity_ratio > 0.2 and eccentricity > 0.9:
        # High concavity + very elongated = possible fusion
        state = "FUSION"
        confidence = min(1.0, concavity_ratio * 2)
    elif pinch_ratio < 0.4 and concavity_ratio > 0.1:
        # Mild pinch = possible fission (lower confidence)
        state = "FISSION"
        confidence = 0.5
    else:
        state = "NORMAL"
        confidence = 1.0 - concavity_ratio  # more convex = more confident

    return state, confidence, concavity_ratio, pinch_ratio


def _compute_pinch_ratio(binary: np.ndarray) -> float:
    """
    Compute the pinch ratio: minimum cross-section width along the
    major axis divided by the maximum width.

    A ratio near 0 indicates a pinch point (fission).
    A ratio near 1 indicates uniform width (normal).
    """
    # Use distance transform to find the "thickness" at each point
    dist = ndimage.distance_transform_edt(binary)

    # Skeleton to find the medial axis
    skeleton = morphology.skeletonize(binary)
    if not np.any(skeleton):
        return 1.0

    # Width along skeleton = 2 * distance_transform value at skeleton pixels
    widths = dist[skeleton] * 2
    if len(widths) < 3:
        return 1.0

    min_width = np.percentile(widths, 5)  # 5th percentile to avoid noise
    max_width = np.percentile(widths, 95)

    if max_width < 1e-8:
        return 1.0

    return min_width / max_width


def categorize_shape(aspect_ratio: float, roundness: float) -> str:
    """
    Categorize mitochondrial shape into Elongated / Circular / Other.
    """
    if aspect_ratio > 2.0:
        return "ELONGATED"
    elif roundness > 0.8 and aspect_ratio < 1.5:
        return "CIRCULAR"
    else:
        return "OTHER"


if __name__ == "__main__":
    import argparse
    import tifffile

    parser = argparse.ArgumentParser(description="Detect fission/fusion state")
    parser.add_argument("--masks", required=True, help="Instance mask .tif")
    parser.add_argument("--output", required=True, help="Output CSV")
    args = parser.parse_args()

    mask = tifffile.imread(args.masks).astype(np.int32)
    df = detect_fission_fusion(mask)
    df.to_csv(args.output, index=False)
    print(f"Detected fission/fusion for {len(df)} instances → {args.output}")
