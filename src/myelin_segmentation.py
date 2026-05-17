"""
Myelin sheath and axon segmentation + per-neuron G-ratio calculation.

The G-ratio here is the MYELINATION G-ratio:
    G-ratio = d_axon / d_fibre

Where:
- d_axon: equivalent diameter of the axon (inner boundary of myelin)
- d_fibre: equivalent diameter of the total nerve fibre (outer boundary of myelin)

Interpretation:
- 0.6–0.8: Healthy myelination
- > 0.8:   Demyelinated / thinly remyelinating (UNHEALTHY)
- < 0.6:   Hypermyelinated or actively remyelinating
"""
import numpy as np
import pandas as pd
from skimage import measure, morphology, filters, segmentation
from pathlib import Path
from typing import Tuple


def segment_myelin_from_gt(gt_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment myelin sheaths from a ground truth or thresholded EM image.

    In EM images, myelin appears as concentric dark rings around lighter axons.
    This function identifies the dark rings (myelin) and the enclosed
    lighter regions (axons).

    Args:
        gt_mask: Binary mask or intensity image where myelin is dark (0)
                 and axons/background are light (255/1).

    Returns:
        myelin_instance_mask: int32 label map of myelin sheaths
        axon_instance_mask: int32 label map of enclosed axons
    """
    # If binary, invert (myelin = dark = foreground)
    if gt_mask.max() <= 1:
        binary = gt_mask < 0.5
    else:
        binary = gt_mask < 128

    # Clean up
    cleaned = morphology.remove_small_objects(binary, min_size=500)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3))

    # Label connected myelin regions
    myelin_labels = measure.label(cleaned).astype(np.int32)

    # Find axons: holes inside myelin rings
    filled = morphology.remove_small_holes(cleaned, area_threshold=50000)
    axon_regions = np.logical_and(filled, ~cleaned)
    axon_labels = measure.label(axon_regions).astype(np.int32)

    return myelin_labels, axon_labels


def compute_neuron_gratio(myelin_mask: np.ndarray, axon_mask: np.ndarray,
                          pixel_size_um: float = 0.008) -> pd.DataFrame:
    """
    Compute G-ratio for each neuron (axon + myelin pair).

    G-ratio = d_axon / d_fibre
    where d_fibre includes both myelin + axon.

    Args:
        myelin_mask: Instance label map of myelin sheaths
        axon_mask: Instance label map of axons
        pixel_size_um: Physical pixel size in µm

    Returns:
        DataFrame with neuron_id, d_axon, d_fibre, g_ratio, myelin_health
    """
    axon_props = measure.regionprops(axon_mask)
    records = []

    for axon_prop in axon_props:
        axon_area = axon_prop.area
        d_axon = 2 * np.sqrt(axon_area / np.pi) * pixel_size_um

        # Find the enclosing myelin: dilate the axon and check overlap with myelin
        axon_binary = (axon_mask == axon_prop.label)
        dilated = morphology.binary_dilation(axon_binary, morphology.disk(5))
        overlap_labels = np.unique(myelin_mask[dilated])
        overlap_labels = overlap_labels[overlap_labels > 0]

        if len(overlap_labels) == 0:
            continue

        # Take the myelin region with most overlap
        best_myelin_label = max(
            overlap_labels,
            key=lambda lbl: np.sum(myelin_mask[dilated] == lbl)
        )

        # Total fibre = axon + its surrounding myelin
        fibre_mask = np.logical_or(
            axon_binary,
            (myelin_mask == best_myelin_label)
        )
        fibre_area = np.sum(fibre_mask)
        d_fibre = 2 * np.sqrt(fibre_area / np.pi) * pixel_size_um

        g_ratio = d_axon / (d_fibre + 1e-8)

        # Classify myelin health
        if 0.6 <= g_ratio <= 0.8:
            myelin_health = "WELL_MYELINATED"
        else:
            myelin_health = "POORLY_MYELINATED"

        records.append({
            "neuron_id": int(axon_prop.label),
            "axon_area_um2": round(axon_area * pixel_size_um ** 2, 6),
            "d_axon_um": round(d_axon, 4),
            "d_fibre_um": round(d_fibre, 4),
            "g_ratio": round(g_ratio, 4),
            "myelin_health": myelin_health,
            "centroid_row": int(axon_prop.centroid[0]),
            "centroid_col": int(axon_prop.centroid[1]),
        })

    return pd.DataFrame(records)


def classify_myelin_health(g_ratio: float) -> str:
    """Classify myelin health from G-ratio value."""
    if 0.6 <= g_ratio <= 0.8:
        return "WELL_MYELINATED"
    elif g_ratio > 0.8:
        return "POORLY_MYELINATED"  # demyelinated / thin remyelination
    else:
        return "POORLY_MYELINATED"  # hypermyelinated or abnormal


if __name__ == "__main__":
    import argparse
    import tifffile

    parser = argparse.ArgumentParser(description="Myelin segmentation & neuron G-ratio")
    parser.add_argument("--input", required=True, help="EM image or GT mask .tif")
    parser.add_argument("--output", required=True, help="Output CSV for neuron G-ratios")
    parser.add_argument("--pixel-size", type=float, default=0.008)
    args = parser.parse_args()

    img = tifffile.imread(args.input)
    myelin_mask, axon_mask = segment_myelin_from_gt(img)
    df = compute_neuron_gratio(myelin_mask, axon_mask, args.pixel_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Computed G-ratio for {len(df)} neurons → {args.output}")
