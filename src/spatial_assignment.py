"""
Spatial assignment: assign each mitochondrion to its parent neuron/cell
and generate cross-tabulation reports.

This module links the mitochondria segmentation results with the myelin/axon
segmentation to determine which mitochondria are inside well-myelinated vs.
poorly-myelinated cells.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def assign_mitos_to_neurons(mito_mask: np.ndarray,
                            axon_mask: np.ndarray,
                            neuron_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each mitochondrion to its enclosing neuron/axon.

    Strategy: for each mito instance, check which axon region contains
    the majority of its pixels.

    Args:
        mito_mask: Instance mask of mitochondria
        axon_mask: Instance mask of axons (from myelin segmentation)
        neuron_df: DataFrame with neuron_id and myelin_health columns

    Returns:
        DataFrame mapping mito_id → neuron_id, myelin_health
    """
    mito_labels = np.unique(mito_mask)
    mito_labels = mito_labels[mito_labels > 0]

    assignments = []
    for mito_id in mito_labels:
        mito_pixels = (mito_mask == mito_id)
        # Which axon regions overlap with this mito?
        overlapping_axons = axon_mask[mito_pixels]
        overlapping_axons = overlapping_axons[overlapping_axons > 0]

        if len(overlapping_axons) == 0:
            # Mito not inside any axon — could be in extracellular space or glia
            assignments.append({
                "mito_id": int(mito_id),
                "neuron_id": None,
                "myelin_health": "UNASSIGNED",
            })
            continue

        # Majority vote: assign to the axon with most overlap
        values, counts = np.unique(overlapping_axons, return_counts=True)
        best_axon = int(values[np.argmax(counts)])

        # Look up myelin health for this neuron
        neuron_row = neuron_df[neuron_df["neuron_id"] == best_axon]
        if not neuron_row.empty:
            myelin_health = neuron_row.iloc[0]["myelin_health"]
        else:
            myelin_health = "UNKNOWN"

        assignments.append({
            "mito_id": int(mito_id),
            "neuron_id": best_axon,
            "myelin_health": myelin_health,
        })

    return pd.DataFrame(assignments)


def generate_per_image_report(mito_features: pd.DataFrame,
                              fission_fusion_df: pd.DataFrame,
                              assignment_df: pd.DataFrame,
                              neuron_df: pd.DataFrame,
                              image_name: str) -> dict:
    """
    Generate the complete per-image report with all 13 required metrics.

    Args:
        mito_features: DataFrame with mito morphological features + health label
        fission_fusion_df: DataFrame with fission/fusion states
        assignment_df: DataFrame mapping mitos to neurons
        neuron_df: DataFrame with neuron G-ratios and myelin health
        image_name: Identifier for this image

    Returns:
        Dictionary with all required metrics
    """
    # Merge all data
    df = mito_features.copy()
    if "instance_id" in fission_fusion_df.columns:
        ff_map = fission_fusion_df.set_index("instance_id")["fission_fusion_state"].to_dict()
        df["fission_fusion_state"] = df["label"].map(ff_map).fillna("NORMAL")
    else:
        df["fission_fusion_state"] = "NORMAL"

    if not assignment_df.empty:
        assign_map = assignment_df.set_index("mito_id")["myelin_health"].to_dict()
        df["myelin_context"] = df["label"].map(assign_map).fillna("UNASSIGNED")
    else:
        df["myelin_context"] = "UNASSIGNED"

    # Determine health label column
    health_col = "label_final" if "label_final" in df.columns else "label_rule_based"
    if health_col not in df.columns:
        health_col = None

    # Shape categorization
    if "shape_category" not in df.columns:
        from src.fission_fusion import categorize_shape
        df["shape_category"] = df.apply(
            lambda r: categorize_shape(r.get("aspect_ratio", 1), r.get("roundness", 0.5)),
            axis=1
        )

    total = len(df)

    report = {
        "image_name": image_name,
        "total_mitochondria": total,
    }

    if health_col:
        healthy = df[health_col] == "HEALTHY"
        unhealthy = df[health_col] == "UNHEALTHY"

        report["healthy_count"] = int(healthy.sum())
        report["unhealthy_count"] = int(unhealthy.sum())

        # Cross-tabulations with myelin
        well_myel = df["myelin_context"] == "WELL_MYELINATED"
        poor_myel = df["myelin_context"] == "POORLY_MYELINATED"

        report["healthy_in_well_myelinated"] = int((healthy & well_myel).sum())
        report["unhealthy_in_well_myelinated"] = int((unhealthy & well_myel).sum())
        report["healthy_in_poorly_myelinated"] = int((healthy & poor_myel).sum())
        report["unhealthy_in_poorly_myelinated"] = int((unhealthy & poor_myel).sum())
    else:
        report["healthy_count"] = 0
        report["unhealthy_count"] = 0
        report["healthy_in_well_myelinated"] = 0
        report["unhealthy_in_well_myelinated"] = 0
        report["healthy_in_poorly_myelinated"] = 0
        report["unhealthy_in_poorly_myelinated"] = 0

    # Shape distribution
    shape_counts = df["shape_category"].value_counts().to_dict()
    report["shape_elongated"] = shape_counts.get("ELONGATED", 0)
    report["shape_circular"] = shape_counts.get("CIRCULAR", 0)
    report["shape_other"] = shape_counts.get("OTHER", 0)

    # Shape × myelin
    if "myelin_context" in df.columns:
        well = df[df["myelin_context"] == "WELL_MYELINATED"]
        poor = df[df["myelin_context"] == "POORLY_MYELINATED"]
        report["shape_in_well_myelinated"] = well["shape_category"].value_counts().to_dict()
        report["shape_in_poorly_myelinated"] = poor["shape_category"].value_counts().to_dict()

    # Fission/Fusion × myelin
    ff_counts = df["fission_fusion_state"].value_counts().to_dict()
    report["fission_count"] = ff_counts.get("FISSION", 0)
    report["fusion_count"] = ff_counts.get("FUSION", 0)
    report["normal_state_count"] = ff_counts.get("NORMAL", 0)

    if "myelin_context" in df.columns:
        well_ff = df[df["myelin_context"] == "WELL_MYELINATED"]["fission_fusion_state"].value_counts().to_dict()
        poor_ff = df[df["myelin_context"] == "POORLY_MYELINATED"]["fission_fusion_state"].value_counts().to_dict()
        report["fission_fusion_in_well_myelinated"] = well_ff
        report["fission_fusion_in_poorly_myelinated"] = poor_ff

    # Neuron G-ratios
    if not neuron_df.empty:
        report["neuron_count"] = len(neuron_df)
        report["mean_neuron_gratio"] = round(float(neuron_df["g_ratio"].mean()), 4)
        report["neuron_gratios"] = neuron_df[["neuron_id", "g_ratio", "myelin_health"]].to_dict("records")
    else:
        report["neuron_count"] = 0
        report["mean_neuron_gratio"] = None
        report["neuron_gratios"] = []

    return report


def reports_to_csv(reports: list, output_path: Path):
    """Save list of per-image report dicts to CSV (flattened)."""
    # Flatten nested fields for CSV
    flat_records = []
    for r in reports:
        flat = {k: v for k, v in r.items()
                if not isinstance(v, (dict, list))}
        flat_records.append(flat)

    df = pd.DataFrame(flat_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} image reports → {output_path}")
