"""Visualisation utilities for mitochondria health pipeline."""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from pathlib import Path


def overlay_predictions(image: np.ndarray,
                        instance_mask: np.ndarray,
                        labels: dict,
                        alpha: float = 0.4) -> np.ndarray:
    """
    Overlay colour-coded instance predictions on the original EM image.
    Green = HEALTHY, Red = UNHEALTHY
    """
    overlay = color.gray2rgb(image.astype(np.float64))
    for instance_id, health in labels.items():
        colour = np.array([0, 1, 0]) if health == "HEALTHY" else np.array([1, 0, 0])
        mask = (instance_mask == instance_id)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colour
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def plot_gratio_distribution(gratio_values: dict, save_path: str = None):
    """Plot G-ratio distribution for healthy vs unhealthy populations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in gratio_values.items():
        ax.hist(values, bins=30, alpha=0.6, label=label)
    ax.axvline(0.6, color="gray", linestyle="--", label="Healthy range (0.6–0.8)")
    ax.axvline(0.8, color="gray", linestyle="--")
    ax.set_xlabel("G-ratio (d_inner / d_outer)")
    ax.set_ylabel("Count")
    ax.set_title("G-Ratio Distribution by Health Class")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def export_crops_as_png(instance_mask: np.ndarray, raw_image: np.ndarray,
                        output_dir: Path):
    """Export individual mitochondrion crops as PNG for the web dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]

    from PIL import Image

    for lbl in labels:
        mask = (instance_mask == lbl)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1

        crop = raw_image[r_min:r_max, c_min:c_max].copy()
        # Zero out pixels outside the instance
        crop_mask = mask[r_min:r_max, c_min:c_max]
        crop[~crop_mask] = 0

        img_uint8 = (crop * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(output_dir / f"mito_{lbl:04d}.png")
