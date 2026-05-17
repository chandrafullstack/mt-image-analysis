"""Ingest manually labeled EM images from incoming folders into dashboard metrics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, measure, morphology

# Keep this aligned with the project pipeline defaults.
PIXEL_SIZE_UM = 0.008
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

INCOMING_HEALTHY = Path("incoming/healthy")
INCOMING_UNHEALTHY = Path("incoming/unhealthy")
OUTGOING_HEALTHY = Path("outgoing/processed/healthy")
OUTGOING_UNHEALTHY = Path("outgoing/processed/unhealthy")
OUTGOING_REJECTED = Path("outgoing/rejected")

METRICS_PATH = Path("outputs/metrics/features_with_gratio.csv")
CROPS_DIR = Path("outputs/crops")


@dataclass
class IngestResult:
    processed: int
    rejected: int
    appended: int


def _ensure_paths() -> None:
    for p in [
        INCOMING_HEALTHY,
        INCOMING_UNHEALTHY,
        OUTGOING_HEALTHY,
        OUTGOING_UNHEALTHY,
        OUTGOING_REJECTED,
        METRICS_PATH.parent,
        CROPS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _iter_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def _safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_dir / f"{stamp}_{src.name}"
    shutil.move(str(src), str(dst))
    return dst


def _shape_category(aspect_ratio: float, roundness: float) -> str:
    if aspect_ratio > 2.0:
        return "ELONGATED"
    if roundness > 0.8 and aspect_ratio < 1.5:
        return "CIRCULAR"
    return "OTHER"


def _fission_fusion(solidity: float, eccentricity: float) -> str:
    concavity = 1.0 - solidity
    if concavity > 0.15 and eccentricity > 0.85:
        return "FISSION"
    if concavity > 0.2 and eccentricity > 0.9:
        return "FUSION"
    return "NORMAL"


def _extract_features(image_path: Path) -> dict | None:
    # Professional uploads are expected to be mitochondria-centered crops.
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return None

    # Otsu segmentation with fallback for near-uniform images.
    try:
        thr = filters.threshold_otsu(arr)
        binary = arr < thr
    except ValueError:
        binary = arr < arr.mean()

    binary = morphology.remove_small_objects(binary, min_size=40)
    binary = morphology.remove_small_holes(binary, area_threshold=40)

    labels = measure.label(binary)
    props = measure.regionprops(labels)
    if not props:
        return None

    prop = max(props, key=lambda p: p.area)
    if prop.area < 40:
        return None

    s = PIXEL_SIZE_UM
    area_um2 = prop.area * (s ** 2)
    perim_um = prop.perimeter * s
    major_um = prop.axis_major_length * s
    minor_um = prop.axis_minor_length * s
    aspect_ratio = major_um / (minor_um + 1e-8)
    form_factor = (4 * np.pi * area_um2) / (perim_um ** 2 + 1e-8)
    roundness = (4 * area_um2) / (np.pi * major_um ** 2 + 1e-8)

    # Approximate mitochondrial g-ratio from a thin erosion shell.
    single = labels == prop.label
    membrane_px = max(1, int(round(0.01 / PIXEL_SIZE_UM)))
    inner = morphology.erosion(single, morphology.disk(membrane_px))
    area_inner = float(inner.sum())
    d_outer = 2 * np.sqrt(prop.area / np.pi) * s
    d_inner = 2 * np.sqrt(area_inner / np.pi) * s if area_inner > 0 else 0.0
    g_ratio = d_inner / (d_outer + 1e-8) if d_outer > 0 else np.nan

    return {
        "area": float(prop.area),
        "perimeter": float(prop.perimeter),
        "area_um2": round(float(area_um2), 6),
        "perimeter_um": round(float(perim_um), 4),
        "major_axis_um": round(float(major_um), 4),
        "minor_axis_um": round(float(minor_um), 4),
        "aspect_ratio": round(float(aspect_ratio), 3),
        "form_factor": round(float(form_factor), 4),
        "roundness": round(float(roundness), 4),
        "eccentricity": round(float(prop.eccentricity), 4),
        "solidity": round(float(prop.solidity), 4),
        "g_ratio": round(float(g_ratio), 4) if np.isfinite(g_ratio) else None,
    }


def ingest_incoming_feedback(quiet: bool = False) -> IngestResult:
    _ensure_paths()

    if METRICS_PATH.exists():
        df_existing = pd.read_csv(METRICS_PATH)
    else:
        df_existing = pd.DataFrame()

    next_label = 1
    if "label" in df_existing and len(df_existing) > 0:
        next_label = int(pd.to_numeric(df_existing["label"], errors="coerce").max()) + 1

    pending = []
    pending.extend([(p, "HEALTHY") for p in _iter_images(INCOMING_HEALTHY)])
    pending.extend([(p, "UNHEALTHY") for p in _iter_images(INCOMING_UNHEALTHY)])

    if not pending:
        return IngestResult(processed=0, rejected=0, appended=0)

    new_rows = []
    rejected = 0

    for image_path, label_final in pending:
        feats = _extract_features(image_path)
        if feats is None:
            rejected += 1
            _safe_move(image_path, OUTGOING_REJECTED)
            continue

        instance_id = next_label
        next_label += 1

        crop_filename = f"mito_{instance_id:04d}.png"
        crop_dst = CROPS_DIR / crop_filename
        Image.open(image_path).convert("L").save(crop_dst)

        shape_category = _shape_category(feats["aspect_ratio"], feats["roundness"])
        ff_state = _fission_fusion(feats["solidity"], feats["eccentricity"])

        unhealthy_score = 0 if label_final == "HEALTHY" else 3
        new_rows.append({
            "label": instance_id,
            "source_file": f"incoming_{image_path.name}",
            "area": feats["area"],
            "perimeter": feats["perimeter"],
            "area_um2": feats["area_um2"],
            "perimeter_um": feats["perimeter_um"],
            "major_axis_um": feats["major_axis_um"],
            "minor_axis_um": feats["minor_axis_um"],
            "aspect_ratio": feats["aspect_ratio"],
            "form_factor": feats["form_factor"],
            "roundness": feats["roundness"],
            "eccentricity": feats["eccentricity"],
            "solidity": feats["solidity"],
            "g_ratio": feats["g_ratio"],
            "unhealthy_score": unhealthy_score,
            "label_rule_based": label_final,
            "label_final": label_final,
            "shape_category": shape_category,
            "fission_fusion_state": ff_state,
            "myelin_context": "UNASSIGNED",
        })

        if label_final == "HEALTHY":
            _safe_move(image_path, OUTGOING_HEALTHY)
        else:
            _safe_move(image_path, OUTGOING_UNHEALTHY)

    if not new_rows:
        return IngestResult(processed=len(pending), rejected=rejected, appended=0)

    df_new = pd.DataFrame(new_rows)
    if len(df_existing) == 0:
        df_out = df_new
    else:
        # Ensure all expected columns exist before concatenation.
        all_cols = sorted(set(df_existing.columns).union(set(df_new.columns)))
        df_out = pd.concat(
            [df_existing.reindex(columns=all_cols), df_new.reindex(columns=all_cols)],
            ignore_index=True,
        )

    df_out.to_csv(METRICS_PATH, index=False)

    if not quiet:
        print(f"Ingested {len(new_rows)} incoming image(s); rejected {rejected}.")

    return IngestResult(processed=len(pending), rejected=rejected, appended=len(new_rows))


if __name__ == "__main__":
    result = ingest_incoming_feedback(quiet=False)
    print(
        f"Done. Processed={result.processed}, Appended={result.appended}, Rejected={result.rejected}"
    )
