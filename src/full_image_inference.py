"""Production inference on full EM images with multiple cellular structures.

Pipeline:
1) Preprocess full grayscale image.
2) Detect mitochondria candidates (U-Net or heuristic fallback).
3) Extract instance crops + morphology features.
4) Classify each instance as healthy/unhealthy (CNN or rule-based fallback).
5) Write dashboard-ready outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from skimage import exposure, filters, measure, morphology

from src.cnn_model import build_resnet_classifier, build_unet

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PIXEL_SIZE_UM = 0.008


def _load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
        if arr.ndim == 3:
            arr = arr[0]
    else:
        arr = np.array(Image.open(path).convert("L"))

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def _preprocess(img: np.ndarray) -> np.ndarray:
    return exposure.equalize_adapthist(img, clip_limit=0.03).astype(np.float32)


def _tile_predict_unet(img: np.ndarray, model, device, tile_size=512, overlap=64) -> np.ndarray:
    h, w = img.shape
    stride = tile_size - overlap

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for r in range(0, h, stride):
        for c in range(0, w, stride):
            r2 = min(r + tile_size, h)
            c2 = min(c + tile_size, w)
            patch = img[r:r2, c:c2]

            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = np.pad(
                    patch,
                    ((0, tile_size - patch.shape[0]), (0, tile_size - patch.shape[1])),
                    mode="reflect",
                )

            x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = model(x).squeeze().cpu().numpy()

            pred = pred[: (r2 - r), : (c2 - c)]
            prob_sum[r:r2, c:c2] += pred
            count[r:r2, c:c2] += 1.0

    return prob_sum / (count + 1e-8)


def _segment_mito(img_prep: np.ndarray, seg_method: str, unet_weights: str | None) -> np.ndarray:
    if seg_method == "unet" and unet_weights:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for --seg-method unet") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_unet(pretrained=False)
        model.load_state_dict(torch.load(unet_weights, map_location=device))
        model.to(device).eval()

        prob = _tile_predict_unet(img_prep, model, device)
        binary = prob > 0.5
    else:
        # Heuristic fallback for labs without trained segmentation weights.
        thr = filters.threshold_otsu(img_prep)
        binary = img_prep < thr * 0.95

    binary = morphology.remove_small_objects(binary, min_size=80)
    binary = morphology.remove_small_holes(binary, area_threshold=80)
    return measure.label(binary).astype(np.int32)


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


def _rule_label(aspect_ratio, form_factor, roundness, g_ratio):
    score = 0
    if aspect_ratio < 1.5:
        score += 1
    if form_factor < 0.4 or form_factor > 1.2:
        score += 1
    if roundness > 0.7:
        score += 1
    if np.isfinite(g_ratio) and (g_ratio < 0.5 or g_ratio > 0.85):
        score += 1
    return ("UNHEALTHY" if score >= 2 else "HEALTHY"), score


def _load_classifier(weights_path: str | None):
    if not weights_path:
        return None, None

    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("PyTorch + torchvision are required for --classifier-weights") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet_classifier(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    return (model, device), tf


def _classify_crop(crop_gray: np.ndarray, classifier_bundle, tf):
    if classifier_bundle is None:
        return None
    model, device = classifier_bundle
    pil = Image.fromarray((crop_gray * 255).clip(0, 255).astype(np.uint8))
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
    return "UNHEALTHY" if pred == 1 else "HEALTHY"


def run_full_image_inference(
    input_dir: Path,
    metrics_out: Path,
    crops_out: Path,
    seg_method: str = "heuristic",
    unet_weights: str | None = None,
    classifier_weights: str | None = None,
):
    input_dir = Path(input_dir)
    metrics_out = Path(metrics_out)
    crops_out = Path(crops_out)
    crops_out.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    classifier_bundle, clf_tf = _load_classifier(classifier_weights)

    rows = []
    instance_id = 1

    files = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    for path in files:
        raw = _load_gray(path)
        prep = _preprocess(raw)
        inst = _segment_mito(prep, seg_method=seg_method, unet_weights=unet_weights)

        for prop in measure.regionprops(inst):
            if prop.area < 80:
                continue

            mask = inst == prop.label
            rr, cc = np.where(mask)
            r1, r2 = rr.min(), rr.max() + 1
            c1, c2 = cc.min(), cc.max() + 1

            crop = raw[r1:r2, c1:c2].copy()
            crop_mask = mask[r1:r2, c1:c2]
            crop[~crop_mask] *= 0.3

            s = PIXEL_SIZE_UM
            area_um2 = prop.area * (s ** 2)
            perimeter_um = prop.perimeter * s
            major_um = prop.axis_major_length * s
            minor_um = prop.axis_minor_length * s
            aspect_ratio = major_um / (minor_um + 1e-8)
            form_factor = (4 * np.pi * area_um2) / (perimeter_um ** 2 + 1e-8)
            roundness = (4 * area_um2) / (np.pi * major_um ** 2 + 1e-8)

            membrane_px = max(1, int(round(0.01 / PIXEL_SIZE_UM)))
            inner = morphology.erosion(crop_mask, morphology.disk(membrane_px))
            area_inner = float(inner.sum())
            d_outer = 2 * np.sqrt(prop.area / np.pi) * s
            d_inner = 2 * np.sqrt(area_inner / np.pi) * s if area_inner > 0 else 0.0
            g_ratio = d_inner / (d_outer + 1e-8) if d_outer > 0 else np.nan

            label_rule, unhealthy_score = _rule_label(aspect_ratio, form_factor, roundness, g_ratio)
            label_model = _classify_crop(crop, classifier_bundle, clf_tf)
            label_final = label_model if label_model is not None else label_rule

            crop_name = f"mito_{instance_id:04d}.png"
            Image.fromarray((crop * 255).clip(0, 255).astype(np.uint8)).save(crops_out / crop_name)

            rows.append(
                {
                    "label": instance_id,
                    "source_file": path.name,
                    "area": float(prop.area),
                    "perimeter": float(prop.perimeter),
                    "area_um2": round(float(area_um2), 6),
                    "perimeter_um": round(float(perimeter_um), 4),
                    "major_axis_um": round(float(major_um), 4),
                    "minor_axis_um": round(float(minor_um), 4),
                    "aspect_ratio": round(float(aspect_ratio), 3),
                    "form_factor": round(float(form_factor), 4),
                    "roundness": round(float(roundness), 4),
                    "eccentricity": round(float(prop.eccentricity), 4),
                    "solidity": round(float(prop.solidity), 4),
                    "g_ratio": round(float(g_ratio), 4) if np.isfinite(g_ratio) else None,
                    "unhealthy_score": int(unhealthy_score),
                    "label_rule_based": label_rule,
                    "label_final": label_final,
                    "shape_category": _shape_category(aspect_ratio, roundness),
                    "fission_fusion_state": _fission_fusion(float(prop.solidity), float(prop.eccentricity)),
                    "myelin_context": "UNASSIGNED",
                }
            )
            instance_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(metrics_out, index=False)
    print(f"Processed images: {len(files)}")
    print(f"Detected mitochondria: {len(df)}")
    print(f"Metrics -> {metrics_out}")
    print(f"Crops -> {crops_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full-image mitochondria detection + health classification")
    parser.add_argument("--input-dir", required=True, help="Directory with full EM images")
    parser.add_argument(
        "--metrics-out",
        default="outputs/metrics/features_with_gratio.csv",
        help="Dashboard metrics CSV output path",
    )
    parser.add_argument(
        "--crops-out",
        default="outputs/crops",
        help="Dashboard crops output directory",
    )
    parser.add_argument(
        "--seg-method",
        default="heuristic",
        choices=["heuristic", "unet"],
        help="Segmentation method for full images",
    )
    parser.add_argument("--unet-weights", default=None, help="Path to trained U-Net weights (.pt)")
    parser.add_argument("--classifier-weights", default=None, help="Path to trained classifier weights (.pt)")
    args = parser.parse_args()

    run_full_image_inference(
        input_dir=Path(args.input_dir),
        metrics_out=Path(args.metrics_out),
        crops_out=Path(args.crops_out),
        seg_method=args.seg_method,
        unet_weights=args.unet_weights,
        classifier_weights=args.classifier_weights,
    )
