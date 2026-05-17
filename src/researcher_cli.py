"""Researcher-friendly entrypoint: point to healthy/unhealthy folders and update dashboard data."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.incoming_feedback import (
    INCOMING_HEALTHY,
    INCOMING_UNHEALTHY,
    SUPPORTED_EXTS,
    ingest_incoming_feedback,
)
from src.full_image_inference import run_full_image_inference
from src.cnn_model import train_classifier
from src.train_unet import train_unet


def _copy_images(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        target = dst_dir / path.name
        # Prevent overwrite when duplicate filenames exist.
        if target.exists():
            stem = path.stem
            suffix = path.suffix
            i = 1
            while (dst_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            target = dst_dir / f"{stem}_{i}{suffix}"
        shutil.copy2(path, target)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest expert-labeled EM images by pointing to source folders."
    )
    parser.add_argument(
        "--healthy-dir",
        type=str,
        help="Folder containing healthy mitochondria crops",
    )
    parser.add_argument(
        "--unhealthy-dir",
        type=str,
        help="Folder containing unhealthy mitochondria crops",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch dashboard after ingestion",
    )
    parser.add_argument(
        "--full-image-dir",
        type=str,
        help="Run full-image mitochondria detection + classification on this folder",
    )
    parser.add_argument(
        "--seg-method",
        choices=["heuristic", "unet"],
        default="heuristic",
        help="Segmentation method for full-image mode",
    )
    parser.add_argument(
        "--unet-weights",
        type=str,
        default=None,
        help="Path to trained U-Net weights for full-image mode",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to trained classifier weights for full-image mode",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for dashboard server when --serve is used",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for dashboard server when --serve is used",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain local models from labeled lab data before inference",
    )
    parser.add_argument(
        "--skip-classifier-train",
        action="store_true",
        help="Skip ResNet healthy/unhealthy retraining",
    )
    parser.add_argument(
        "--skip-segmentation-train",
        action="store_true",
        help="Skip U-Net segmentation retraining",
    )
    parser.add_argument(
        "--train-classifier-data",
        default="data/labeled/crops",
        help="Labeled crop folder (HEALTHY/UNHEALTHY) for ResNet retraining",
    )
    parser.add_argument(
        "--train-seg-images",
        default="data/labeled/segmentation/images",
        help="Annotated image folder for U-Net retraining",
    )
    parser.add_argument(
        "--train-seg-masks",
        default="data/labeled/segmentation/masks",
        help="Annotated mask folder for U-Net retraining",
    )
    parser.add_argument(
        "--models-out",
        default="outputs/models",
        help="Output folder for trained model weights",
    )
    parser.add_argument("--clf-epochs", type=int, default=25)
    parser.add_argument("--clf-batch-size", type=int, default=16)
    parser.add_argument("--clf-lr", type=float, default=1e-4)
    parser.add_argument("--unet-epochs", type=int, default=30)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    parser.add_argument("--unet-lr", type=float, default=1e-4)
    parser.add_argument("--unet-image-size", type=int, default=512)
    args = parser.parse_args()

    models_out = Path(args.models_out)

    if args.retrain:
        print("Retrain requested: updating local models from labeled lab data...")
        models_out.mkdir(parents=True, exist_ok=True)

        if not args.skip_segmentation_train:
            seg_images = Path(args.train_seg_images)
            seg_masks = Path(args.train_seg_masks)
            if seg_images.exists() and seg_masks.exists():
                train_unet(
                    images_dir=seg_images,
                    masks_dir=seg_masks,
                    output_dir=models_out,
                    image_size=args.unet_image_size,
                    epochs=args.unet_epochs,
                    batch_size=args.unet_batch_size,
                    lr=args.unet_lr,
                )
            else:
                print("Skipping U-Net retrain: segmentation image/mask folders not found.")

        if not args.skip_classifier_train:
            clf_data = Path(args.train_classifier_data)
            if clf_data.exists():
                train_classifier(
                    data_dir=clf_data,
                    output_dir=models_out,
                    epochs=args.clf_epochs,
                    lr=args.clf_lr,
                    batch_size=args.clf_batch_size,
                )
            else:
                print("Skipping classifier retrain: labeled crop folder not found.")

        # Auto-use newly trained weights if caller did not explicitly pass paths.
        if args.unet_weights is None:
            candidate = models_out / "unet_best.pt"
            if candidate.exists():
                args.unet_weights = str(candidate)
        if args.classifier_weights is None:
            candidate = models_out / "resnet50_best.pt"
            if candidate.exists():
                args.classifier_weights = str(candidate)

    if args.full_image_dir:
        run_full_image_inference(
            input_dir=Path(args.full_image_dir),
            metrics_out=Path("outputs/metrics/features_with_gratio.csv"),
            crops_out=Path("outputs/crops"),
            seg_method=args.seg_method,
            unet_weights=args.unet_weights,
            classifier_weights=args.classifier_weights,
        )

        if args.serve:
            import uvicorn

            uvicorn.run("app.main:app", host=args.host, port=args.port)
        return

    copied_h = 0
    copied_u = 0

    if args.healthy_dir:
        copied_h = _copy_images(Path(args.healthy_dir), INCOMING_HEALTHY)
    if args.unhealthy_dir:
        copied_u = _copy_images(Path(args.unhealthy_dir), INCOMING_UNHEALTHY)

    result = ingest_incoming_feedback(quiet=False)
    print(
        f"Copied healthy={copied_h}, unhealthy={copied_u}; processed={result.processed}, appended={result.appended}, rejected={result.rejected}"
    )

    if args.serve:
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
