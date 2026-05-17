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
    args = parser.parse_args()

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
