"""Route Claude consensus pseudo-labels into incoming/ training folders.

Reads a CSV produced by ``claude_score_crops --passes >= 1`` and copies the
corresponding crops to ``incoming/healthy/`` or ``incoming/unhealthy/``,
subject to confidence + agreement thresholds.

Default policy (rigorous):
  - final_classification in {HEALTHY, UNHEALTHY}
  - label_agreement      >= 0.67   (>=2/3 passes agree)
  - calibrated_confidence >= 0.55

A separate ``--strict`` policy raises the bars (1.0 agreement, 0.80
calibrated) for the "trust without review" tier.

Single-pass CSVs are also supported (fallback to raw self-confidence) but
strongly discouraged for training data.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


HEALTHY_DIR_DEFAULT = Path("incoming/healthy")
UNHEALTHY_DIR_DEFAULT = Path("incoming/unhealthy")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Make single-pass and consensus CSVs share the same column names."""
    if "final_classification" in df.columns:
        return df
    if "classification" in df.columns:
        df = df.copy()
        df["final_classification"] = df["classification"]
        df["label_agreement"] = 1.0  # only one opinion
        df["calibrated_confidence"] = df["confidence"].fillna(0.0)
        df["mean_majority_confidence"] = df["confidence"].fillna(0.0)
        df["n_passes"] = 1
        return df
    raise SystemExit(
        "CSV missing both 'final_classification' and 'classification' columns."
    )


def route(
    predictions_csv: Path,
    crops_dir: Path,
    project_root: Path,
    healthy_dir: Path,
    unhealthy_dir: Path,
    min_agreement: float,
    min_calibrated: float,
    dry_run: bool,
    prefix: str,
) -> None:
    df = _normalise(pd.read_csv(predictions_csv))

    healthy_dir = (project_root / healthy_dir).resolve()
    unhealthy_dir = (project_root / unhealthy_dir).resolve()
    healthy_dir.mkdir(parents=True, exist_ok=True)
    unhealthy_dir.mkdir(parents=True, exist_ok=True)

    accept = df[
        df["final_classification"].isin(["HEALTHY", "UNHEALTHY"])
        & (df["label_agreement"].fillna(0) >= min_agreement)
        & (df["calibrated_confidence"].fillna(0) >= min_calibrated)
    ].copy()

    reject = df.drop(accept.index)

    print(f"Loaded {len(df)} rows from {predictions_csv}")
    print(f"  accepted: {len(accept)}   rejected: {len(reject)}")
    print(
        "  accept thresholds: "
        f"agreement>={min_agreement}, calibrated>={min_calibrated}"
    )
    print("Accept breakdown:")
    print(accept["final_classification"].value_counts().to_string())
    print("\nReject reasons (sample):")
    for _, r in reject.head(8).iterrows():
        print(
            f"  {r['crop_file']}: cls={r['final_classification']} "
            f"agree={r.get('label_agreement')} calib={r.get('calibrated_confidence')}"
        )

    copied = {"HEALTHY": 0, "UNHEALTHY": 0}
    missing: list[str] = []

    for _, r in accept.iterrows():
        src = crops_dir / r["crop_file"]
        if not src.exists():
            missing.append(str(src))
            continue
        cls = r["final_classification"]
        dst_dir = healthy_dir if cls == "HEALTHY" else unhealthy_dir
        dst_name = f"{prefix}{src.name}" if prefix else src.name
        dst = dst_dir / dst_name
        if dry_run:
            print(f"[dry-run] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        copied[cls] += 1

    print(
        f"\n{'Would copy' if dry_run else 'Copied'} "
        f"HEALTHY={copied['HEALTHY']} UNHEALTHY={copied['UNHEALTHY']} "
        f"(missing={len(missing)})"
    )
    if missing:
        for m in missing[:5]:
            print(f"  missing src: {m}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--crops-dir", type=Path, required=True,
                   help="Directory containing the source PNG crops referenced "
                        "by the CSV.")
    p.add_argument("--project-root", type=Path,
                   default=Path(__file__).resolve().parents[1],
                   help="Project root used to resolve incoming/ dirs.")
    p.add_argument("--healthy-dir", type=Path, default=HEALTHY_DIR_DEFAULT)
    p.add_argument("--unhealthy-dir", type=Path, default=UNHEALTHY_DIR_DEFAULT)
    p.add_argument("--min-agreement", type=float, default=0.67)
    p.add_argument("--min-calibrated", type=float, default=0.55)
    p.add_argument("--strict", action="store_true",
                   help="Use stricter thresholds: agreement=1.0, calibrated>=0.80.")
    p.add_argument("--prefix", type=str, default="pl_",
                   help="Filename prefix to mark pseudo-labels in incoming/.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.strict:
        args.min_agreement = 1.0
        args.min_calibrated = 0.80
    route(
        predictions_csv=args.predictions_csv,
        crops_dir=args.crops_dir,
        project_root=args.project_root,
        healthy_dir=args.healthy_dir,
        unhealthy_dir=args.unhealthy_dir,
        min_agreement=args.min_agreement,
        min_calibrated=args.min_calibrated,
        dry_run=args.dry_run,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
