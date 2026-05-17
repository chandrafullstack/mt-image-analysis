"""Rule-based labeling and label merging for mitochondria health classification."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def assign_rule_based_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign preliminary healthy/unhealthy labels based on literature thresholds.
    """
    df = features_df.copy()
    df["unhealthy_score"] = 0

    # Aspect ratio: healthy > 2.0, unhealthy ≈ 1.0
    df.loc[df["aspect_ratio"] < 1.5, "unhealthy_score"] += 1

    # Form factor: healthy 0.6–1.0, unhealthy < 0.4 or > 1.2
    df.loc[(df["form_factor"] < 0.4) | (df["form_factor"] > 1.2), "unhealthy_score"] += 1

    # Roundness: unhealthy → high roundness (spherical)
    df.loc[df["roundness"] > 0.7, "unhealthy_score"] += 1

    # G-ratio: healthy 0.6–0.8
    if "g_ratio" in df.columns:
        df.loc[(df["g_ratio"] < 0.5) | (df["g_ratio"] > 0.85), "unhealthy_score"] += 1

    # Threshold: ≥ 2 unhealthy indicators → UNHEALTHY
    df["label_rule_based"] = np.where(df["unhealthy_score"] >= 2, "UNHEALTHY", "HEALTHY")
    df["label_confidence"] = np.where(
        df["unhealthy_score"].isin([0, 3, 4]), "high", "low"
    )

    return df


def merge_labels(rule_df: pd.DataFrame,
                 ilastik_labels: dict = None,
                 fiji_overrides: dict = None) -> pd.DataFrame:
    """
    Merge labels from all sources with priority:
    Fiji manual override > Ilastik classification > rule-based threshold.
    """
    df = rule_df.copy()

    # Start with rule-based as default
    df["label_final"] = df["label_rule_based"]

    # Override with Ilastik where available
    if ilastik_labels:
        df["label_ilastik"] = df["label"].map(
            lambda lbl: ilastik_labels.get(int(lbl), None)
        )
        mask = df["label_ilastik"].notna()
        df.loc[mask, "label_final"] = df.loc[mask, "label_ilastik"]

    # Override with Fiji expert corrections (highest priority)
    if fiji_overrides:
        for instance_id, override_label in fiji_overrides.items():
            df.loc[df["label"] == instance_id, "label_final"] = override_label

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label mitochondria")
    parser.add_argument("--features", help="Features CSV with G-ratio")
    parser.add_argument("--output", required=True, help="Output labeled CSV")
    parser.add_argument("--merge", action="store_true", help="Merge labels mode")
    parser.add_argument("--rule", help="Rule-based labels CSV (for merge)")
    parser.add_argument("--ilastik", help="Ilastik labels CSV (for merge)")
    args = parser.parse_args()

    if args.merge:
        rule_df = pd.read_csv(args.rule)
        ilastik_labels = {}
        if args.ilastik and Path(args.ilastik).exists():
            il_df = pd.read_csv(args.ilastik)
            ilastik_labels = dict(zip(il_df["instance_id"], il_df["label"]))
        merged = merge_labels(rule_df, ilastik_labels)
        merged.to_csv(args.output, index=False)
        print(f"Merged labels → {args.output}")
    else:
        df = pd.read_csv(args.features)
        labeled = assign_rule_based_labels(df)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(args.output, index=False)
        print(f"Rule-based labels assigned → {args.output}")
