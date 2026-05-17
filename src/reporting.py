"""
Per-image summary reporting — generates the complete output table
with all 13 metrics required by the researcher.
"""
import pandas as pd
import json
from pathlib import Path
from src.spatial_assignment import generate_per_image_report, reports_to_csv


def generate_report_from_csvs(features_csv: Path,
                              fission_csv: Path,
                              assignment_csv: Path,
                              neuron_csv: Path,
                              output_csv: Path,
                              output_json: Path = None):
    """
    Generate the full per-image report from pre-computed CSVs.
    """
    mito_df = pd.read_csv(features_csv)
    ff_df = pd.read_csv(fission_csv) if fission_csv.exists() else pd.DataFrame()
    assign_df = pd.read_csv(assignment_csv) if assignment_csv.exists() else pd.DataFrame()
    neuron_df = pd.read_csv(neuron_csv) if neuron_csv.exists() else pd.DataFrame()

    # Group by source image
    reports = []
    if "source_file" in mito_df.columns:
        for img_name, group in mito_df.groupby("source_file"):
            report = generate_per_image_report(
                group, ff_df, assign_df, neuron_df, img_name
            )
            reports.append(report)
    else:
        report = generate_per_image_report(
            mito_df, ff_df, assign_df, neuron_df, "all_images"
        )
        reports.append(report)

    # Save flat CSV
    reports_to_csv(reports, output_csv)

    # Save full JSON with nested data
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"Saved full JSON report → {output_json}")

    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-image summary report")
    parser.add_argument("--features", required=True, help="Features CSV with labels")
    parser.add_argument("--fission", default="", help="Fission/fusion CSV")
    parser.add_argument("--assignment", default="", help="Mito-to-neuron assignment CSV")
    parser.add_argument("--neurons", default="", help="Neuron G-ratio CSV")
    parser.add_argument("--output", required=True, help="Output report CSV")
    parser.add_argument("--json", default="", help="Optional output JSON (full detail)")
    args = parser.parse_args()

    generate_report_from_csvs(
        Path(args.features),
        Path(args.fission) if args.fission else Path("nonexistent"),
        Path(args.assignment) if args.assignment else Path("nonexistent"),
        Path(args.neurons) if args.neurons else Path("nonexistent"),
        Path(args.output),
        Path(args.json) if args.json else None,
    )
