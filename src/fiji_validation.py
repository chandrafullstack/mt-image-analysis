"""Fiji validation: import measurements and cross-validate."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def import_fiji_measurements(csv_path: Path) -> pd.DataFrame:
    """
    Import Fiji 'Analyze Particles' or ROI Manager measurements.
    """
    df = pd.read_csv(csv_path)
    rename_map = {
        "Area": "area_um2",
        "Perim.": "perimeter_um",
        "Circ.": "circularity",
        "AR": "aspect_ratio",
        "Round": "roundness",
        "Solidity": "solidity",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def compare_fiji_vs_python(fiji_df: pd.DataFrame,
                           python_df: pd.DataFrame,
                           tolerance: float = 0.05) -> pd.DataFrame:
    """
    Cross-validate Python-computed features against Fiji measurements.
    Flags discrepancies > tolerance (fractional).
    """
    common_cols = [c for c in fiji_df.columns if c in python_df.columns
                   and c not in ("label", "source_file")]
    comparison = pd.DataFrame()

    n = min(len(fiji_df), len(python_df))
    for col in common_cols:
        fiji_vals = fiji_df[col].iloc[:n].values.astype(float)
        py_vals = python_df[col].iloc[:n].values.astype(float)
        diff = np.abs(fiji_vals - py_vals) / (np.abs(py_vals) + 1e-8)
        comparison[f"{col}_pct_diff"] = diff
        comparison[f"{col}_ok"] = diff < tolerance

    ok_cols = [c for c in comparison.columns if c.endswith("_ok")]
    if ok_cols:
        comparison["all_pass"] = comparison[ok_cols].all(axis=1)

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate against Fiji measurements")
    parser.add_argument("--fiji", required=True, help="Fiji measurements CSV")
    parser.add_argument("--python", required=True, help="Python features CSV")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    fiji_df = import_fiji_measurements(Path(args.fiji))
    python_df = pd.read_csv(args.python)
    result = compare_fiji_vs_python(fiji_df, python_df, args.tolerance)

    pass_rate = result["all_pass"].mean() * 100 if "all_pass" in result else 0
    print(f"Validation pass rate: {pass_rate:.1f}%")
    print(result.describe())
