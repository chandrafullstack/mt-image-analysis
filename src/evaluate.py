"""Evaluation metrics and benchmarking."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_metrics(y_true: list, y_pred: list, y_prob: list = None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Unhealthy"],
                yticklabels=["Healthy", "Unhealthy"], ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def benchmark_methods(results: dict) -> pd.DataFrame:
    """
    Compare multiple methods.
    results: {"CNN": {"accuracy": 0.9, ...}, "Claude": {...}, "Rule-based": {...}}
    """
    return pd.DataFrame(results).T.round(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", required=True, help="Output figures directory")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth and predictions
    gt_path = results_dir / "features_with_gratio.csv"
    if gt_path.exists():
        df = pd.read_csv(gt_path)
        if "label_final" in df.columns:
            y_true = (df["label_final"] == "UNHEALTHY").astype(int).tolist()

            # Evaluate rule-based
            if "label_rule_based" in df.columns:
                y_pred_rule = (df["label_rule_based"] == "UNHEALTHY").astype(int).tolist()
                rule_metrics = compute_metrics(y_true, y_pred_rule)
                print("Rule-based:", rule_metrics)
                plot_confusion_matrix(
                    y_true, y_pred_rule,
                    str(output_dir / "confusion_rule_based.png")
                )

    print(f"Evaluation complete → {output_dir}")
