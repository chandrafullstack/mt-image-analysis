"""One-line import check across every src module."""
import importlib
import sys
import traceback
from pathlib import Path

# Make project root importable so `src.*` resolves no matter where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODS = [
    "full_image_inference", "train_pseudo_labels", "claude_score_crops",
    "claude_classifier", "cnn_model", "researcher_cli", "segmentation",
    "preprocessing", "features", "reporting", "visualise", "evaluate",
    "train_unet", "incoming_feedback", "route_pseudo_labels", "fission_fusion",
    "myelin_segmentation", "spatial_assignment", "gratio", "labeling",
    "ilastik_integration", "fiji_validation", "data_download",
    "sft_dataset_builder",
]

ok = 0
for m in MODS:
    try:
        importlib.import_module(f"src.{m}")
        print(f"  OK   src.{m}")
        ok += 1
    except Exception as exc:
        print(f"  FAIL src.{m}: {type(exc).__name__}: {exc}")
print(f"\n{ok}/{len(MODS)} modules import cleanly.")
