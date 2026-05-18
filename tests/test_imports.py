"""Smoke test: every src.* module imports without runtime errors.

Catches the class of bug where a function uses a name (e.g. `torch`) that's
imported by a sibling function but not by the enclosing one. Run with::

    python -m pytest tests/ -v
    # or without pytest installed:
    python tests/test_imports.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Modules that should always import on a base install.
CORE_MODULES = [
    "src.full_image_inference",
    "src.train_pseudo_labels",
    "src.claude_score_crops",
    "src.claude_classifier",
    "src.cnn_model",
    "src.researcher_cli",
    "src.segmentation",
    "src.preprocessing",
    "src.features",
    "src.reporting",
    "src.train_unet",
    "src.incoming_feedback",
    "src.route_pseudo_labels",
    "src.fission_fusion",
    "src.myelin_segmentation",
    "src.spatial_assignment",
    "src.gratio",
    "src.labeling",
    "src.fiji_validation",
    "src.data_download",
    "src.sft_dataset_builder",
]

# Optional modules whose deps are not in the base requirements.
OPTIONAL_MODULES = {
    "src.visualise": "matplotlib",
    "src.evaluate": "matplotlib",
    "src.ilastik_integration": "h5py",
}


def test_core_modules_import() -> None:
    failures: list[str] = []
    for name in CORE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Core module import failures:\n  " + "\n  ".join(failures)


def test_optional_modules_import_when_deps_present() -> None:
    skipped: list[str] = []
    failures: list[str] = []
    for name, dep in OPTIONAL_MODULES.items():
        try:
            importlib.import_module(dep)
        except ImportError:
            skipped.append(f"{name} (missing {dep})")
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if skipped:
        print(f"[info] Skipped optional modules: {skipped}")
    assert not failures, "Optional module import failures:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    test_core_modules_import()
    test_optional_modules_import_when_deps_present()
    print("OK")
