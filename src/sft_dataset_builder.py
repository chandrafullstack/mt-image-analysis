"""Build a JSONL dataset for QLoRA SFT of Qwen2.5-VL from local labeled EM images.

Converts researcher-labeled mitochondria images in:

    incoming/healthy/      -> label = HEALTHY
    incoming/unhealthy/    -> label = UNHEALTHY

(or any --healthy-dir / --unhealthy-dir you point at) into a single JSONL file
suitable for parameter-efficient fine-tuning of Qwen2.5-VL, following the SFT
spec described in PROJECT_CHECKPOINT_2026-05-17.md ("Step 2: Define SFT dataset
format" and "Step 3: Data splits").

Output record schema (one JSON object per line)::

    {
        "image_path": "<absolute path to image>",
        "label":      "HEALTHY" | "UNHEALTHY",
        "prompt":     "<fixed instruction prompt>",
        "response":   "HEALTHY. <short rationale>" | "UNHEALTHY. <short rationale>",
        "split":      "train" | "val" | "test",
        "messages":   [  # Qwen2.5-VL chat format, ready for HF TRL / Qwen trainers
            {"role": "system", "content": [{"type": "text", "text": "..."}]},
            {"role": "user",   "content": [
                {"type": "image", "image": "<image_path>"},
                {"type": "text",  "text": "<prompt>"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<response>"}]}
        ]
    }

Splits are stratified by label, 70 / 15 / 15 by default, with a fixed seed for
reproducibility.

Example::

    python -m src.sft_dataset_builder \
        --healthy-dir incoming/healthy \
        --unhealthy-dir incoming/unhealthy \
        --output data/sft/qwen25vl_mito_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep aligned with src/incoming_feedback.py
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_HEALTHY_DIR = Path("incoming/healthy")
DEFAULT_UNHEALTHY_DIR = Path("incoming/unhealthy")
DEFAULT_OUTPUT = Path("data/sft/qwen25vl_mito_sft.jsonl")

# Fixed prompts -> stable supervision signal (see checkpoint Step 2).
SYSTEM_PROMPT = (
    "You are a careful electron-microscopy assistant that classifies a single "
    "mitochondrion crop as HEALTHY or UNHEALTHY based on morphology "
    "(membrane integrity, cristae structure, shape, and density)."
)

USER_PROMPT = (
    "Classify this mitochondrion crop as HEALTHY or UNHEALTHY. "
    "Answer with the single label on the first line, followed by one short "
    "sentence giving the main morphological reason."
)

# Conservative default rationales. Researchers can edit the JSONL afterwards
# if they want richer per-image rationales; the format stays the same.
DEFAULT_RATIONALE = {
    "HEALTHY":   "Intact outer membrane with regular cristae and uniform matrix density.",
    "UNHEALTHY": "Disrupted membrane or swollen matrix with disorganized or lost cristae.",
}


@dataclass
class BuildStats:
    healthy: int
    unhealthy: int
    train: int
    val: int
    test: int
    output_path: Path

    def summary(self) -> str:
        total = self.healthy + self.unhealthy
        return (
            f"Wrote {total} records to {self.output_path}\n"
            f"  by label : HEALTHY={self.healthy}  UNHEALTHY={self.unhealthy}\n"
            f"  by split : train={self.train}  val={self.val}  test={self.test}"
        )


def _iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def _stratified_split(
    items: list[Path],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    """Shuffle and split a single-label list into train/val/test."""
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Guarantee at least one sample in val/test when possible.
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def _build_record(image_path: Path, label: str, split: str) -> dict:
    rationale = DEFAULT_RATIONALE[label]
    response = f"{label}. {rationale}"
    abs_path = str(image_path.resolve())
    return {
        "image_path": abs_path,
        "label":      label,
        "prompt":     USER_PROMPT,
        "response":   response,
        "split":      split,
        "messages": [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [
                 {"type": "image", "image": abs_path},
                 {"type": "text",  "text": USER_PROMPT},
             ]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response}]},
        ],
    }


def build_sft_jsonl(
    healthy_dir: Path = DEFAULT_HEALTHY_DIR,
    unhealthy_dir: Path = DEFAULT_UNHEALTHY_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 1337,
) -> BuildStats:
    """Build the QLoRA SFT JSONL file from local labeled folders."""
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 (rest goes to test).")

    healthy_imgs = _iter_images(healthy_dir)
    unhealthy_imgs = _iter_images(unhealthy_dir)

    if not healthy_imgs and not unhealthy_imgs:
        raise FileNotFoundError(
            f"No supported images found under {healthy_dir} or {unhealthy_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )

    rng = random.Random(seed)
    healthy_split = _stratified_split(healthy_imgs, train_frac, val_frac, rng)
    unhealthy_split = _stratified_split(unhealthy_imgs, train_frac, val_frac, rng)

    records: list[dict] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        for img in healthy_split[split]:
            records.append(_build_record(img, "HEALTHY", split))
            counts[split] += 1
        for img in unhealthy_split[split]:
            records.append(_build_record(img, "UNHEALTHY", split))
            counts[split] += 1

    # Shuffle final ordering so a single label doesn't cluster.
    rng.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return BuildStats(
        healthy=len(healthy_imgs),
        unhealthy=len(unhealthy_imgs),
        train=counts["train"],
        val=counts["val"],
        test=counts["test"],
        output_path=output_path,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qwen2.5-VL QLoRA SFT JSONL from incoming/healthy and "
            "incoming/unhealthy (see PROJECT_CHECKPOINT_2026-05-17.md)."
        )
    )
    parser.add_argument("--healthy-dir",   type=Path, default=DEFAULT_HEALTHY_DIR)
    parser.add_argument("--unhealthy-dir", type=Path, default=DEFAULT_UNHEALTHY_DIR)
    parser.add_argument("--output",        type=Path, default=DEFAULT_OUTPUT,
                        help="Path to output .jsonl file")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.15,
                        help="Test fraction = 1 - train_frac - val_frac")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = _parse_args()
    stats = build_sft_jsonl(
        healthy_dir=args.healthy_dir,
        unhealthy_dir=args.unhealthy_dir,
        output_path=args.output,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(stats.summary())


if __name__ == "__main__":
    main()
