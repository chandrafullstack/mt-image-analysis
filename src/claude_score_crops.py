"""Cost-capped Claude vision scorer for mitochondria crops.

Reads PNG crops from a directory, samples N of them, and asks Claude to classify
each one as HEALTHY or UNHEALTHY using the criteria from
'Research Needs - Nanna.pptx' (slides 3-9). Writes per-crop predictions + a
running cost log to a CSV.

HARD COST CAP: stops *before* making any call that would push the running
estimated USD spend past --budget-usd (default $100). Cost is estimated from
the SDK-reported input/output tokens and the model's published price.

Usage::

    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python -m src.claude_score_crops \
        --crops-dir outputs/crops/round0_200nm \
        --output-csv outputs/predictions/round0_200nm_claude.csv \
        --sample 100 \
        --model claude-sonnet-4-5-20250929 \
        --budget-usd 100

Pricing table is in USD per *million* tokens. Update PRICES if you switch models.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import anthropic


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens). Source: anthropic.com/pricing as of May 2026.
# If a model isn't listed here, the script refuses to run — you must explicitly
# add it to avoid surprise spend.
# ---------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    # model_id : (input_usd_per_M, output_usd_per_M)
    # Opus tier
    "claude-opus-4-7":              (15.0, 75.0),
    "claude-opus-4-6":              (15.0, 75.0),
    "claude-opus-4-5":              (15.0, 75.0),
    "claude-opus-4-5-20251101":     (15.0, 75.0),
    "claude-opus-4-5-20250929":     (15.0, 75.0),
    "claude-opus-4-1-20250805":     (15.0, 75.0),
    "claude-opus-4-20250514":       (15.0, 75.0),
    # Sonnet tier
    "claude-sonnet-4-6":             (3.0, 15.0),
    "claude-sonnet-4-5":             (3.0, 15.0),
    "claude-sonnet-4-5-20250929":    (3.0, 15.0),
    "claude-sonnet-4-20250514":      (3.0, 15.0),
    # Haiku tier
    "claude-haiku-4-5":              (1.0,  5.0),
    "claude-haiku-4-5-20251001":     (1.0,  5.0),
}


# ---------------------------------------------------------------------------
# PPT-aligned prompt. Mirrors slides 3, 4, 5, 6 (mitochondria health + state).
# Myelin and g-ratio are out of scope for *crop-level* classification.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert electron-microscopy (EM) cell biologist
classifying single-mitochondrion crops as HEALTHY or UNHEALTHY.

Use ONLY these criteria (from the researcher's reference slides):

HEALTHY indicators:
- Inner cristae (folded inner-membrane lines) are clearly VISIBLE inside the
  organelle.
- Double membrane is intact and well defined.
- Matrix density looks uniform.
- A mitochondrion that is fissioning (pinched in the middle) or fusing
  (joined with another) STILL counts as HEALTHY if cristae are visible.

UNHEALTHY indicators:
- Inner cristae are ABSENT, faint, or disorganized.
- Membrane is disrupted, broken, or missing.
- Matrix is swollen, vacuolated, or electron-lucent (washed-out).

If the crop is not clearly a mitochondrion (background, nucleus, vesicle,
artifact, myelin sheath, partial cell), set classification to "NOT_MITO".

Respond ONLY with valid JSON in this exact schema:
{
  "classification": "HEALTHY" | "UNHEALTHY" | "NOT_MITO",
  "confidence": 0.0-1.0,
  "cristae_visibility": "clear" | "faint" | "absent" | "n_a",
  "membrane_integrity": "intact" | "disrupted" | "n_a",
  "fission_or_fusion": "fission" | "fusion" | "none" | "n_a",
  "reasoning": "one short sentence"
}
""".strip()


USER_TEXT = (
    "Classify this single-mitochondrion EM crop. "
    "Return only the JSON specified."
)


# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    budget_usd: float
    input_per_m: float
    output_per_m: float
    in_tokens: int = 0
    out_tokens: int = 0
    spent_usd: float = 0.0

    def add(self, in_t: int, out_t: int) -> None:
        self.in_tokens += in_t
        self.out_tokens += out_t
        self.spent_usd = (
            self.in_tokens / 1_000_000 * self.input_per_m
            + self.out_tokens / 1_000_000 * self.output_per_m
        )

    def would_exceed(self, projected_call_usd: float = 0.05) -> bool:
        # Block any call that *might* push us past the cap.
        return self.spent_usd + projected_call_usd > self.budget_usd

    def summary(self) -> str:
        return (
            f"tokens in/out = {self.in_tokens}/{self.out_tokens} | "
            f"spent ~${self.spent_usd:.4f} of ${self.budget_usd:.2f} cap"
        )


def _image_to_b64(path: Path) -> tuple[str, str]:
    img = Image.open(path).convert("L")
    # Re-encode as PNG; downsize if huge to keep token cost predictable.
    max_side = 768
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/png"


def _classify_one(
    client: anthropic.Anthropic,
    model: str,
    image_path: Path,
    temperature: float = 1.0,
) -> tuple[dict, int, int]:
    b64, media_type = _image_to_b64(image_path)
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": USER_TEXT},
                ],
            }
        ],
    )
    raw = resp.content[0].text.strip()
    # Tolerate ```json fences.
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip("` \n")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"classification": "PARSE_ERROR", "raw_response": raw}
    return parsed, int(resp.usage.input_tokens), int(resp.usage.output_tokens)


def _consensus_classify(
    client: anthropic.Anthropic,
    model: str,
    image_path: Path,
    passes: int,
) -> tuple[dict, int, int]:
    """Run the classifier N independent times and aggregate.

    Calibrated confidence = (mean self-confidence among majority-label passes)
                            * (label_agreement).
    This penalises both unstable answers (low agreement) and tentative
    answers (low self-confidence) without trusting Claude's number alone.
    """
    from collections import Counter

    pass_labels: list[str] = []
    pass_confs: list[float] = []
    pass_details: list[dict] = []
    in_t_total = 0
    out_t_total = 0

    # Use temperature=1.0 for diversity across passes (default for Anthropic).
    # First pass at slightly lower temperature anchors the distribution.
    temps = [0.4] + [1.0] * (passes - 1) if passes > 1 else [0.0]

    for i in range(passes):
        parsed, in_t, out_t = _classify_one(
            client, model, image_path, temperature=temps[i]
        )
        in_t_total += in_t
        out_t_total += out_t
        label = str(parsed.get("classification", "PARSE_ERROR")).upper()
        try:
            conf = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        pass_labels.append(label)
        pass_confs.append(conf)
        pass_details.append(parsed)

    label_counts = Counter(pass_labels)
    final_label, majority_count = label_counts.most_common(1)[0]
    agreement = majority_count / passes
    majority_confs = [
        c for lbl, c in zip(pass_labels, pass_confs) if lbl == final_label
    ]
    mean_majority_conf = (
        sum(majority_confs) / len(majority_confs) if majority_confs else 0.0
    )
    calibrated = round(mean_majority_conf * agreement, 4)

    # Pick the highest-self-confidence majority pass to source per-criterion
    # fields and reasoning from (avoids averaging categorical strings).
    best_idx = max(
        (i for i, lbl in enumerate(pass_labels) if lbl == final_label),
        key=lambda i: pass_confs[i],
        default=0,
    )
    best = pass_details[best_idx]

    aggregated = {
        "final_classification": final_label,
        "label_agreement": round(agreement, 4),
        "mean_majority_confidence": round(mean_majority_conf, 4),
        "calibrated_confidence": calibrated,
        "n_passes": passes,
        "pass_labels": "|".join(pass_labels),
        "pass_confidences": "|".join(f"{c:.2f}" for c in pass_confs),
        "cristae_visibility": best.get("cristae_visibility"),
        "membrane_integrity": best.get("membrane_integrity"),
        "fission_or_fusion": best.get("fission_or_fusion"),
        "reasoning": best.get("reasoning"),
        # Auto-flag rows that should be reviewed before training.
        "review_flag": int(
            final_label not in {"HEALTHY", "UNHEALTHY"}
            or agreement < 0.67
            or calibrated < 0.55
        ),
    }
    return aggregated, in_t_total, out_t_total


def score_crops(
    crops_dir: Path,
    output_csv: Path,
    sample: int,
    model: str,
    budget_usd: float,
    seed: int,
    passes: int = 1,
    exclude_csv: Path | None = None,
) -> None:
    if model not in PRICES:
        raise SystemExit(
            f"Model {model!r} is not in the local pricing table. Add it to "
            f"PRICES in this file (with current $/Mtok) before running."
        )
    # Optionally hydrate ANTHROPIC_API_KEY from .env if python-dotenv is installed.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except ImportError:
            pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY env var is not set. Export it (or put it in a "
            ".env file with python-dotenv installed) and re-run."
        )
    if passes < 1:
        raise SystemExit("--passes must be >= 1")

    all_crops = sorted(crops_dir.glob("*.png"))
    if not all_crops:
        raise SystemExit(f"No PNG crops found in {crops_dir}")

    excluded: set[str] = set()
    if exclude_csv is not None:
        for path in (exclude_csv if isinstance(exclude_csv, list) else [exclude_csv]):
            if not Path(path).exists():
                print(f"[warn] exclude-csv not found: {path}")
                continue
            try:
                import csv as _csv
                with open(path, newline="", encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        name = row.get("crop_file")
                        if name:
                            excluded.add(name)
            except Exception as exc:
                print(f"[warn] failed to load exclude-csv {path}: {exc}")
        before = len(all_crops)
        all_crops = [p for p in all_crops if p.name not in excluded]
        print(f"Excluded {before - len(all_crops)} crops already labelled in prior CSVs.")

    rng = random.Random(seed)
    sample_n = min(sample, len(all_crops))
    chosen = rng.sample(all_crops, sample_n)
    print(
        f"Found {len(all_crops)} eligible crops; sampling {sample_n} (seed={seed}); "
        f"passes-per-crop={passes}."
    )

    in_per_m, out_per_m = PRICES[model]
    tracker = CostTracker(
        budget_usd=budget_usd, input_per_m=in_per_m, output_per_m=out_per_m
    )
    client = anthropic.Anthropic()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Write incrementally so a KeyboardInterrupt / crash never wastes spend.
    if passes == 1:
        fieldnames = [
            "crop_file", "classification", "confidence",
            "cristae_visibility", "membrane_integrity", "fission_or_fusion",
            "reasoning", "input_tokens", "output_tokens",
            "running_cost_usd", "model",
        ]
    else:
        fieldnames = [
            "crop_file", "final_classification",
            "label_agreement", "mean_majority_confidence",
            "calibrated_confidence", "review_flag",
            "n_passes", "pass_labels", "pass_confidences",
            "cristae_visibility", "membrane_integrity", "fission_or_fusion",
            "reasoning", "input_tokens", "output_tokens",
            "running_cost_usd", "model",
        ]
    import csv as _csv
    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    writer = _csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    # Guard: per-call budget projection scales with passes.
    per_call_guard = 0.05 * passes

    try:
      for crop_path in tqdm(chosen, desc=f"Claude {model} x{passes}"):
        if tracker.would_exceed(projected_call_usd=per_call_guard):
            print(
                f"\nSTOP: would exceed ${budget_usd:.2f} budget. "
                f"{tracker.summary()}",
                file=sys.stderr,
            )
            break

        try:
            if passes == 1:
                parsed, in_t, out_t = _classify_one(client, model, crop_path)
                tracker.add(in_t, out_t)
                row = {
                    "crop_file": crop_path.name,
                    "classification": parsed.get("classification"),
                    "confidence": parsed.get("confidence"),
                    "cristae_visibility": parsed.get("cristae_visibility"),
                    "membrane_integrity": parsed.get("membrane_integrity"),
                    "fission_or_fusion": parsed.get("fission_or_fusion"),
                    "reasoning": parsed.get("reasoning"),
                    "input_tokens": in_t,
                    "output_tokens": out_t,
                    "running_cost_usd": round(tracker.spent_usd, 6),
                    "model": model,
                }
            else:
                agg, in_t, out_t = _consensus_classify(
                    client, model, crop_path, passes
                )
                tracker.add(in_t, out_t)
                row = {
                    "crop_file": crop_path.name,
                    **agg,
                    "input_tokens": in_t,
                    "output_tokens": out_t,
                    "running_cost_usd": round(tracker.spent_usd, 6),
                    "model": model,
                }
        except anthropic.APIError as exc:
            row = {
                "crop_file": crop_path.name,
                "final_classification" if passes > 1 else "classification":
                    "API_ERROR",
                "reasoning": str(exc)[:300],
                "model": model,
            }

        rows.append(row)
        writer.writerow(row)
        csv_file.flush()
        # Polite pacing; Anthropic rate limits are generous but not infinite.
        time.sleep(0.2)
    finally:
        csv_file.close()

    print(f"\nWrote {len(rows)} rows -> {output_csv}")
    print(tracker.summary())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--crops-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--sample", type=int, default=100,
                   help="Random sample size; capped to available crops.")
    p.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    p.add_argument("--budget-usd", type=float, default=100.0,
                   help="Hard cap. Script aborts before the next call if "
                        "running total would exceed this.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--passes", type=int, default=1,
                   help="Independent classification passes per crop. "
                        ">1 enables consensus + calibrated confidence "
                        "(recommended: 3). Cost scales linearly.")
    p.add_argument("--exclude-csv", type=Path, action="append", default=None,
                   help="Path to a previous predictions CSV; crops listed in "
                        "its `crop_file` column will be excluded from sampling. "
                        "Repeatable.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    score_crops(
        crops_dir=args.crops_dir,
        output_csv=args.output_csv,
        sample=args.sample,
        model=args.model,
        budget_usd=args.budget_usd,
        seed=args.seed,
        passes=args.passes,
        exclude_csv=args.exclude_csv,
    )


if __name__ == "__main__":
    main()
