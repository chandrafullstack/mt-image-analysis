"""Claude Vision API classifier for mitochondria health."""
import argparse
import base64
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

SYSTEM_PROMPT = """You are an expert cell biologist specialising in electron microscopy
analysis of mitochondria. You will be shown cropped EM images of individual mitochondria.

Classify each mitochondrion as HEALTHY or UNHEALTHY based on these criteria:

HEALTHY indicators:
- Elongated, tubular or filamentous shape (aspect ratio > 2)
- Dense, well-organised cristae (inner membrane folds)
- Intact, clearly defined double membrane
- Form factor 0.6–1.0

UNHEALTHY indicators:
- Rounded, swollen, or fragmented shape
- Sparse, disorganised, or absent cristae
- Disrupted or absent double membrane
- Vacuolated or electron-lucent matrix

Respond ONLY with valid JSON in this exact format:
{
  "classification": "HEALTHY" or "UNHEALTHY",
  "confidence": 0.0 to 1.0,
  "aspect_ratio_estimate": float,
  "cristae_quality": "dense" or "sparse" or "absent",
  "membrane_integrity": "intact" or "disrupted",
  "reasoning": "one sentence explanation"
}"""


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image array to base64 PNG string for the API."""
    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def classify_single_mito(image_crop: np.ndarray,
                         model: str = "claude-sonnet-4-20250514") -> dict:
    """Send a single mitochondrion crop to Claude for classification."""
    img_b64 = image_to_base64(image_crop)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Classify this mitochondrion. Respond only with the JSON format specified.",
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"classification": "ERROR", "raw_response": raw}


def classify_batch(crops_dir: Path, output_csv: Path,
                   model: str = "claude-sonnet-4-20250514"):
    """Classify all PNG crops in a directory."""
    crops_dir = Path(crops_dir)
    results = []

    crop_files = sorted(crops_dir.glob("*.png"))
    for crop_path in tqdm(crop_files, desc="Claude API classification"):
        img = np.array(Image.open(crop_path).convert("L")).astype(np.float32) / 255.0
        result = classify_single_mito(img, model)
        result["instance_id"] = crop_path.stem
        result["source_file"] = crop_path.name
        results.append(result)

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Claude classified {len(df)} instances → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude Vision API classifier")
    parser.add_argument("--crops", required=True, help="Directory of crop PNG images")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    classify_batch(Path(args.crops), Path(args.output), args.model)
