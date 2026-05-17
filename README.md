# Mitochondria Health Classification

Automated pipeline to classify healthy vs. unhealthy mitochondria in brain cell EM images, calculate G-ratios, and compare traditional deep learning with Claude Vision API.

## Quick Start

```bash
# 1. Set up environment
conda create -n mito python=3.10 -y && conda activate mito
pip install -r requirements.txt

# 2. Set Claude API key (for vision classification)
set ANTHROPIC_API_KEY=your-key-here

# 3. Download public datasets
python src/data_download.py

# 4. Preprocess
python src/preprocessing.py --input data/raw/epfl_rat/training.tif --output data/processed/

# 5. Segment mitochondria
python src/segmentation.py --input data/processed/ --output outputs/predictions/

# 6. Extract features + G-ratio
python src/features.py --masks outputs/predictions/ --output outputs/metrics/features.csv
python src/gratio.py --features outputs/metrics/features.csv --masks outputs/predictions/ --output outputs/metrics/features_with_gratio.csv

# 7. Label (rule-based → Ilastik refinement → merge)
python src/labeling.py --features outputs/metrics/features_with_gratio.csv --output data/labeled/rule_labels.csv

# 8. Train CNN or run Claude API classification
python src/cnn_model.py --data data/labeled/ --output outputs/models/
python src/claude_classifier.py --crops outputs/crops/ --output outputs/metrics/claude_results.csv

# 9. Evaluate
python src/evaluate.py --results outputs/metrics/ --output outputs/figures/

# 10. Launch interactive dashboard
uvicorn app.main:app --reload --port 8000
# → Open http://localhost:8000
```

## Researcher Workflow (Use Your Own Images)

This project supports a simple expert-feedback loop using folders:

- `incoming/healthy`
- `incoming/unhealthy`
- `outgoing/processed/healthy`
- `outgoing/processed/unhealthy`
- `outgoing/rejected`

How it works:

1. Researchers drop mitochondria-centered EM crops into `incoming/healthy` or `incoming/unhealthy`.
2. Open or refresh the dashboard page (`/api/summary` and `/api/gratio-data` trigger auto-ingestion).
3. New images are processed and appended into `outputs/metrics/features_with_gratio.csv`.
4. Crops are copied into `outputs/crops` for hover previews on the chart.
5. Processed files are moved to `outgoing/processed/...`.
6. Invalid or unreadable files are moved to `outgoing/rejected`.

Accepted file types:

- `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

Recommended input quality:

- Single mitochondrion near the center
- Grayscale EM crop
- Reasonable contrast between organelle and background
- Avoid montage sheets or multi-panel figures

Manual ingestion command (optional):

```bash
python src/incoming_feedback.py
```

Windows shortcut:

- Double-click `scripts/ingest_incoming.bat`

## Deploy On Another Researcher's Computer (Windows)

### Option A: Run from source (recommended)

1. Install Python 3.10+.
2. Open PowerShell in the project root.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. Open browser:

```text
http://127.0.0.1:8000
```

Windows shortcut:

- Double-click `scripts/start_dashboard.bat`

### Option B: Conda environment (lab machines)

```bash
conda env create -f environment.yml
conda activate mito
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Common startup issue

If you see `ModuleNotFoundError: No module named 'app'`, it usually means the command was launched from the wrong folder. Start from the project root (`mito_classifier`) and re-run the same command.

## Dashboard

The web dashboard displays an interactive Plotly scatter chart of G-ratios. **Hover over any data point** to see the actual EM crop image of that mitochondrion.

## Pipeline Steps

1. **Data Download** — EPFL & MitoEM public EM datasets
2. **Preprocessing** — Normalize, CLAHE, tile to 512×512
3. **Segmentation** — MitoNet (empanada) or U-Net
4. **Feature Extraction** — Morphology + G-ratio
5. **Labeling** — Rule-based + Ilastik + Fiji validation
6. **Classification** — CNN (ResNet-50) and Claude Vision API
7. **Evaluation** — Accuracy, F1, AUC, confusion matrix
8. **Dashboard** — FastAPI + Plotly with hover image popups

## Tools Used

- **empanada / MitoNet** — Pre-trained EM mitochondria segmentation
- **Ilastik** — Interactive object classification for label bootstrapping
- **Fiji (ImageJ)** — Manual validation and G-ratio spot-checks
- **Claude API** — Vision-language AI classification
- **FastAPI + Plotly** — Interactive web dashboard
