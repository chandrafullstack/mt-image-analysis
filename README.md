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

## Is Deployment Self-Sufficient?

Yes for local dashboard inference and rule-based classification.

- No cloud service is required for the dashboard path.
- Researchers can run everything on their own Windows machine after installing dependencies.
- Claude API is optional and only used if you explicitly run `src/claude_classifier.py`.

## Dependency Breakdown

Core runtime (required for researcher workflow):

- Python 3.10+
- numpy, scipy, pandas
- scikit-image, Pillow, tifffile
- fastapi, uvicorn, jinja2

Optional model training/inference stack:

- torch, torchvision, segmentation-models-pytorch, albumentations

Optional Claude comparison:

- anthropic package
- `ANTHROPIC_API_KEY` environment variable

## Do Researchers Need a Claude Subscription?

No, not for the default workflow.

- Dashboard ingestion and visualization do not call Claude.
- Claude is only for optional experiment/comparison runs in `src/claude_classifier.py`.

## Classification Engine (Current State)

There are two classification paths:

1. Operational path (used by dashboard today)
	- Feature extraction from each crop (area, perimeter, aspect ratio, form factor, roundness, eccentricity, solidity, approximate g-ratio)
	- Label source for researcher-uploaded data is the folder choice (`incoming/healthy` or `incoming/unhealthy`)
	- Additional tags computed: shape category and fission/fusion heuristic state

2. Research path (optional)
	- CNN module scaffold in `src/cnn_model.py` (ResNet-50 + U-Net utilities)
	- Rule-based label assignment and merge logic in `src/labeling.py`
	- Claude Vision API comparison in `src/claude_classifier.py`

Note: `src/cnn_model.py` currently contains a training skeleton (not a fully wired crop loader/training experiment pipeline yet).

## How Training Is Done

Current practical training signal:

- Expert labels from incoming folders become supervised records in `outputs/metrics/features_with_gratio.csv`.
- These records can be exported to build a train/val split for CNN experiments.

Current repo status for deep training:

- Rule-based + expert override logic is implemented.
- Full production training loop for ResNet/U-Net is partially scaffolded and requires finishing dataset loaders and experiment scripts.

## Preprocessing Steps

For stack-based pipeline (`src/preprocessing.py`):

1. Load TIFF (single image or stack)
2. Normalize intensities to [0, 1]
3. Apply CLAHE contrast enhancement
4. Tile into overlapping 512x512 patches (default overlap 64)
5. Save tiles

For researcher incoming crops (`src/incoming_feedback.py`):

1. Load grayscale crop
2. Otsu thresholding (fallback to mean threshold)
3. Remove tiny objects/holes
4. Keep largest connected component as mitochondrion candidate
5. Compute morphology metrics + approximate g-ratio via erosion shell
6. Save crop to dashboard gallery and append row to metrics CSV

## Assumptions

- Input images are EM grayscale crops, ideally one mitochondrion per image.
- Uploaded healthy/unhealthy folders are trusted expert labels.
- Pixel size defaults to 0.008 um/pixel in feature conversions.
- Fission/fusion state is heuristic (solidity/eccentricity based), not temporal tracking.
- Myelin context in this flow defaults to `UNASSIGNED` unless separate myelin segmentation outputs are provided.

## Clear Data-to-Result Flow

1. Researchers provide labeled crops:
	- `incoming/healthy`
	- `incoming/unhealthy`
2. App ingests new files (auto on dashboard API call or manual run)
3. Features are extracted and appended to:
	- `outputs/metrics/features_with_gratio.csv`
4. Crop images are copied to:
	- `outputs/crops`
5. Original files are moved to:
	- `outgoing/processed/healthy`
	- `outgoing/processed/unhealthy`
	- `outgoing/rejected` (if unreadable)
6. Dashboard reads updated CSV and renders results immediately.

## Production-Ready Multi-Object Flow (Full EM Images)

For full EM images containing multiple cellular structures, the pipeline is now two-stage:

1. **Mitochondria detection/segmentation** on each full image
2. **Per-instance health classification** for each detected mitochondrion

Use this command:

```bash
python -m src.researcher_cli --full-image-dir "C:\\lab\\full_em_images" --seg-method heuristic --serve
```

Outputs are dashboard-ready:

- `outputs/metrics/features_with_gratio.csv`
- `outputs/crops/mito_XXXX.png`

For stronger production segmentation, use trained U-Net weights:

```bash
python -m src.researcher_cli --full-image-dir "C:\\lab\\full_em_images" --seg-method unet --unet-weights "outputs/models/unet_best.pt" --classifier-weights "outputs/models/resnet50_best.pt" --serve
```

Windows helper script for full-image flow:

- `scripts/run_full_image_flow.bat "C:\path\full_images"`

## Production Training (No Longer Skeleton)

`src/cnn_model.py` now supports actual training with:

- Real image loading from folders
- Stratified train/validation split
- Augmentation + normalization
- Best-model checkpointing
- Saved split manifest + training history

Expected training data structure:

```text
data/labeled/crops/
	HEALTHY/
	UNHEALTHY/
```

Train command:

```bash
python src/cnn_model.py --data data/labeled/crops --output outputs/models --epochs 25 --batch-size 16
```

Windows helper script for training:

- `scripts/train_classifier.bat "data\labeled\crops" "outputs\models"`

Artifacts:

- `outputs/models/resnet50_best.pt`
- `outputs/models/resnet50_last.pt`
- `outputs/models/training_history.csv`
- `outputs/models/dataset_split.csv`

## Point-To-Folders CLI (Researcher Friendly)

Researchers can point directly to their own folders (without manual copying):

```bash
python -m src.researcher_cli --healthy-dir "C:\\lab\\healthy" --unhealthy-dir "C:\\lab\\unhealthy" --serve
```

Or for full EM images with many structures:

```bash
python -m src.researcher_cli --full-image-dir "C:\\lab\\full_em_images" --serve
```

Windows helper script:

- `scripts/run_researcher_flow.bat "C:\path\healthy" "C:\path\unhealthy"`

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
