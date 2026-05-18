# Mitochondria Health Classifier

End-to-end pipeline for finding mitochondria in electron-microscopy (EM)
images and scoring each one as **healthy** or **unhealthy**.

Built around:

- A **U-Net** segmenter that locates individual mitochondria in a full image.
- A **ResNet-50** classifier that scores each detected mitochondrion.
- A **Claude vision** labelling loop (3-pass self-consensus) that builds the
  training set without large-scale manual annotation.
- A **FastAPI dashboard** for browsing per-image results, crops, and metrics.

---

## What's in the box

```
mito_classifier/
├── src/                     # All pipeline code (segmentation, training, inference, labelling)
├── app/                     # FastAPI dashboard (routes, templates, static assets)
├── scripts/                 # CLI wrappers and helper jobs (.bat + .py)
├── tests/                   # Smoke tests (module imports, pricing table sanity)
├── configs/                 # YAML configs
├── notebooks/               # Exploratory notebooks (optional)
├── data/                    # Inputs (gitignored — supply your own)
├── outputs/                 # Models, predictions, crops (gitignored)
├── requirements.txt
├── environment.yml
└── README.md
```

Data folders and `outputs/` are intentionally **not** tracked — every
artefact in them is reproducible from the scripts here.

---

## Quick start

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Or use `conda env create -f environment.yml`.)

### 2. Set your Claude API key

Copy `.env.example` to `.env` and paste your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it directly:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

The key is only required for the **labelling** step (during training).
Trained models run with no external calls.

### 3. Run inference on your own images

Drop EM images (PNG / JPG / TIFF) into a folder, then:

```powershell
python -m src.full_image_inference `
  --input-dir "C:\path\to\images" `
  --metrics-out outputs/metrics/my_metrics.csv `
  --crops-out outputs/crops/my_crops `
  --seg-method unet `
  --unet-weights outputs/models/unet_best.pt `
  --classifier-weights outputs/models/resnet50_best.pt
```

Pixel size override (e.g. confocal stacks):

```powershell
... --pixel-size-um 0.02
```

### 4. Launch the dashboard

```powershell
scripts\start_dashboard.bat
# then open http://localhost:8000
```

---

## Training pipeline

The classifier is trained on Claude-generated pseudo-labels. The loop:

1. Extract crops from a labelled segmentation pass.
2. Send a sample to Claude with a 3-pass self-consensus protocol
   ([src/claude_score_crops.py](src/claude_score_crops.py)).
3. Keep only labels with ≥0.67 vote agreement and ≥0.55 calibrated
   confidence.
4. Train ResNet-50 with a **group-aware** train/val/test split
   (different source images in each split) so the test metric is
   honest ([src/train_pseudo_labels.py](src/train_pseudo_labels.py)).
5. Sweep classification thresholds against val
   ([scripts/threshold_sweep.py](scripts/threshold_sweep.py)).
6. Retrain on hard negatives from the deployed model
   ([scripts/round2_hardneg_select.py](scripts/round2_hardneg_select.py)).

Example training command:

```powershell
python -m src.train_pseudo_labels `
  --consensus-csv outputs/predictions/round0_200nm_consensus.csv `
  --consensus-csv outputs/predictions/round1_consensus.csv `
  --metrics-csv outputs/metrics/round0_200nm.csv `
  --crops-dir outputs/crops/round0_200nm `
  --output-dir outputs/models_v2 `
  --epochs 25 --batch-size 16
```

`--consensus-csv` is repeatable — passing multiple CSVs auto-merges
and de-duplicates by `crop_file`.

---

## Cost control

The Claude scorer enforces a **hard USD cap** on every run via
`--budget-usd`. It refuses to make a call that would push the running
total past the cap. Pricing for supported models is hard-coded in
[src/claude_score_crops.py](src/claude_score_crops.py); a missing
model causes an immediate abort rather than surprise spend.

---

## Tests

```powershell
python -m pytest tests/ -v
# or:
python tests/test_imports.py
python scripts/import_check.py
```

CI runs the same tests on every push — see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Data sources

Public datasets the included scripts can pull / consume:

- **EPFL Rat Hippocampus** (electron microscopy of CA1 region)
- **MitoEM** (rat + human cortex, instance-segmented mitochondria)

See [src/data_download.py](src/data_download.py).

---

## Notes for researchers

- Drop your own labelled images into `incoming/healthy/` and
  `incoming/unhealthy/`; the dashboard auto-ingests them.
- Outputs are written under `outputs/`; nothing is sent to any
  external service except the Claude labelling step (which only fires
  when you explicitly run [src/claude_score_crops.py](src/claude_score_crops.py)).
- See [RESEARCHER_STEP_BY_STEP.md](RESEARCHER_STEP_BY_STEP.md) (kept
  locally, not tracked) for the long-form walk-through.

---

## License

MIT — see [LICENSE](LICENSE).
