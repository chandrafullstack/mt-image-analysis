# Project Checkpoint - 2026-05-17

## Purpose of this document

This is a full handoff checkpoint so work can continue on a different laptop without losing context.

## Repository

- Remote: https://github.com/chandrafullstack/mt-image-analysis.git
- Branch: main
- Latest commit at checkpoint creation: 27a4ec3

## What has been built so far

### Core pipeline and dashboard

- End-to-end mitochondria pipeline scaffold created
- FastAPI dashboard added and connected to metrics outputs
- Plotly visualizations added with hover crop previews
- Dashboard APIs expanded for summary and per-instance views

### Data and feature flow

- Public EPFL data download and processing integrated
- Feature extraction includes morphology and g-ratio
- Rule-based healthy versus unhealthy logic integrated
- Shape categories and fission/fusion heuristics added

### Researcher-friendly operations

- Incoming and outgoing folder workflow created:
  - incoming/healthy
  - incoming/unhealthy
  - outgoing/processed/healthy
  - outgoing/processed/unhealthy
  - outgoing/rejected
- Automatic ingestion into dashboard metrics implemented
- Beginner-friendly researcher guide added
- One-command researcher CLI added

### Production upgrades completed

- Full-image inference pipeline implemented (multiple structures per EM image)
- Two-stage logic:
  1. Detect mitochondria in full image
  2. Classify each detected instance
- Classifier training upgraded from skeleton to real train/val workflow
- U-Net segmentation training script implemented for image/mask pairs
- Retrain option added to pipeline via CLI

## Key files added or updated

- src/researcher_cli.py
- src/full_image_inference.py
- src/cnn_model.py
- src/train_unet.py
- src/incoming_feedback.py
- scripts/run_full_image_flow.bat
- scripts/train_classifier.bat
- scripts/train_unet.bat
- scripts/retrain_and_run_full_image.bat
- RESEARCHER_STEP_BY_STEP.md
- README.md

## Current learning logic

### What learns

1. U-Net segmentation model learns from local image plus mask pairs
2. ResNet classifier learns from local healthy and unhealthy crops

### What does not learn automatically

- Incoming folder ingestion updates metrics and dashboard data
- Incoming workflow does not by itself retrain models unless retrain mode is called

## Retrain workflow currently available

Run retrain and inference in one command:

python -m src.researcher_cli --retrain --full-image-dir "C:\lab\em_full_images" --seg-method unet --serve

What this does:

1. Trains U-Net from:
   - data/labeled/segmentation/images
   - data/labeled/segmentation/masks
2. Trains ResNet from:
   - data/labeled/crops/HEALTHY
   - data/labeled/crops/UNHEALTHY
3. Uses newly trained weights if present:
   - outputs/models/unet_best.pt
   - outputs/models/resnet50_best.pt
4. Runs full-image inference and opens dashboard

## Clarification on Claude and learning

- Claude is optional in this project
- Claude is currently for optional classification/comparison runs
- Claude is not the automatic local retraining engine in the current pipeline

## Important operational note from prior runs

There were repeated uvicorn launch issues caused by running from the wrong folder or import context.

Stable pattern:

1. Open terminal in project root
2. Use:

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

If port is busy:

Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

## Local SFT plan for new laptop (detailed)

## Goal

Create an optional vision-language SFT classifier trained only on local researcher-labeled EM images.

## Scope decision

- Keep current U-Net plus ResNet as primary production baseline
- Build SFT track as secondary experimental model
- Promote SFT model only if it outperforms baseline on held-out data

## Is this local?

Yes. The plan is fully local:

1. Download open base model locally
2. Prepare local dataset from researcher images
3. Run QLoRA SFT on local GPU
4. Evaluate locally
5. Optionally export local inference endpoint or script

## Step 1: Check laptop GPU and choose model size

Suggested VRAM guidance:

- 8 GB: tiny VLM or aggressive quantization only
- 12 GB: ~3B VLM with QLoRA feasible
- 16 GB: 3B to 7B QLoRA feasible
- 24 GB+: 7B much more comfortable

Recommended first local model candidates:

- Qwen2.5-VL 3B instruct (good starter for local SFT)
- If memory allows, Qwen2.5-VL 7B instruct

## Step 2: Define SFT dataset format

Create JSONL records with fields like:

- image_path
- prompt
- response
- split

Example conceptual record:

- image_path: path to EM crop
- prompt: classify mitochondrion as HEALTHY or UNHEALTHY and provide one short reason
- response: HEALTHY or UNHEALTHY plus short rationale

Important:

- Use only high-confidence researcher labels
- Keep prompt template fixed to reduce noise
- Hold out test set before training

## Step 3: Data splits

Recommended split:

- Train: 70 percent
- Validation: 15 percent
- Test: 15 percent

Stratify by label and, if possible, by acquisition session or specimen to reduce leakage.

## Step 4: Training approach

- Use parameter-efficient fine-tuning (LoRA/QLoRA)
- Train only adapters, not full base model weights
- Mixed precision and gradient checkpointing enabled

Initial hyperparameter starting point:

- Epochs: 2 to 4
- Learning rate: 2e-5 to 1e-4 search
- Batch size: as memory allows with gradient accumulation
- Max image resolution: tune to fit memory

## Step 5: Evaluation protocol

Evaluate on held-out test split against current baseline:

1. Accuracy
2. Precision, recall, F1 for UNHEALTHY class
3. Confusion matrix
4. Error review by researcher

Promotion rule recommendation:

- Promote SFT model only if UNHEALTHY recall and F1 improve with acceptable precision and stable behavior

## Step 6: Integration strategy

Add SFT model as optional classifier mode:

- baseline mode: ResNet classifier
- sft mode: local VLM classifier
- compare mode: run both and log disagreements

Keep dashboard source unchanged by writing final decisions to the same metrics CSV schema.

## Step 7: Safety and quality gates before production use

1. Hold-out test performance meets threshold
2. Manual review on random subset passes
3. No severe failure modes on out-of-distribution images
4. Versioned model artifact and reproducible config saved

## Suggested implementation checklist on new laptop

1. Clone repo and checkout main
2. Create Python environment and install dependencies
3. Verify baseline pipeline run on a small sample
4. Prepare SFT dataset builder script from local labeled data
5. Train first 3B VLM with QLoRA
6. Evaluate against baseline ResNet
7. Integrate optional inference switch if SFT is better

## Minimal command reminders

Clone and enter:

git clone https://github.com/chandrafullstack/mt-image-analysis.git
cd mt-image-analysis

Install baseline dependencies:

pip install -r requirements.txt

Run baseline full-image pipeline:

python -m src.researcher_cli --full-image-dir "C:\lab\em_full_images" --serve

Run local retrain plus inference:

python -m src.researcher_cli --retrain --full-image-dir "C:\lab\em_full_images" --seg-method unet --serve

## Recent milestone commits

- 27a4ec3 feat(pipeline): add one-command local retrain and run workflow
- 95c2faa Add non-ML researcher step-by-step operations guide
- a5d086a Add U-Net segmentation training pipeline and explain lab-specific weights
- fdee43a Add production full-image detection/classification and real training pipeline
- b60774b Add detailed researcher flow docs and folder-pointing CLI
- 4f13e92 Add incoming/outgoing researcher workflow and deployment guide
- 3ff0a43 Initial mitochondria pipeline + dashboard

## Handoff summary

The project is in a usable production-oriented state with local retraining support. The next major track is optional local VLM SFT as an experimental classifier lane, benchmarked against existing U-Net plus ResNet before promotion.
