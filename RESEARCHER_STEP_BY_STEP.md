# Researcher Step-by-Step Guide (No ML Knowledge Required)

This guide is for researchers who only want to:

1. Put EM images in a folder
2. Run one command (or double-click a script)
3. See results in the dashboard

## What You Need Before Starting

- A Windows computer
- Python installed (3.10 or newer)
- This project folder downloaded
- Your EM images in one folder

You do NOT need a Claude subscription for normal use.

## One-Time Setup (Do This Once)

1. Open PowerShell.
2. Go to the project folder:

```powershell
cd C:\Users\Z00593BV\Documents\Claude\Projects\MT\mito_classifier
```

3. Install required packages:

```powershell
pip install -r requirements.txt
```

4. Quick check:

```powershell
python -c "import fastapi, uvicorn, jinja2; print('Setup OK')"
```

If this prints Setup OK, continue.

## Workflow A: Analyze Full EM Images (Recommended)

Use this when each image contains many structures and multiple mitochondria.

### Step 1: Put Images in a Folder

Example:

- C:\lab\em_full_images

Supported formats:

- .png, .jpg, .jpeg, .tif, .tiff, .bmp

### Step 2: Run Analysis + Open Dashboard

In PowerShell from project root:

```powershell
python -m src.researcher_cli --full-image-dir "C:\lab\em_full_images" --serve
```

Or with script:

```powershell
scripts\run_full_image_flow.bat "C:\lab\em_full_images"
```

### Step 3: Open Dashboard in Browser

- http://127.0.0.1:8000

You will see detected mitochondria and classification results.

### Step 4: Where Results Are Saved

- outputs/metrics/features_with_gratio.csv
- outputs/crops/

## Workflow B: Use Expert-Labeled Crops (Healthy vs Unhealthy)

Use this when researchers already have cropped mitochondria and labels.

### Step 1: Drop Files Into Folders

- incoming/healthy
- incoming/unhealthy

### Step 2: Start Dashboard

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or script:

```powershell
scripts\start_dashboard.bat
```

### Step 3: Refresh Dashboard

When dashboard loads, ingestion runs automatically.

### Step 4: Processed Files Move Automatically

- outgoing/processed/healthy
- outgoing/processed/unhealthy
- outgoing/rejected

## If You Want Better Accuracy (Optional Advanced)

You can train your own models using your lab data.

- Segmentation model: U-Net
- Health classifier: ResNet

This is optional. The app works without this step.

## How to Prepare Segmentation Masks (Simple Explanation)

A mask is a black/white image matching your EM image size exactly.

- White (255) = mitochondria
- Black (0) = not mitochondria

For image `tile_001.png`, save mask as either:

- tile_001_mask.png (recommended)
- tile_001.png (same name)

Place data here:

- Images: data/labeled/segmentation/images
- Masks: data/labeled/segmentation/masks

## Train U-Net Segmentation (Optional)

```powershell
python src/train_unet.py --images-dir data/labeled/segmentation/images --masks-dir data/labeled/segmentation/masks --output outputs/models --epochs 30 --batch-size 4
```

Output files:

- outputs/models/unet_best.pt
- outputs/models/unet_last.pt

## Train Health Classifier (Optional)

Put labeled crops in:

- data/labeled/crops/HEALTHY
- data/labeled/crops/UNHEALTHY

Run:

```powershell
python src/cnn_model.py --data data/labeled/crops --output outputs/models --epochs 25 --batch-size 16
```

Output files:

- outputs/models/resnet50_best.pt
- outputs/models/resnet50_last.pt

## Run Full Image Analysis With Trained Models (Optional)

```powershell
python -m src.researcher_cli --full-image-dir "C:\lab\em_full_images" --seg-method unet --unet-weights "outputs/models/unet_best.pt" --classifier-weights "outputs/models/resnet50_best.pt" --serve
```

## Troubleshooting

### Error: ModuleNotFoundError: No module named app

Cause: command is run from wrong folder.

Fix:

1. In PowerShell:

```powershell
cd C:\Users\Z00593BV\Documents\Claude\Projects\MT\mito_classifier
```

2. Run again.

### Dashboard does not open on port 8000

Fix:

1. Free port:

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
```

2. Restart dashboard:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Minimum Daily Routine for Researchers

1. Put new full EM images in one folder.
2. Run one command:

```powershell
python -m src.researcher_cli --full-image-dir "C:\lab\today_images" --serve
```

3. Open dashboard and review results.
4. Export or share outputs from outputs/metrics/features_with_gratio.csv.
