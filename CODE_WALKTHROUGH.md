# Code walkthrough — how the training pipeline actually works

A page you can read top-to-bottom (or talk through) to explain what the
code does, in the order it runs. Written for someone comfortable with
Python but not deep into PyTorch internals.

The pipeline has **three** training stages, each in its own file:

| Stage | File | What it produces |
|---|---|---|
| 1. **Pseudo-labelling** (Claude scores crops) | [src/claude_score_crops.py](src/claude_score_crops.py) | A consensus CSV: one row per crop, with a HEALTHY/UNHEALTHY vote |
| 2. **Classifier training** (ResNet-50) | [src/train_pseudo_labels.py](src/train_pseudo_labels.py) | `resnet50_best.pt` — the model the dashboard uses |
| 3. **Segmenter training** (U-Net, optional) | [src/train_unet.py](src/train_unet.py) | `unet_best.pt` — finds mitochondria in a full image |

At inference time, [src/full_image_inference.py](src/full_image_inference.py)
glues them together: U-Net finds blobs → crops them → ResNet-50 scores
each crop.

Below, each section walks through one file. **Bold names** are
function names you can search for in the file.

---

## Stage 1 — Pseudo-labelling
*File: [src/claude_score_crops.py](src/claude_score_crops.py)*

The point: we have thousands of unlabelled crops and no patience to
label them by hand. So we ask Claude (vision API) to look at each one
and say "HEALTHY" or "UNHEALTHY". To keep labels honest we ask three
times and only keep labels where Claude agrees with itself.

Read the file in this order:

1. **`SYSTEM_PROMPT`** (top of file) — the instructions Claude is given.
   This is where the *definition* of healthy/unhealthy lives. If the
   researcher disagrees with a call, this is usually where to fix it.

2. **`CostTracker`** — small class that keeps a running USD total and
   refuses to make a call that would push past `--budget-usd`. *(This
   is the only class in the file; it exists because we need state that
   survives between calls. Could be a global, but a class is cleaner.)*

3. **`_image_to_b64(path)`** — reads a crop PNG and turns it into the
   base64 string the API expects.

4. **`_classify_one(...)`** — single call. Sends one image to Claude,
   parses the JSON answer, returns the label + confidence.

5. **`_consensus_classify(...)`** — the "three votes" loop. Calls
   `_classify_one` three times with different seeds, then computes:
   - `final_classification` — majority vote
   - `label_agreement` — fraction of votes that matched the majority
     (0.67 = 2 out of 3, 1.00 = unanimous)
   - `calibrated_confidence` — average confidence × agreement

6. **`score_crops(...)`** — top-level loop over every crop in a
   metrics CSV. Writes the consensus CSV when done.

7. **`main()`** — CLI entry point.

**Output:** `outputs/predictions/round*_consensus.csv` — one row per
crop with the columns the next stage needs.

---

## Stage 2 — Classifier training
*File: [src/train_pseudo_labels.py](src/train_pseudo_labels.py)*
*(This is the main file. The file's top docstring has a "Function map"
header you can read first.)*

This is the file the researcher most needs to understand. Read in this
order:

### Step 1 — `assemble_dataset(...)`

Takes the consensus CSVs from Stage 1 and joins them to the
**segmentation metrics CSVs** (which have one row per detected
mitochondrion with its `source_file` — the parent TIF). Then filters
to only the crops where:

- the label is HEALTHY or UNHEALTHY (drop UNCLEAR), and
- `label_agreement >= 0.67` (≥ 2/3 votes agreed), and
- `calibrated_confidence >= 0.55`.

Returns one big pandas DataFrame, one row per usable crop, with the
columns: `crop_path`, `source_file`, `y` (0=HEALTHY, 1=UNHEALTHY).

### Step 2 — `make_splits(...)`

The critical step. We split by **source image**, not by crop, using
`sklearn.GroupShuffleSplit`. Why: two crops from the same TIF share
the same staining intensity, the same microscope, the same
focus-noise. If they're in different splits the test score is
artificially high — the model is "memorising" image-level noise.

The function holds out:
- **test** = ~20% of source images (model never sees these),
- **val**  = ~20% of what's left (used to pick the best epoch),
- **train** = the rest.

If there are too few source images for grouping (small pilot runs),
it falls back to a stratified crop-level split and prints a warning.

### Step 3a — `_make_dataset_class(...)`

Defines a tiny `_MitoCropDataset` class. PyTorch *requires* a class
here — its `DataLoader` calls `__len__` and `__getitem__` to feed
batches. We just:

- open the PNG (`Image.open(...).convert("L")` → grayscale),
- run it through the `transform` (resize, augment, normalise),
- return `(tensor, int_label)`.

It's wrapped in a function only so we can defer the `torch` import.

### Step 3b — The training loop *(inside `train()`)*

For each epoch:

```
for batch of crops in train_loader:
    predictions = model(batch)
    loss        = CrossEntropyLoss(predictions, true_labels)
    loss.backward()      # compute gradients
    optimizer.step()     # update weights
```

After every epoch we:

- Run the model on the **val** loader, compute `val_macro_f1`.
- If it's the best val score so far, save the weights to
  `resnet50_best.pt`.

A few details worth flagging to the researcher:

- We start from **ResNet-50 pretrained on ImageNet** (`weights="IMAGENET1K_V2"`).
  The model already "knows" edges, textures and shapes from 1M
  natural images — we only have to teach it the final HEALTHY vs
  UNHEALTHY distinction.
- We replace the final layer (`model.fc = nn.Linear(..., 2)`) so it
  outputs 2 logits instead of 1000.
- We pass **class weights** into the loss because HEALTHY outnumbers
  UNHEALTHY ~4–5:1; without this the model would learn to call
  everything HEALTHY.
- We also use a `WeightedRandomSampler` so each batch sees a roughly
  balanced mix.

### Step 4 — `_eval_split(...)` and `_print_metrics(...)`

After training we reload the **best** weights and evaluate on
train / val / test. The numbers go to console *and* to
`evaluation_summary.json` so we have a permanent record.

Key metric to report to the researcher: **macro F1 on the test set**.
That's the honest "unseen images" number.

---

## Stage 3 — Segmenter training (optional)
*File: [src/train_unet.py](src/train_unet.py)*

Only needed if we want better than the default Otsu-threshold
segmenter. Read in this order:

1. **`_load_gray(path)`** / **`_resize(...)`** — image utilities.
2. **`build_pairs(images_dir, masks_dir)`** — pairs each EM image with
   its ground-truth mask by matching filenames.
3. **`EMMaskDataset(Dataset)`** — the PyTorch dataset wrapper, same
   pattern as Stage 2.
4. **`dice_score(...)`** / **`iou_score(...)`** — the segmentation
   metrics. (Dice = overlap / mean-area; IoU = overlap / union.)
5. **`run_epoch(...)`** — one pass over the data. Used for both
   training and validation; the `train` flag toggles `backward()`.
6. **`train_unet(...)`** — orchestrates the full training run.

**Output:** `unet_best.pt`.

---

## Inference — putting it all together
*File: [src/full_image_inference.py](src/full_image_inference.py)*

This is the file the dashboard runs on every new image. Read in this
order:

1. **`_load_gray(path)`** / **`_preprocess(img)`** — load + contrast-
   stretch.
2. **`_segment_mito(img, seg_method, unet_weights)`** —
   - if a U-Net weights file is provided, runs the U-Net (tile-wise
     via `_tile_predict_unet`),
   - otherwise falls back to a classical Otsu threshold + morphology
     clean-up. *This is why the pipeline still works without a
     trained U-Net.*
3. **`_load_classifier(weights_path)`** — loads the ResNet-50 weights
   produced by Stage 2 and the eval-time transforms.
4. **`_classify_crop(crop, bundle, tf)`** — runs one crop through the
   classifier and returns `(label, prob)`.
5. **Helper rules** — `_shape_category`, `_fission_fusion`,
   `_rule_label` add the non-ML descriptors (round vs elongated,
   fission vs fusion, etc) from skimage shape measurements.
6. **`run_full_image_inference(...)`** — the main loop:
   for each image →
     preprocess →
     segment →
     for each blob: crop → classify → compute shape features →
     write a row to the output CSV + save the crop PNG.

**Output:** `outputs/metrics/*.csv` (one row per detected
mitochondrion) and `outputs/crops/*.png` (the cropped images the
dashboard shows).

---

## A note on classes

You'll see exactly **five** classes across `src/`:

| Class | File | Why it's a class |
|---|---|---|
| `_MitoCropDataset` | `train_pseudo_labels.py` | PyTorch's `DataLoader` requires it |
| `EMMaskDataset` | `train_unet.py` | Same reason |
| `MitoHealthDataset` | `cnn_model.py` | Same reason (legacy) |
| `CostTracker` | `claude_score_crops.py` | Holds running USD total across API calls |
| `BuildStats`, `IngestResult` | `sft_dataset_builder.py`, `incoming_feedback.py` | `@dataclass` — these are just typed records, not really OO |

Every other function in the pipeline is a plain top-level `def`. So
when explaining the code, you can describe it as "a series of
functions called in order" and only have to point out a class when
PyTorch's API forced our hand.

---

## TL;DR for the researcher

> Stage 1 calls Claude three times per crop and keeps the ones it
> agrees with itself on. Stage 2 takes those labels and fine-tunes
> ResNet-50, being very careful to keep crops from the same source
> image together in one split so the test score is honest. Stage 3 is
> an optional better-than-default segmenter. Inference is just:
> segment → crop → classify, one row per blob.
