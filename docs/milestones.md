# Milestone Details

This document contains the detailed, step-by-step instructions for each project milestone. For a high-level overview, see the [README](../README.md).

---

## Milestone 1: Data Indexing & Label Harmonization

M1 prepares **reproducible, training-ready datasets** for a plant disease classification project. It standardizes two datasets with different label conventions (**PlantVillage** and **PlantDoc**).

### 1. Baseline Data Protocol (Source + Target)

This project evaluates robustness under dataset domain shift:

- **Source domain (PlantVillage, PV):** used for **train / validation / test**
- **Target domain (PlantDoc, PD):** used as a **held-out target test set** for cross-domain evaluation

> Baseline rule: **PlantDoc test is never used for tuning or model selection.**
>
> (Optional) PD fine-tuning experiments belong to later milestones and do not belong to PD test untouched.

---

### 2. Repository Expectations

#### 2.1 Dataset placement

Place raw datasets under `data/raw/` exactly as follows:
```
data/raw/
├── plantvillage/
│   ├── Apple___Apple_scab/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── Apple___Cedar_apple_rust/
│   └── ...
└── plantdoc/
    ├── train/
    │   ├── Apple Scab Leaf/
    │   │   ├── image001.jpg
    │   │   └── ...
    │   ├── Apple rust leaf/
    │   └── ...
    └── test/
        ├── Apple Scab Leaf/
        └── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png` (case-insensitive)

#### 2.2 Environment setup

From the **repository root**:
```bash
pip install -r requirements.txt
```

---

### 3. Milestone 1 Outputs (What you get)

M1 produces three categories of artifacts:

1. **Image inventories** (one row per image; disk scan results)
2. **Label harmonization map** (raw label → canonical label)
3. **Training-ready split CSVs** (mapped + filtered + seeded splits)

Downstream training (M2–M5) reads only the split CSVs.

---

### 4. How to Run M1 (Reproducible Pipeline)

Run the following steps in order from the repository root.

#### Step 1 — Index datasets (image inventory)
```bash
python scripts/make_index.py
```

**Writes:**

- `data/interim/plantvillage_index.csv`
- `data/interim/plantdoc_train_index.csv`
- `data/interim/plantdoc_test_index.csv`
- `outputs/dataset_summary.txt`

**Schema:** `dataset, split, raw_label, filepath_rel`

---

#### Step 2 — Build label mapping (intersection-only 26 classes)
```bash
python scripts/apply_label_map.py
```

**Reads:**

- `data/interim/*.csv`

**Writes:**

- `src/data/label_map.csv`

**Policy:**

- Only the **26 confirmed PV↔PD correspondences** are included (`include=1`)
- All other labels are excluded (`include=0`) with notes

**Schema:** `dataset, raw_label, canonical_label, include, notes`

**Expected console validation:**

- 26 unique canonical labels
- 26 included PV labels
- 26 included PD labels
- 52 included rows total (26 + 26)

---

#### Step 3 — Generate mapped split files (training-ready CSVs)
```bash
python scripts/build_mapped_splits.py
```

**Reads:**

- `data/interim/*.csv` (image-level inventory)
- `src/data/label_map.csv` (label-level mapping)

**Produces:**

- frozen label space (`canonical_id` 0–25)
- mapped PV inventory (filtered to shared classes)
- mapped PD target test set (filtered to shared classes)
- seeded, stratified PV splits: train/val/test

Writes to the splits directory configured in the script (commonly `data/splits/`).

---

### 5. Output Locations (Files to expect)

After Steps 1–3, you should have:
```bash
data/interim/
├── plantvillage_index.csv
├── plantdoc_train_index.csv
└── plantdoc_test_index.csv

src/data/
└── label_map.csv

<data_splits_dir>/                # e.g., data/splits/
├── label_space.csv               # canonical_id (0–25) ↔ canonical_label
├── plantvillage_mapped.csv       # PV filtered to 26 shared classes (pre-split)
├── plantdoc_test_mapped.csv      # PD test filtered to shared classes
├── pv_train.csv                  # PV train split (seeded, stratified)
├── pv_val.csv                    # PV val split (seeded, stratified)
└── pv_test.csv                   # PV test split (seeded, stratified)

outputs/
└── dataset_summary.txt
```

---

### 6. Quick Sanity Checks (Optional)

From the folder containing the split files:
```bash
python -c "import pandas as pd; \
print('PV train', len(pd.read_csv('pv_train.csv'))); \
print('PV val', len(pd.read_csv('pv_val.csv'))); \
print('PV test', len(pd.read_csv('pv_test.csv'))); \
print('PD test', len(pd.read_csv('plantdoc_test_mapped.csv')))"
```

---

### 7. Label Harmonization Notes

- **Canonical label format:** `crop__condition`
- **Normalization:** lowercase; spaces/hyphens → underscores
- **Mapping policy:** intersection-only (26 confirmed classes)

---

## Milestone 2: CNN Baseline Training & Evaluation

M2 implements a CNN baseline using pre-trained models (MobileNetV3 and EfficientNetB0) to establish performance benchmarks on both in-domain (PlantVillage) and cross-domain (PlantDoc) test sets.

### Training a Model

Train a CNN model on the PlantVillage training set:

#### MobileNetV3
```bash
# Train MobileNetV3 Small (faster, ~13 mins)
python src/train/train.py --config configs/baseline_mobilenet_v3_small.json
```

#### EfficientNetB0
```bash
# Train EfficientNetB0 (slower, better performance, ~28 mins)
python src/train/train.py --config configs/baseline_efficientnet_b0.json
```

**Optional arguments:**
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--data-dir`: Root directory for images (default: `.`)
- `--splits-dir`: Directory containing split CSVs (default: `data/splits`)
- `--checkpoint-dir`: Where to save model checkpoints (default: `checkpoints`)
- `--debug`: Run with truncated dataset for testing

**Outputs:**
- `checkpoints/cnn_baseline_<model>.pt`: Best model checkpoint (saved when validation accuracy improves)
- `outputs/training_log_<model>.csv`: Training metrics per epoch (loss, accuracy, time)

**Expected Results:**
- MobileNetV3: ~99.11% validation accuracy
- EfficientNetB0: ~99.28% validation accuracy

### Evaluating a Model

Evaluate a trained model on both test sets (PlantVillage and PlantDoc):

#### MobileNetV3
```bash
python src/eval/evaluate.py \
  --model-path checkpoints/baseline/baseline_mobilenet_v3_small.pt \
  --model-name mobilenet_v3_small \
  --output-file outputs/baseline_mobilenet_v3_small.csv
```

#### EfficientNetB0
```bash
python src/eval/evaluate.py \
  --model-path checkpoints/baseline/baseline_efficientnet_b0.pt \
  --model-name efficientnet_b0 \
  --output-file outputs/baseline_efficientnet_b0.csv
```

**Optional arguments:**
- `--batch-size`: Batch size (default: 32)
- `--data-dir`: Root directory for images (default: `.`)
- `--splits-dir`: Directory containing split CSVs (default: `data/splits`)
- `--output-file`: CSV file for results (default: `outputs/evaluation_results.csv`)

**Outputs:**
- `outputs/evaluation_results.csv`: Aggregate metrics (Accuracy, Precision, Recall, F1 scores)
- `outputs/report_<model>_<testset>.txt`: Per-class classification reports

**Expected Results:**

| Model | PV Test (In-Domain) | PlantDoc Test (Cross-Domain) |
|:---|:---|:---|
| **MobileNetV3** | Acc: 98.64%, F1 (Weighted): 0.9864 | Acc: 20.09%, F1 (Weighted): 0.1662 |
| **EfficientNetB0** | Acc: 99.20%, F1 (Weighted): 0.9920 | Acc: 21.00%, F1 (Weighted): 0.1974 |

> **Note:** The severe performance drop on PlantDoc (~98-99% -> ~20-21%) confirms a significant domain gap.

### Visualizing Training Progress

```bash
python src/utils/plot_training_cnn.py --log-file outputs/training_log_mobilenet_v3_small.csv
```

### Understanding the Results

**Per-Class Reports:**
The evaluation script generates detailed per-class metrics in `outputs/report_*.txt`. These reports show:
- Which disease classes transfer well across domains
- Which classes fail completely (0.00 F1-score)
- Class-specific precision, recall, and support

**Domain Gap Analysis:**
Classes with 0.00 scores on PlantDoc indicate severe domain shift. This is expected because:
- PlantVillage: Controlled lab conditions, clean backgrounds, centered leaves
- PlantDoc: Natural field conditions, cluttered backgrounds, variable lighting

---

## Milestone 3: ViT Training + Evaluation

M3 implements a ViT baseline using pre-trained models from the `timm` library to establish performance benchmarks on both in-domain (PlantVillage) and cross-domain (PlantDoc) test sets.

**Key changes from M2:**
- Added `timm` library as part of `requirements.txt`
- Consolidated all CNN training scripts into one for all models
- Use JSON files under `configs/` to specify training configurations instead of command line arguments
- Refactored transforms outside of dataloaders into its own file

### Training a Model

#### 1. Specify a JSON configuration under `configs/`

For example, `configs/baselines/baseline_vit_base_patch16_224.json`:
```json
{
    "model_name": "vit_base_patch16_224",
    "data_dir": ".",
    "splits_dir": "data/splits",
    "checkpoint_dir": "checkpoints/baseline",
    "hyperparameters": {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 0.05
    },
    "transformations": []
}
```

#### 2. Run the training script
```bash
python src/train/train.py --config configs/baselines/baseline_vit_base_patch16_224.json
```

**Outputs:**
- `checkpoints/<config file name>.pt`: Best model checkpoint
- `outputs/training_log_<config file name>.csv`: Training metrics per epoch

**Expected Results:**
- cct_14_7x2_224: ~99.64% validation accuracy
- maxvit_base_tf_224: ~98.97% validation accuracy
- swin_base_patch4_window7_224: ~99.61% validation accuracy
- vit_base_patch16_224: ~99.53% validation accuracy

### Evaluating a Model

```bash
python src/eval/evaluate.py \
  --model-path checkpoints/baseline/baseline_vit_base_patch16_224.pt \
  --model-name vit_base_patch16_224 \
  --output-file outputs/baseline_vit_base_patch16_224.csv
```

**Expected Results:**

| Model | PV Test (In-Domain) | PlantDoc Test (Cross-Domain) |
|:---|:---|:---|
| **cct_14_7x2_224** | Acc: 99.45%, F1 (Weighted): 0.9945 | Acc: 31.96%, F1 (Weighted): 0.2872 |
| **maxvit_base_tf_224** | Acc: 98.95%, F1 (Weighted): 0.9895 | Acc: 33.33%, F1 (Weighted): 0.3017 |
| **swin_base_patch4_window7_224** | Acc: 99.78%, F1 (Weighted): 0.9978 | Acc: 34.25%, F1 (Weighted): 0.2851 |
| **vit_base_patch16_224** | Acc: 98.36%, F1 (Weighted): 0.9834 | Acc: 29.22%, F1 (Weighted): 0.2692 |

### Visualizing Training Progress

```bash
python src/utils/plot_training.py \
  --log-file outputs/training_log_baseline_vit_base_patch16_224.csv \
  --output-name training_metrics_vit_base_patch16_224.png
```

---

## Milestone 4: Robustness Improvements (Augmentations) + Ablations

M4 adds configurable data augmentation to improve cross-domain robustness on PlantDoc.

### Config-Driven Augmentation

Augmentation transforms are specified in the training config JSON:

```json
{
    "model_name": "vit_base_patch16_224",
    "data_dir": ".",
    "splits_dir": "data/splits",
    "checkpoint_dir": "checkpoints/baseline",
    "hyperparameters": {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 0.05
    },
    "transformations": [
      {
        "name": "random_rotation",
        "params": { "degrees": 10 }
      },
      {
        "name": "color_jitter",
        "params": { "brightness": 0, "contrast": 0, "saturation": 0, "hue": 0 }
      }
    ]
}
```

### Augmentation Strategies Evaluated

- **Random Erasing** — best single augmentation (Macro F1 0.37 on PlantDoc)
- **Rotation Only** / **Rotation + Gaussian Blur** — strong on CCT
- **CutMix** / **CutMixUp** / **MixUp** — sample mixing strategies
- **Gaussian Blur Only** — simple spatial augmentation

### Alternative Training Paradigms

- **Supervised Contrastive Learning (SupCon)** — representation learning approach (Macro F1 0.35)
- **Multi-Task ViT** — joint learning with auxiliary objectives (Macro F1 0.21)

See [`results_zip/report.md`](../results_zip/report.md) for the full experiment comparison report.

---

## Milestone 5 (Planned)

- Reliability layer: calibration + abstain option
