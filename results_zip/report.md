# Master Experiment Comparison Report (as of 16-Mar-2026)

**Scope:** Baseline + augmentation experiments + alternative training paradigms (multi-task, supervised contrastive) + perspective (zero-shot and finetuned) + affine across 6 model families on **PlantDoc_Test** and **PV_Test**.
**Decision metric:** **Macro F1 on PlantDoc_Test** (PV_Test is near-saturated and not a useful discriminator).

**Definition:** A **base run** is a standard (experiment x architecture) run excluding `_adamw` / `_epoch15` variants and excluding SupCon/Multi-task paradigms.

---

## 1) Executive Summary (What we ran, what won, what to do next)

### What was evaluated
- Model families: **CCT, EfficientNet-B0, MaxViT, MobileNetV3, Swin, ViT**
- Augmentation experiments: **baseline** + **cutmix, cutmixup, erasing, gaussian-blur-only, mixup, rotate-only, rotate+gaussian, affine, perspective (zero-shot), perspective-finetuned**
- Alternative training paradigms: **Multi-Task ViT**, **Supervised Contrastive Learning (SupCon)**
- Test sets: **PlantDoc_Test** (hard) and **PV_Test** (easy)
- All experiments share the same evaluation pipeline; only the augmentation policy or training paradigm differs.
- **144 total rows** across **13 experiments** and **10 model variants**.

### Key findings
- **PV_Test is saturated:** 67/72 runs have **Macro F1 >= 0.97** (93%) -> not useful for selecting winners.
- **PlantDoc_Test is the meaningful benchmark:** Macro F1 spans ~**0.08-0.45** across all runs.
- **Overall best run on PlantDoc_Test (among all evaluated runs):**
  - **aug-perspective / swin_base_patch4_window7_224** (zero-shot) -- **Acc 0.48, Macro F1 0.45, Weighted F1 0.46**
- **Second best (including variants):**
  - **aug_erasing / vit_base_patch16_224_adamw** -- **Acc 0.41, Macro F1 0.37, Weighted F1 0.39**
- **Best standard augmentation runs (base runs) on PlantDoc_Test:**
  - **aug_erasing / cct_14_7x2_224** -- **Acc 0.39, Macro F1 0.35**
  - **aug-rotate-only / cct_14_7x2_224** -- **Acc 0.39, Macro F1 0.34**
  - **aug-rotate-gaussian / cct_14_7x2_224** -- **Acc 0.37, Macro F1 0.33**
  - **aug-gaussian-blur-only / swin_base_patch4_window7_224** -- **Acc 0.36, Macro F1 0.33**
  - **aug-cutmix / swin_base_patch4_window7_224** -- **Acc 0.37, Macro F1 0.32**
- **Best alternative-paradigm run:**
  - **supcon / vit_base_patch16_224** -- **Acc 0.38, Macro F1 0.35**

---

## 2) Main Results (PlantDoc_Test)

### 2.1 Best run per experiment (including variants / alternative paradigms where present)

Ranked by Macro F1, tie-break by Accuracy. Runs marked **+** are non-base (optimizer/epoch variant or alternative paradigm).

| Experiment | Best Run | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| aug-perspective | **swin_base_patch4_window7_224** | 0.48 | 0.45 | 0.46 |
| aug_erasing | **vit_base_patch16_224_adamw** + | 0.41 | 0.37 | 0.39 |
| supcon | **vit_base_patch16_224** + | 0.38 | 0.35 | 0.35 |
| aug-rotate-only | **cct_14_7x2_224** | 0.39 | 0.34 | 0.35 |
| aug-rotate-gaussian | **cct_14_7x2_224** | 0.37 | 0.33 | 0.33 |
| aug-gaussian-blur-only | **swin_base_patch4_window7_224** | 0.36 | 0.33 | 0.33 |
| affine | **swin_base_patch4_window7_224** | 0.37 | 0.32 | 0.34 |
| aug-cutmix | **swin_base_patch4_window7_224** | 0.37 | 0.32 | 0.33 |
| aug-perspective-finetuned | **cct_14_7x2_224** | 0.37 | 0.32 | 0.34 |
| aug-cutmixup | **swin_base_patch4_window7_224** | 0.34 | 0.32 | 0.32 |
| baseline | **maxvit_base_tf_224** | 0.33 | 0.29 | 0.30 |
| aug-mixup | **swin_base_patch4_window7_224** | 0.33 | 0.28 | 0.29 |
| multi_task_vit | **vit_base_patch16_224** + | 0.23 | 0.21 | 0.21 |

> **+** = non-base run (variant or alternative paradigm); see Sections 4 and 5 for details.

### 2.2 Interpretation
- **Perspective augmentation (zero-shot)** is the clear winner on PlantDoc_Test, with **Swin** achieving **Macro F1 0.45** — a significant margin above all other experiments.
- **Random Erasing** remains the strongest among standard finetuned augmentation runs; the best overall finetuned run is **ViT + AdamW** under `aug_erasing` (Macro F1 0.37).
- **Perspective-finetuned** underperforms the zero-shot perspective, suggesting the frozen backbone + perspective augmentation combination captures more generalizable features for PlantDoc.
- **SupCon (ViT)** achieves **Macro F1 0.35**, matching the best base augmentation run, suggesting representation learning can rival augmentation-only approaches.
- **Affine augmentation** performs comparably to cutmix and cutmixup (Macro F1 0.32).
- **Multi-Task ViT** underperforms (Macro F1 0.21), suggesting this framing alone does not improve PlantDoc generalization.
- **CCT** is consistently strong among standard augmentation runs (rotate-only, rotate+gaussian, perspective-finetuned).
- **Swin** dominates in 6 out of 13 experiments as the best model.

---

## 3) PV_Test Saturation (Why it should not drive selection)

- **72 runs** evaluated on PV_Test
- **67/72 (93%)** have **Macro F1 >= 0.97**
- Conclusion: **PV_Test is saturated** -> prioritize **PlantDoc_Test** for model selection.

---

## 4) Variant Runs (aug_erasing)

`aug_erasing` includes additional runs beyond the standard model set:
- **Epoch variants (`_epoch15`)**: `cct_14_7x2_224`, `maxvit_base_tf_224`
- **Optimizer variants (`_adamw`)**: `swin_base_patch4_window7_224`, `vit_base_patch16_224`

Variants are treated as separate runs due to training differences.

---

## 5) Alternative Training Paradigms (ViT only)

| Experiment | Description | PlantDoc Acc | PlantDoc Macro F1 | PV Acc | PV Macro F1 |
|---|---|---:|---:|---:|---:|
| multi_task_vit | Multi-task learning ViT | 0.23 | 0.21 | 0.98 | 0.97 |
| supcon | Supervised contrastive pre-training | 0.38 | 0.35 | 0.98 | 0.97 |

---

## 6) Perspective Experiments Comparison

Two variants of perspective augmentation were evaluated:

| Variant | Description | Best Model | PlantDoc Acc | PlantDoc Macro F1 |
|---|---|---|---:|---:|
| aug-perspective | Zero-shot (backbone frozen) | swin_base_patch4_window7_224 | 0.48 | 0.45 |
| aug-perspective-finetuned | Finetuned backbone | cct_14_7x2_224 | 0.37 | 0.32 |

The zero-shot variant significantly outperforms the finetuned variant on PlantDoc_Test, suggesting that the frozen backbone preserves pre-trained representations that generalize better to the PlantDoc domain.

---

## 7) Future Work / Next Steps

1. Scale SupCon to other strong architectures: **CCT**, **Swin**, and test **SupCon + Random Erasing**.
2. Strengthen training schedule for top performers: **25-30 epochs**, **cosine LR**, and backbone unfreezing strategy.
3. Combine augmentations: **Random Erasing + Rotate + Gaussian blur**.
4. Investigate why zero-shot perspective outperforms finetuned perspective — explore partial backbone unfreezing strategies.

---

## 8) Notes
- `multi_task_vit` and `supcon` are separate paradigms (no configs under `./configs/`).
- No parsing errors encountered in report files.
- Full run-by-run metrics are stored in `master_results.csv` for filtering/plotting.

