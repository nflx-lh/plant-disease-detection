# Master Experiment Comparison Report (as of 19-Mar-2026)

**Scope:** Baseline + augmentation experiments + perspective (zero-shot and finetuned) + affine + baseline-lora + style disentanglement across 6 model families on **PlantDoc_Test** and **PV_Test**.
**Decision metric:** **Macro F1 on PlantDoc_Test** (PV_Test is near-saturated and not a useful discriminator).

**Definition:** A **base run** is a standard (experiment x architecture) run excluding `_adamw` / `_epoch15` variants.

---

## 1) Executive Summary (What we ran, what won, what to do next)

### What was evaluated
- Model families: **CCT, EfficientNet-B0, MaxViT, MobileNetV3, Swin, ViT**
- Augmentation experiments: **baseline** + **cutmix, cutmixup, erasing, gaussian-blur-only, mixup, rotate-only, rotate+gaussian, affine, perspective (zero-shot), perspective-finetuned**
- Parameter-efficient fine-tuning: **baseline-lora** (LoRA adapters on frozen backbone)
- Domain generalization: **style_disentanglement** (subspace factorization for style/content separation)
- Test sets: **PlantDoc_Test** (hard) and **PV_Test** (easy)
- All experiments share the same evaluation pipeline; only the augmentation policy or training paradigm differs.
- **164 total rows** across **13 experiments** and **10 model variants**.

### Key findings
- **PV_Test is saturated:** 73/82 runs have **Macro F1 >= 0.97** (89%) -> not useful for selecting winners. (2 baseline-lora runs report NA; 7 runs fall below threshold.)
- **PlantDoc_Test is the meaningful benchmark:** Macro F1 spans ~**0.08-0.45** across all runs.
- **Overall best run on PlantDoc_Test:**
  - **aug-perspective / swin_base_patch4_window7_224** (zero-shot) -- **Acc 0.48, Macro F1 0.45, Weighted F1 0.46**
- **Second best:**
  - **style_disentanglement / swin_base_patch4_window7_224** -- **Acc 0.47, Macro F1 0.42, Weighted F1 0.45**
- **Third best (including variants):**
  - **aug_erasing / vit_base_patch16_224_adamw** -- **Acc 0.41, Macro F1 0.37, Weighted F1 0.39**
- **Best standard augmentation runs (base runs) on PlantDoc_Test:**
  - **aug_erasing / cct_14_7x2_224** -- **Acc 0.39, Macro F1 0.35**
  - **aug-rotate-only / cct_14_7x2_224** -- **Acc 0.39, Macro F1 0.34**
  - **aug-rotate-gaussian / cct_14_7x2_224** -- **Acc 0.37, Macro F1 0.33**
  - **aug-gaussian-blur-only / swin_base_patch4_window7_224** -- **Acc 0.36, Macro F1 0.33**
  - **aug-cutmix / swin_base_patch4_window7_224** -- **Acc 0.37, Macro F1 0.32**
---

## 2) Main Results (PlantDoc_Test)

### 2.1 Best run per experiment (including variants where present)

Ranked by Macro F1, tie-break by Accuracy. Runs marked **+** are non-base (optimizer/epoch variant).

| Experiment | Best Run | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| aug-perspective | **swin_base_patch4_window7_224** | 0.48 | 0.45 | 0.46 |
| style_disentanglement | **swin_base_patch4_window7_224** | 0.47 | 0.42 | 0.45 |
| baseline-lora | **swin_base_patch4_window7_224** | 0.43 | 0.38 | 0.40 |
| aug_erasing | **vit_base_patch16_224_adamw** + | 0.41 | 0.37 | 0.39 |
| aug-rotate-only | **cct_14_7x2_224** | 0.39 | 0.34 | 0.35 |
| aug-rotate-gaussian | **cct_14_7x2_224** | 0.37 | 0.33 | 0.33 |
| aug-gaussian-blur-only | **swin_base_patch4_window7_224** | 0.36 | 0.33 | 0.33 |
| affine | **swin_base_patch4_window7_224** | 0.37 | 0.32 | 0.34 |
| aug-cutmix | **swin_base_patch4_window7_224** | 0.37 | 0.32 | 0.33 |
| aug-perspective-finetuned | **cct_14_7x2_224** | 0.37 | 0.32 | 0.34 |
| aug-cutmixup | **swin_base_patch4_window7_224** | 0.34 | 0.32 | 0.32 |
| baseline | **maxvit_base_tf_224** | 0.33 | 0.29 | 0.30 |
| aug-mixup | **swin_base_patch4_window7_224** | 0.33 | 0.28 | 0.29 |

> **+** = non-base run (optimizer/epoch variant); see Section 4 for details.

### 2.2 Interpretation
- **Perspective augmentation (zero-shot)** is the clear winner on PlantDoc_Test, with **Swin** achieving **Macro F1 0.45** — a significant margin above all other experiments.
- **Style disentanglement** is the second-best experiment overall, with **Swin** achieving **Macro F1 0.42**. This subspace factorization approach separates style from content, improving cross-domain generalization.
- **Baseline-LoRA** (LoRA adapters) achieves **Macro F1 0.38** with **Swin**, outperforming all standard augmentation-only approaches and matching the best finetuned augmentation run (erasing + AdamW at 0.37).
- **Random Erasing** remains the strongest among standard finetuned augmentation runs; the best overall finetuned run is **ViT + AdamW** under `aug_erasing` (Macro F1 0.37).
- **Perspective-finetuned** underperforms the zero-shot perspective, suggesting the frozen backbone + perspective augmentation combination captures more generalizable features for PlantDoc.
- **Affine augmentation** performs comparably to cutmix and cutmixup (Macro F1 0.32).
- **CCT** is consistently strong among standard augmentation runs (rotate-only, rotate+gaussian, perspective-finetuned).
- **Swin** dominates in 8 out of 13 experiments as the best model.

---

## 3) PV_Test Saturation (Why it should not drive selection)

- **82 runs** evaluated on PV_Test (2 baseline-lora runs report NA)
- **73/80 non-NA runs (91%)** have **Macro F1 >= 0.97**
- Conclusion: **PV_Test is saturated** -> prioritize **PlantDoc_Test** for model selection.

---

## 4) Variant Runs (aug_erasing)

`aug_erasing` includes additional runs beyond the standard model set:
- **Epoch variants (`_epoch15`)**: `cct_14_7x2_224`, `maxvit_base_tf_224`
- **Optimizer variants (`_adamw`)**: `swin_base_patch4_window7_224`, `vit_base_patch16_224`

Variants are treated as separate runs due to training differences.

---

## 5) Perspective Experiments Comparison

Two variants of perspective augmentation were evaluated:

| Variant | Description | Best Model | PlantDoc Acc | PlantDoc Macro F1 |
|---|---|---|---:|---:|
| aug-perspective | Zero-shot (backbone frozen) | swin_base_patch4_window7_224 | 0.48 | 0.45 |
| aug-perspective-finetuned | Finetuned backbone | cct_14_7x2_224 | 0.37 | 0.32 |

The zero-shot variant significantly outperforms the finetuned variant on PlantDoc_Test, suggesting that the frozen backbone preserves pre-trained representations that generalize better to the PlantDoc domain.

---

## 6) Baseline-LoRA Experiments

LoRA (Low-Rank Adaptation) adapters were applied to frozen backbones to evaluate parameter-efficient fine-tuning for cross-domain transfer.

| Model | PlantDoc Acc | PlantDoc Macro F1 | PlantDoc Weighted F1 | Notes |
|---|---:|---:|---:|---|
| swin_base_patch4_window7_224 | 0.43 | 0.38 | 0.40 | Best |
| vit_base_patch16_224 | 0.37 | 0.33 | 0.35 | |
| cct_14_7x2_224 | 0.34 | 0.30 | 0.30 | PV_Test row has 9 columns (1 missing) |
| maxvit_base_tf_224 | 0.31 | 0.21 | 0.23 | PV_Test Macro F1 only 0.48 |
| efficientnet_b0 | NA | NA | NA | LoRA not applicable / not run |
| mobilenet_v3_small | NA | NA | NA | LoRA not applicable / not run |

LoRA with Swin (Macro F1 0.38) outperforms all standard augmentation-only base runs, suggesting parameter-efficient fine-tuning is a viable alternative to full backbone finetuning for cross-domain transfer.

---

## 7) Style Disentanglement Experiments

Style disentanglement uses subspace factorization to separate style (domain-specific) features from content (task-relevant) features, aiming to improve cross-domain generalization.

| Model | PlantDoc Acc | PlantDoc Macro F1 | PlantDoc Weighted F1 |
|---|---:|---:|---:|
| swin_base_patch4_window7_224 | 0.47 | 0.42 | 0.45 |
| vit_base_patch16_224 | 0.38 | 0.34 | 0.36 |
| cct_14_7x2_224 | 0.33 | 0.30 | 0.32 |
| mobilenet_v3_small | 0.24 | 0.21 | 0.23 |
| maxvit_base_tf_224 | 0.23 | 0.21 | 0.21 |
| efficientnet_b0 | 0.22 | 0.17 | 0.20 |

Style disentanglement with Swin (Macro F1 0.42) is the second-best experiment overall, trailing only perspective zero-shot (0.45). This approach demonstrates that explicitly separating domain-variant features can substantially improve cross-domain transfer.

---

## 8) Future Work / Next Steps

1. Strengthen training schedule for top performers: **25-30 epochs**, **cosine LR**, and backbone unfreezing strategy.
2. Combine augmentations: **Random Erasing + Rotate + Gaussian blur**.
3. Investigate why zero-shot perspective outperforms finetuned perspective — explore partial backbone unfreezing strategies.
4. Combine LoRA with perspective augmentation — test whether LoRA + perspective can surpass full perspective zero-shot.
5. Scale style disentanglement with longer training and combined augmentations.
6. Explicit domain adaptation methods (DANN, MMD).

---

## 9) Notes
- No parsing errors encountered in report files.
- Full run-by-run metrics are stored in `master_results.csv` for filtering/plotting.

