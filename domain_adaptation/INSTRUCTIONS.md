# Domain Adaptation — Quick Start Guide

Three domain adaptation approaches to improve DINOv3 on SPEED+ real domains (sunlamp, lightbox).

All commands should be run from the project root directory.

---

## Option 4: Test-Time Adaptation (TTA) — Start Here

**No retraining needed.** Takes any existing checkpoint and adapts it at inference time using unlabeled real images.

Three methods available:

```bash
# TENT — entropy minimization on LayerNorm params (recommended)
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method tent --num_steps 5 --lr 1e-4

# Norm adaptation — simplest, just updates LayerNorm statistics
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method norm_adapt

# MEMO — per-sample adaptation with augmentations
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method memo --memo_augmentations 8
```

Output: side-by-side comparison of baseline vs adapted metrics (SLAB, PCK, pixel error) for each domain.

---

## Option 3: Domain Generalization (DSU + MixStyle) — Requires Retraining

Injects feature-level perturbations into ViT blocks during training. Trains **only on synthetic data** — no target domain images needed. The model learns to be invariant to domain-specific feature statistics.

```bash
# Train from scratch
python -m domain_adaptation.option3_domain_generalization.train_dg \
    --config domain_adaptation/option3_domain_generalization/config_dg.yaml

# Warm-start from existing checkpoint
python -m domain_adaptation.option3_domain_generalization.train_dg \
    --config domain_adaptation/option3_domain_generalization/config_dg.yaml \
    --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
```

Key hyperparameters in `config_dg.yaml`:
- `dsu_alpha` / `dsu_beta`: perturbation strength (default 0.3, try 0.1–0.5)
- `mixstyle_prob`: probability of style mixing (default 0.5)
- `apply_to_last_n_blocks`: which ViT blocks to modify (default: last 4)

---

## Option 2: Self-Training with Pseudo-Labels — Most Complex

Iterative approach:
1. Run inference on unlabeled real images (style split) to get pseudo-labels
2. Filter by heatmap confidence
3. Fine-tune on synthetic (GT labels) + real (pseudo-labels)
4. Repeat

```bash
# Run self-training (3 iterations by default)
python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train.yaml

# With custom checkpoint
python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train.yaml \
    --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
```

Key hyperparameters in `config_self_train.yaml`:
- `confidence_threshold`: min heatmap confidence to accept a pseudo-label (default 0.3)
- `pseudo_label_weight`: loss weight for pseudo-labeled samples (default 0.5)
- `num_iterations`: self-training rounds (default 3)
- `epochs_per_iteration`: fine-tuning epochs per round (default 10)

---

## Data Protocol

All options use the same split protocol to avoid training on test data:

| Split file | Purpose | Used by |
|---|---|---|
| `data/splits/sunlamp_style.txt` | Adaptation (unlabeled) | Option 2, Option 4 |
| `data/splits/sunlamp_test.txt` | Evaluation only | All options |
| `data/splits/lightbox_style.txt` | Adaptation (unlabeled) | Option 2, Option 4 |
| `data/splits/lightbox_test.txt` | Evaluation only | All options |

- **Style split** (~20%): used for pseudo-label generation (Option 2) or TTA adaptation (Option 4). No GT labels used.
- **Test split** (~80%): used for evaluation only. GT labels used only to compute metrics.
- Option 3 does not use any real images during training.

---

## Recommended Order

1. **Option 4 (TTA/TENT)** — quickest to test, no retraining, works with any checkpoint
2. **Option 3 (DSU + MixStyle)** — requires retraining but strongest theoretical basis for domain generalization
3. **Option 2 (Self-Training)** — most complex, works best when the model already performs decently on the target domain

## Evaluation Metrics

All options report:
- **SLAB score** (lower is better): `mean(2*arccos(|<q_pred,q_gt>|) + ||t_pred-t_gt||/||t_gt||)`
- **PCK@0.05**: percentage of correct keypoints
- **Pixel error**: mean keypoint error in pixels
- **Rotation error**: orientation error in degrees
- **Translation error**: relative position error
