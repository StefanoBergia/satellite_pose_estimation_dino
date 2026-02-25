# `evaluate_robust.py` — Example Commands

Robust EPnP pose evaluation with adaptive confidence, RANSAC, optional LM refinement, and cascading inlier schedules.

---

## Quick Reference

```bash
# DINO baseline (val + lightbox + sunlamp)
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth

# HRNet-W32 (colleague's pipeline)
python evaluate_robust.py --checkpoint outputs_hrnet_w32/converted_model.pth --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --no_conf_filter --rmse_inliers_thr 15
```

---

## DINO Examples

### Baseline (default settings)
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth
```

### With LM refinement + cascading inlier schedule
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --refine_lm 1 --min_inliers_schedule "11,9,8,6,4"
```

### With LM + retrimming
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --refine_lm 1 --refine_retrim 1 --refine_keep_frac 0.8 --min_inliers_schedule "11,9,8,6,4"
```

### Test split only (exclude style/adaptation images)
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --test_split
```

### Single split
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits sunlamp
```

### Custom output file
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --output results_dino_baseline.txt
```

### Domain Generalization checkpoint
```bash
python evaluate_robust.py --checkpoint outputs_domain_generalization/best_model.pth --test_split
```

### Self-training checkpoint (iteration 3)
```bash
python evaluate_robust.py --checkpoint outputs_self_training/iter3/best_model.pth --test_split
```

### MS-SSIM checkpoint
```bash
python evaluate_robust.py --checkpoint outputs_dino_msssim/best_model.pth
```

### FDA checkpoint
```bash
python evaluate_robust.py --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth
```

---

## HRNet-W32 Examples

### Colleague's full pipeline (gt_crop + crop_pnp + resize_first + argmax)
```bash
python evaluate_robust.py --checkpoint outputs_hrnet_w32/converted_model.pth --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --no_conf_filter --rmse_inliers_thr 15
```

### HRNet-W32 + MS-SSIM
```bash
python evaluate_robust.py --checkpoint outputs_hrnet_w32_msssim/best_model.pth --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --no_conf_filter --rmse_inliers_thr 15
```

### HRNet-W32 with DINO settings (YOLO crop, soft-argmax, confidence filtering)
```bash
python evaluate_robust.py --checkpoint outputs_hrnet_w32/converted_model.pth
```

---

## All CLI Arguments

### General

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | *required* | Path to model checkpoint |
| `--splits` | str (nargs+) | `val lightbox sunlamp` | Splits to evaluate |
| `--batch_size` | int | `64` | Batch size |
| `--num_workers` | int | `2` | DataLoader workers |
| `--output` | str | `<ckpt_dir>/results_robust.txt` | Output results file |
| `--test_split` | flag | off | Use `*_test.txt` subsets for lightbox/sunlamp (excludes style/adaptation images) |
| `--splits_dir` | str | `data/splits` | Directory with `*_test.txt` / `*_style.txt` files |

### Robust PnP

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--confidence` | float | `0.95` | Confidence threshold for keypoint selection |
| `--min_landmarks` | int | `8` | Min keypoints for PnP; if fewer pass threshold, take top-N |
| `--reproj_error` | float | `15.0` | RANSAC reprojection error (px) |
| `--ransac_confidence` | float | `0.99` | RANSAC confidence parameter |
| `--ransac_iterations` | int | `200` | RANSAC max iterations |
| `--no_conf_filter` | flag | off | Skip confidence pre-filtering; feed all visible keypoints to RANSAC |
| `--no_crop` | flag | off | Feed full images (no YOLO bbox crop) |

### Cascading Inlier Schedule

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--min_inliers_schedule` | str | `""` (disabled) | Comma-separated descending min inlier thresholds (e.g. `11,9,8,6,4`) |

### LM Refinement

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--refine_lm` | int | `0` | Enable Levenberg-Marquardt refinement after EPnP |
| `--refine_retrim` | int | `0` | Enable 2-pass LM with outlier retrimming |
| `--refine_keep_frac` | float | `0.8` | Fraction of inliers to keep after retrimming |
| `--refine_min_keep` | int | `6` | Min points to keep after retrimming |

### GT Crop / Colleague's Pipeline

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gt_crop` | flag | off | Crop around GT keypoints instead of YOLO bbox |
| `--crop_pnp` | flag | off | Run PnP in crop-resized space with adjusted K |
| `--resize_first` | flag | off | Resize full image to `image_size` before cropping (colleague's pipeline) |
| `--kpt_extractor` | str | `softargmax` | `softargmax` or `argmax` (hard argmax on sigmoid heatmaps) |
| `--rmse_inliers_thr` | float | `0.0` (disabled) | Reject PnP solutions with inlier reprojection RMSE > threshold (px) |
| `--min_kpt_area` | float | `0.0` (disabled) | Min bbox area of visible keypoints to accept PnP |
| `--t_ratio_max` | float | `0.0` (disabled) | Max `\|\|t_est\|\|/z_expected` ratio to accept PnP |
