# DINOv3 vs HRNet-W32 Evaluation Pipeline Comparison

Both models use the same `evaluate_robust.py` script but with very different flags. This document explains every parameter difference and why the two pipelines produce different results.

---

## 1. Model Architecture

| | DINOv3 (ViT-L/16) | HRNet-W32 |
|---|---|---|
| **Backbone** | `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L, 1024-dim) | `timm/hrnet_w32` (128-ch stride-4 features) |
| **Head** | `HeatmapKeypointHead`: 2x deconv (14->28->56) + bilinear to 64x64 + 1x1 conv | `HRNetW32KeypointHead`: 2x Conv2d(3x3) + BN + ReLU + Dropout2d(0.1) + 1x1 conv |
| **Heatmap size** | 64x64 | 128x128 |
| **Soft-argmax beta** | Learned `nn.Parameter` (initialized to `heatmap_size=64`) | Fixed `beta=50` |
| **Coord grid** | `linspace(0, 1, 64)` | `linspace(0, 127, 128) / 127` (equivalent to `linspace(0, 1, 128)`) |
| **Output** | Keypoints in [0, 1] normalized coords | Keypoints in [0, 1] normalized coords |
| **Config** | `config.yaml` | `config_hrnet_w32.yaml` |

Both heads output [0, 1] normalized coordinates and raw logit heatmaps.

---

## 2. Image Preprocessing

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **Image size** | 224x224 | 512x512 |
| **Crop method** | YOLO bbox crop (detector-dependent) | GT keypoint crop (`--gt_crop`) |
| **Resize strategy** | Crop first, then resize to 224 | Resize full image to 512 first, then crop (`--resize_first`) |
| **ImageNet norm** | Yes (`imagenet_normalize: true`, default) | No (`imagenet_normalize: false`) |
| **Heatmap sigma** | 1.5 | 2.0 |
| **Batch size** | 64 | 8 (training) |

**Key difference**: `--resize_first` changes the crop_box coordinate space. With resize-first, the crop box is in 512x512 space; without it, the crop box is in the original 1920x1200 space. This affects camera intrinsics computation.

---

## 3. Keypoint Extraction

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **Method** | `softargmax` (default) | `argmax` (`--kpt_extractor argmax`) |
| **Implementation** | Spatial softmax on logits * learned beta, weighted sum of [0,1] grid | Hard argmax on sigmoid heatmaps, then normalize to [0,1] |

From `evaluate_robust.py` lines 369-382, argmax extraction:
```python
probs = torch.sigmoid(hm)          # sigmoid, not softmax
idx = torch.argmax(flat, dim=-1)   # hard argmax
x_hm / (W_hm - 1)                 # normalize to [0, 1]
```

Soft-argmax is differentiable but can give sub-pixel precision; argmax snaps to discrete heatmap cells but is more robust to multimodal distributions.

---

## 4. Coordinate System (PnP Space)

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **PnP space** | Full-image pixels (`--crop_pnp` OFF) | Crop-resized pixels (`--crop_pnp` ON) |
| **Coord mapping** | `kp_px = pred * (crop_w, crop_h) + (x1, y1)` | `kp_px = pred * coord_scale` |
| **coord_scale** | N/A | `(heatmap_size - 1) * image_size / heatmap_size` = `127 * 512/128` = **508.0** |

The `coord_scale` formula (line 426) maps from [0,1] normalized coordinates to crop-space pixels:
- The model outputs [0, 1] (via soft-argmax over a grid of size H)
- Colleague's convention: soft-argmax produces [0, H-1] pixel coords, then `* (image_size / heatmap_size)` stretches to [0, 508]
- Our normalized output `* (H-1) * (image_size/H)` = same result

---

## 5. Camera Intrinsics

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **K matrix** | Original full-image K (from `camera.json`) | Crop-adjusted K via `compute_crop_K()` |
| **Distortion coeffs** | Full `dist_coeffs` from `camera.json` | `None` (no distortion) |
| **Resize-first scaling** | N/A | `K[0,:] *= 512/1920`, `K[1,:] *= 512/1200` before crop adjustment |

When `--crop_pnp` is ON, the pipeline:
1. If `--resize_first`: scales K to match the 512x512 resize
2. Calls `compute_crop_K(K_base, crop_box, image_size)` to shift principal point for the crop region
3. Passes `dist_coeffs=None` (distortion already negligible in crop space)

When `--crop_pnp` is OFF (DINOv3 default):
1. Uses original K directly
2. Passes full `dist_coeffs`

---

## 6. RANSAC / EPnP Parameters

| Parameter | DINOv3 (defaults) | HRNet-W32 (flags) |
|---|---|---|
| **Reprojection error** | 15.0 px | 12.0 px (`--reproj_error 12`) |
| **RANSAC iterations** | 200 | 500 (`--ransac_iterations 500`) |
| **RANSAC confidence** | 0.99 | 0.999 (`--ransac_confidence 0.999`) |
| **Cascading inlier schedule** | Disabled (empty) | `[11, 9, 8, 6, 4]` (`--min_inliers_schedule "11,9,8,6,4"`) |
| **PnP solver** | `cv2.SOLVEPNP_EPNP` | `cv2.SOLVEPNP_EPNP` (same) |

The cascading inlier schedule (lines 248-258) accepts a RANSAC solution if it has >= any threshold in the descending list. E.g., if RANSAC finds 7 inliers, it checks 11 (fail), 9 (fail), 8 (fail), 6 (pass) -> accepted. This rejects solutions with very few inliers while still allowing moderate solutions through.

---

## 7. Pose Refinement (Levenberg-Marquardt)

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **LM refinement** | OFF (`--refine_lm 0`, default) | ON (`--refine_lm 1`) |
| **LM retrim** | OFF (`--refine_retrim 0`, default) | ON (`--refine_retrim 1`) |
| **Keep fraction** | 0.8 (default) | 0.8 (default) |
| **Min keep** | 6 (default) | 6 (default) |

Two-pass LM retrim (lines 69-95):
1. First LM pass: refine RANSAC pose on all inlier points
2. Compute per-point reprojection error after pass 1
3. Keep best 80% of points (min 6)
4. Second LM pass: refine on trimmed inlier set

Acceptance condition: `t_z > 0` and all values finite (line 283).

---

## 8. Validation Gates

| Gate | DINOv3 | HRNet-W32 |
|---|---|---|
| **Inlier RMSE** | Disabled (`--rmse_inliers_thr 0`, default) | 15.0 px (`--rmse_inliers_thr 15`) |
| **Min keypoint area** | Disabled (`--min_kpt_area 0`, default) | 5000 px^2 (`--min_kpt_area 5000`) |
| **Translation ratio** | Disabled (`--t_ratio_max 0`, default) | 10x (`--t_ratio_max 10`) |

Gate behavior (lines 469-521):
- **RMSE gate**: Checks reprojection RMSE of inlier points *before* LM refinement (using base RANSAC pose). Rejects if > threshold.
- **Area gate**: Rejects if the bounding box area of inlier 2D keypoints is < threshold. Catches degenerate cases where all inliers are clustered.
- **t_ratio gate** (GT-free): Estimates expected depth from the crop box as `z_expected = f * model_extent / crop_size_px`, then computes `t_ratio = ||t_pred|| / z_expected`. First rolls back LM refinement if ratio > threshold, then rejects if ratio is still outside `[1/threshold, threshold]`. No ground truth is used.

> **Note**: The t_ratio gate was originally implemented (both in our pipeline and in the colleague's
> `eval_pnp_crop_dynamic_lm_refinement.py`) using `||t_pred|| / ||t_gt||`, which requires ground truth
> at inference time. This was replaced with the crop-based `z_expected` estimate to make the gate
> GT-free and scientifically sound. The colleague's script has a `--disable_t_ratio_gate` flag,
> confirming it was intended as a development tool, not for final reported numbers. The CLI interface
> (`--t_ratio_max 10`) is unchanged — only the internal reference value changed.

---

## 9. Confidence Filtering

| | DINOv3 | HRNet-W32 |
|---|---|---|
| **Confidence pre-filter** | ON (threshold 0.95, default) | OFF (`--no_conf_filter`) |
| **Min landmarks** | 8 (default) | 8 (default, but moot with `--no_conf_filter`) |

With confidence filtering ON (DINOv3):
1. Select visible keypoints with heatmap confidence >= 0.95
2. If fewer than 8 pass, take top 8 by confidence
3. Pass selected keypoints to RANSAC

With `--no_conf_filter` (HRNet-W32):
- All visible keypoints go directly to RANSAC (line 447: `conf = None`)
- RANSAC handles outlier rejection internally

---

## 10. Score Computation

Both pipelines use the identical SLAB score formula (`compute_pose_errors`, lines 293-320):

```
orientation_error = 2 * arccos(|dot(q_pred, q_gt)|)
position_error   = ||t_pred - t_gt|| / ||t_gt||
orient_score = 0 if orientation_error < 0.002949 rad (0.169 deg) else orientation_error
pos_score    = 0 if position_error < 0.002173 else position_error
SLAB = orient_score + pos_score
```

Final SLAB is the mean over all solved samples. Failed samples contribute to the `dropped` count but do not affect the SLAB average.

---

## 11. Command Examples

### DINOv3 (from `slurm/evaluate_all.sh`)

```bash
python evaluate_robust.py \
    --checkpoint outputs_keypoints_heatmap/best_model.pth \
    --splits val lightbox sunlamp \
    --test_split \
    --batch_size 64
```

All PnP parameters use defaults: reproj_error=15, confidence=0.95, no LM, no cascading, no validation gates.

### HRNet-W32 (from `slurm/evaluate_hrnet_w32.sh`)

```bash
python evaluate_robust.py \
    --checkpoint outputs_hrnet_w32/converted_model.pth \
    --gt_crop --crop_pnp --resize_first \
    --kpt_extractor argmax \
    --reproj_error 12 --ransac_confidence 0.999 --ransac_iterations 500 \
    --min_kpt_area 5000 --t_ratio_max 10 \
    --refine_lm 1 --refine_retrim 1 \
    --min_inliers_schedule "11,9,8,6,4" \
    --rmse_inliers_thr 15 \
    --no_conf_filter
```

### DINOv3 with HRNet-style settings (for fair comparison)

To evaluate a DINOv3 model using the same PnP post-processing as HRNet:
```bash
python evaluate_robust.py \
    --checkpoint outputs_keypoints_heatmap/best_model.pth \
    --reproj_error 12 --ransac_confidence 0.999 --ransac_iterations 500 \
    --refine_lm 1 --refine_retrim 1 \
    --min_inliers_schedule "11,9,8,6,4"
```

Note: `--gt_crop`, `--crop_pnp`, `--resize_first`, and `--no_conf_filter` are specific to matching the colleague's pipeline and may not be appropriate for DINOv3 models trained with YOLO crops.

### HRNet-W32 with DINOv3-style defaults (original evaluation, no post-processing)

```bash
python evaluate_robust.py \
    --checkpoint outputs_hrnet_w32/converted_model.pth \
    --refine_lm 0 --min_inliers_schedule ""
```

---

## 12. Results

### HRNet-W32 Sunlamp Baseline (with full post-processing)

From `outputs_hrnet_w32/results_robust.txt` (epoch 68, with GT crop + LM + cascading):

| Split | px_err | px_rmse | PCK | SLAB | Rot (deg) | Pos (%) | t (m) | Solved% |
|---|---|---|---|---|---|---|---|---|
| val | 2.11 | 3.66 | 99.8% | 0.0140 | 0.63 | 0.31% | 0.020 | 99.8% |
| lightbox | 10.70 | 32.78 | 90.9% | 0.0583 | 2.79 | 0.96% | 0.059 | 94.9% |
| sunlamp | 28.80 | 62.61 | 74.2% | 0.0962 | 4.64 | 1.52% | 0.094 | 76.0% |

---

## Summary of Key Differences

| Aspect | DINOv3 Default | HRNet-W32 Full | Impact |
|---|---|---|---|
| Image size | 224 | 512 | Higher resolution = more precise keypoints |
| Heatmap size | 64 | 128 | Finer spatial resolution for localization |
| Crop method | YOLO bbox | GT keypoints | GT crop = no detector noise |
| Keypoint extraction | Soft-argmax | Hard argmax | Argmax more robust to multimodal heatmaps |
| PnP space | Full-image pixels | Crop pixels | Crop-space avoids distortion, tighter K |
| Distortion | Full dist_coeffs | None | Crop-space renders distortion negligible |
| RANSAC reproj | 15 px | 12 px | Tighter = fewer false inliers |
| RANSAC iterations | 200 | 500 | More iterations = better RANSAC solution |
| RANSAC confidence | 0.99 | 0.999 | Higher = more thorough search |
| LM refinement | OFF | 2-pass retrim | Refines pose, removes outliers |
| Cascading inliers | OFF | [11,9,8,6,4] | Rejects weak RANSAC solutions |
| Validation gates | None | RMSE(15), area(5000), t_ratio(10) | Rejects degenerate solutions |
| Confidence filter | 0.95 threshold | Disabled | RANSAC handles outliers directly |

When comparing models fairly, either align all post-processing parameters or report results with both default and matched settings.
