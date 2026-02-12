# Evaluation Metrics

## Main Results Table

| Column | Name | Description |
|--------|------|-------------|
| `loss` | Keypoint Loss | Mean visibility-weighted MSE between predicted and GT keypoints in crop-relative [0,1] coordinates. |
| `px_err` | Pixel Error | Mean Euclidean distance (in original 1920x1200 image pixels) between predicted and GT keypoints, averaged over visible keypoints. |
| `px_rmse` | Pixel RMSE | Root-mean-square of pixel distances. Penalizes outlier keypoints more than `px_err`. |
| `pck` | PCK | Percentage of Correct Keypoints. A keypoint is "correct" if its pixel error < `pck_threshold` (default 5%) of the bounding box diagonal. Range [0, 1]. |
| `epnp_slab` | **SLAB Score** | Official SPEC2021/SPEED+ competition metric. Lower is better. See formula below. |
| `epnp_ori` | Orientation Score | Orientation component of SLAB. Units: radians. |
| `epnp_pos` | Position Score | Position component of SLAB. Units: relative (dimensionless). |
| `epnp_rot` | Rotation Error | Mean rotation error in degrees between predicted and GT pose. |
| `epnp_t` | Translation Error | Mean relative translation error: `||t_pred - t_gt|| / ||t_gt||`. |
| `epnp_ok%` | PnP Success Rate | Fraction of samples where EPnP+RANSAC succeeded (returned a valid pose). |
| `epnp_drop` | PnP Drops | Number of samples where PnP failed. These are excluded from all `epnp_*` averages. |

## SLAB Score (primary metric)

The SLAB score is the official SPEC2021 metric used in the SPEED+ competition:

```
score = mean( orientation_score + position_score )

orientation_score = 2 * arccos(|<q_pred, q_gt>|)    (radians)
position_score    = ||t_pred - t_gt|| / ||t_gt||     (relative)
```

- Orientation errors below 0.169 deg (0.00295 rad) are zeroed (machine precision threshold)
- Position errors below 0.002173 are zeroed
- **Lower is better**
- Reported as mean over all samples where PnP succeeded

## How pose is estimated

This pipeline predicts **keypoints only** (2D image locations of 11 Tango spacecraft keypoints). Pose is recovered via **EPnP+RANSAC** (`cv2.solvePnPRansac`) as a post-processing step at evaluation time:

1. Predict 2D keypoints in crop-relative [0,1] coordinates
2. Convert back to full-image pixel coordinates using the crop box
3. Filter to visible keypoints (visibility > 0)
4. Solve PnP with at least 4 points, `reprojectionError=8.0 px`, 100 RANSAC iterations
5. A solution is accepted if PnP succeeds **and** has >= 4 RANSAC inliers

Samples where PnP fails are counted as "drops" and excluded from pose metric averages.

## Distance-Binned Metrics

The `--stats` flag produces metrics binned by ground-truth camera distance (3-4m, 4-5m, ..., 9-10m). This shows how performance degrades at longer ranges where the spacecraft appears smaller.

## Min Inliers Threshold Analysis

Also produced by `--stats`. Shows what the metrics would be if we required more RANSAC inliers to accept a PnP solution. Higher thresholds reject more samples (higher drop%) but the accepted solutions tend to be more accurate (lower SLAB). Useful for understanding the quality-vs-coverage tradeoff.

The default headline metrics use **min_inliers = 4** (essentially: PnP must succeed at all).
