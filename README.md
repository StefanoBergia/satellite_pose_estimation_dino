# Satellite Pose Estimation with DINOv3

Keypoint regression and pose estimation model for the Tango spacecraft using the [SPEED+](https://purl.stanford.edu/wv398fc4383) dataset. A DINOv3 ViT-L/16 backbone extracts features from bbox-cropped images. Three configurable modes provide increasing capability from keypoints to full 6-DoF pose.

## Modes

Set `model.mode` in `config.yaml`:

| Mode | Heads | Losses | Augmentation |
|------|-------|--------|-------------|
| `keypoint_only` | Keypoint | L_kp | Full (flip, rotation, jitter) |
| `keypoint_pose` | Keypoint + Direct Pose | L_kp + L_pose | Photometric only (jitter) |
| `keypoint_pose_pnp` | Keypoint + Direct Pose + Diff. PnP | L_kp + L_pose + L_pnp | Photometric only + PnP warmup |

## Architecture

```
                          ┌→ Keypoint Head → 11×2 coords ──→ L_keypoint
                          │                       │
Image → Crop → DINOv3 CLS ┤                       └→ Diff. PnP → pose ──→ L_pnp
                          │
                          └→ Direct Pose Head → R (6D→3×3) + t (3D) ──→ L_direct
```

- **Backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (300M params, frozen by default)
- **Keypoint Head:** MLP → sigmoid → 11 keypoints in crop-relative [0, 1]
- **Pose Head:** MLP → 6D continuous rotation (Zhou et al., 2019) + 3D translation
- **Diff. PnP:** Iterative Procrustes on predicted 2D keypoints + known 3D model points
- **Rotation repr:** 6D continuous → Gram-Schmidt → proper rotation matrix

## Project Structure

```
├── config.yaml            # All hyperparameters and mode selection
├── train.py               # Training entry point
├── inference.py           # Visualization of keypoints and pose axes
├── requirements.txt
├── camera.json            # SPEED+ camera intrinsics
├── tango3Dpoints.json     # 11 spacecraft 3D keypoints in body frame
├── tangoPoints.mat        # Original MATLAB format
├── src/
│   ├── dataset.py         # SpeedPlusKeypointDataset (YOLO labels + pose JSON)
│   ├── transforms.py      # Augmentations (spatial auto-disabled for pose modes)
│   ├── model.py           # SatellitePoseModel (configurable heads + diff. PnP)
│   ├── losses.py          # Keypoint MSE, geodesic rotation, translation L1
│   ├── trainer.py         # Training loop, multi-loss, PnP warmup, multi-split eval
│   └── utils.py           # Metrics (pixel error, PCK, rotation/translation error)
└── slurm/
    └── train.sh           # SLURM submission template
```

## Dataset

SPEED+ images with YOLO-format keypoint labels + JSON pose labels:

| Split    | Samples | Description               |
|----------|---------|---------------------------|
| train    | 47,966  | Synthetic training images |
| val      | 11,994  | Synthetic validation      |
| lightbox | 6,740   | Real-world (lightbox)     |
| sunlamp  | 2,791   | Real-world (sunlamp)      |

**Keypoint labels** (YOLO): `class cx cy w h x1 y1 v1 ... x11 y11 v11`
- Coordinates normalized [0, 1] relative to full image (1920x1200, grayscale)
- Visibility: 0 = not labeled, 1 = occluded, 2 = visible

**Pose labels** (JSON): `{"filename", "q_vbs2tango_true": [x,y,z,w], "r_Vo2To_vbs_true": [x,y,z]}`

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Keypoint only (default)
python train.py --config config.yaml

# Quick test with a small subset (100 samples per split)
python train.py --config config.yaml --subset_size 100

# Change mode in config.yaml:
#   model.mode: keypoint_pose          # adds direct pose head
#   model.mode: keypoint_pose_pnp      # adds diff. PnP on top

# SLURM (edit partition/GPU in slurm/train.sh first)
sbatch slurm/train.sh
```

### Key Config Options

```yaml
model:
  mode: keypoint_only           # keypoint_only | keypoint_pose | keypoint_pose_pnp
  freeze_backbone: true         # false to fine-tune (uses lr_backbone)

train:
  lr: 1.0e-3                   # head learning rate
  lr_backbone: 1.0e-5          # backbone lr (when unfrozen)

pose:
  lambda_keypoint: 1.0         # keypoint loss weight
  lambda_direct_pose: 1.0      # direct pose loss weight
  lambda_pnp_pose: 0.5         # PnP pose loss weight
  pnp_warmup_epochs: 10        # skip PnP loss for first N epochs
  pnp_rampup_epochs: 10        # linearly ramp PnP weight over M epochs
```

### Design Decisions

- **Spatial augmentations are automatically disabled** for `keypoint_pose` and `keypoint_pose_pnp` modes (flips/rotations would invalidate pose GT). Only photometric augmentations (color jitter) remain active.
- **PnP warmup**: early keypoints are random, making PnP gradients unstable. The PnP loss is held at zero for `pnp_warmup_epochs`, then linearly ramped over `pnp_rampup_epochs`.
- **6D rotation** avoids the discontinuity problems of quaternion regression.

## Inference & Visualization

```bash
python inference.py --checkpoint outputs/best_model.pth --split val --num_samples 20 --out_dir outputs/viz
```

- Keypoints: predicted (red) vs ground truth (green), connected by yellow lines
- Pose axes (pose modes): predicted 3D axes projected onto the image (X=red, Y=green, Z=blue)
- Use `--split lightbox` or `--split sunlamp` to inspect domain gap performance

## Metrics

**Keypoints:**
- **Pixel Error:** Mean Euclidean distance in original image coordinates (pixels)
- **PCK:** Percentage of Correct Keypoints within 5% of bbox diagonal

**Pose** (when mode != keypoint_only):
- **Rotation Error:** Geodesic distance in degrees
- **Translation Error:** Euclidean distance in meters
- **SLAB Score:** Official [SPEC2021](https://kelvins.esa.int/pose-estimation-2021/scoring/) competition metric — `score = mean(orientation_error + relative_position_error)` per image, where orientation error = `2*arccos(|<q_pred, q_gt>|)` and position error = `||t_pred - t_gt|| / ||t_gt||`, with machine-precision thresholds zeroed out. Lower is better.

## Losses

- **L_keypoint:** Visibility-weighted MSE (v=2 → weight 1.0, v=1 → 0.5, v=0 → ignored)
- **L_direct_pose:** Geodesic rotation loss + L1 translation loss
- **L_pnp_pose:** Same as direct, but applied to pose recovered from diff. PnP
- **Total:** `λ_kp * L_kp + λ_direct * L_direct + λ_pnp(epoch) * L_pnp`
