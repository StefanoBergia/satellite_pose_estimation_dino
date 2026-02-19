# Domain Adaptation Plan for DINOv3 Satellite Pose Estimation

## Context
- DINOv3 ViT-L/16 backbone, SPEED+ dataset, 11 keypoints, heatmap head
- Trained on synthetic images, evaluated on sunlamp + lightbox (real domains)
- FDA augmentation already in use during training (config.yaml)
- Tried TeleStyle + AdaIN style transfer — both failed (geometry not preserved)
- Goal: improve DINOv3 results on standard SPEED+ real-domain test sets

## Dataset Paths
- Root: `/nfs/home/caracciolo/dataset/speedplus_yolo`
- Train images: `images/train`, labels: `labels/train`
- Val images: `images/val`, labels: `labels/val`
- Sunlamp images: `images/sunlamp`, labels: `labels/sunlamp`
- Lightbox images: `images/lightbox`, labels: `labels/lightbox`
- Pose JSONs:
  - Train: `/nfs/home/caracciolo/dataset/speedplusv2/synthetic/train.json`
  - Val: `/nfs/home/caracciolo/dataset/speedplusv2/synthetic/validation.json`
  - Sunlamp: `/nfs/home/caracciolo/dataset/speedplusv2/sunlamp/test.json`
  - Lightbox: `/nfs/home/caracciolo/dataset/speedplusv2/lightbox/test.json`
- Camera: `camera.json`, 3D points: `tango3Dpoints.json`

## Existing Code
- `train.py` — main training script
- `evaluate.py` / `evaluate_robust.py` — evaluation
- `src/model.py` — SatellitePoseModel + KeypointHead + HeatmapKeypointHead
- `src/trainer.py` — training loop
- `src/losses.py` — loss functions
- `config.yaml` — hyperparameters (batch_size=64, num_workers=2, heatmap head, keypoint_pnp mode)
- `data/splits/` — FDA splits (sunlamp_style.txt, sunlamp_test.txt, lightbox_style.txt, lightbox_test.txt)

## CRITICAL: No Training on Test
- SPEED+ sunlamp/lightbox are test sets
- `data/splits/` already has style/test splits (20%/80% by default via prepare_fda_splits.py)
- For Option 2 (self-training): use ONLY `*_style.txt` images for adaptation, evaluate on `*_test.txt`
- NEVER use ground-truth labels from sunlamp/lightbox for supervised training

---

## Option 2: Self-Training with Pseudo-Labels
**Folder: `domain_adaptation/option2_self_training/`**

### Approach
1. Take best pretrained checkpoint
2. Run inference on unlabeled real images (sunlamp_style + lightbox_style splits ONLY)
3. Generate pseudo-labels (keypoint predictions) with confidence filtering
4. Fine-tune model on synthetic + pseudo-labeled real images
5. Iterate (optionally)

### Implementation Plan
- `self_train.py`: Main script
  - Load pretrained model checkpoint
  - Run inference on style-split real images to generate pseudo-labels
  - Filter by confidence (e.g., heatmap peak value threshold)
  - Create mixed dataset: synthetic (with GT labels) + real (with pseudo-labels)
  - Fine-tune with lower LR on mixed data
  - Support multiple iterations
- `config_self_train.yaml`: Config
  - `pretrained_checkpoint`: path to best model
  - `confidence_threshold`: min heatmap confidence to accept pseudo-label (e.g., 0.5)
  - `pseudo_label_weight`: loss weight for pseudo-labeled samples (e.g., 0.5)
  - `target_domains`: [sunlamp, lightbox]
  - `num_iterations`: number of self-training rounds
  - `lr`: lower than original (e.g., 1e-4 for head, 1e-6 for backbone)
- Evaluate on `*_test.txt` splits ONLY

### Key Details
- Use existing `data/splits/sunlamp_style.txt` and `lightbox_style.txt` for adaptation
- Evaluate on `data/splits/sunlamp_test.txt` and `lightbox_test.txt`
- Pseudo-labels = model's own heatmap predictions → soft-argmax coordinates
- Confidence = max heatmap activation per keypoint
- Mixed training: alternate batches or merge datasets with weighting

---

## Option 3: Domain Generalization (DSU + Feature Perturbation)
**Folder: `domain_adaptation/option3_domain_generalization/`**

### Approach
Train ONLY on synthetic data but inject feature-level perturbations to make model
domain-invariant. No target domain images needed.

### Implementation Plan
- `dsu_module.py`: Domain Shifting Uncertainty module
  - Works on ViT intermediate features (after each transformer block or after specific blocks)
  - During training: perturb feature statistics (mean/std) with Gaussian noise
    - mu_perturbed = mu + epsilon * sigma, where epsilon ~ N(0, alpha)
    - sigma_perturbed = sigma * (1 + epsilon2), where epsilon2 ~ N(0, beta)
  - During eval: no perturbation (deterministic)
  - Hyperparams: alpha (mean noise), beta (std noise), apply_to_blocks (which ViT blocks)
- `style_perturbation.py`: MixStyle / feature-level style mixing
  - Randomly swap/interpolate feature statistics between samples in a batch
  - lambda ~ Beta(alpha, alpha), mix stats between random pairs
- `train_dg.py`: Training script
  - Load base config + DG-specific config
  - Wrap ViT blocks with DSU modules
  - Train normally on synthetic data with DSU active
  - Optionally combine with MixStyle
- `config_dg.yaml`: Config
  - `dsu_alpha`: 0.1-0.5 (mean perturbation strength)
  - `dsu_beta`: 0.1-0.5 (std perturbation strength)
  - `dsu_blocks`: which ViT blocks to apply to (e.g., last 4)
  - `mixstyle_prob`: probability of applying MixStyle (e.g., 0.5)
  - `mixstyle_alpha`: Beta distribution param (e.g., 0.1)

### Key Details
- DSU paper: "Uncertainty Modeling for Out-of-Distribution Generalization" (Kang et al., 2022)
- MixStyle paper: "Domain Generalization with MixStyle" (Zhou et al., 2021)
- For ViT: apply DSU after LayerNorm in transformer blocks (perturb post-norm features)
- DINOv3 hidden_size=1024, 24 transformer blocks
- Only modify forward pass, no new datasets needed
- Compatible with existing augmentations (can stack on top of FDA)

---

## Option 4: Test-Time Adaptation (TTA)
**Folder: `domain_adaptation/option4_tta/`**

### Approach
Take a trained model and adapt it at inference time using unlabeled target images.
No retraining needed.

### Implementation Plan
- `tta.py`: Test-time adaptation engine
  - **Batch Norm / Layer Norm adaptation**: Update running statistics using target domain data
    - For ViT: collect feature statistics from target batch, update LayerNorm params
  - **Entropy minimization (TENT)**: Minimize prediction entropy on target images
    - Only update normalization layer parameters (affine: gamma, beta)
    - Loss = -sum(p * log(p)) over heatmap predictions
    - Few gradient steps (1-10) per batch
  - **MEMO**: Marginal entropy minimization with one test point
    - Apply multiple augmentations to single test image
    - Minimize entropy of averaged predictions
- `evaluate_tta.py`: Evaluation script with TTA
  - Load pretrained model
  - Before evaluation: run TTA adaptation on style-split images
  - Evaluate on test-split images
  - Compare with baseline (no TTA)
- `config_tta.yaml`: Config
  - `method`: "tent" | "norm_adapt" | "memo"
  - `num_steps`: gradient steps for TENT (e.g., 1-10)
  - `lr`: learning rate for TTA (e.g., 1e-4)
  - `batch_size`: TTA batch size
  - `augmentations`: for MEMO (list of transforms)

### Key Details
- TENT paper: "Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2021)
- MEMO paper: "Test-Time Training with Masked Autoencoders" or "MEMO: Test Time Robustness via Adaptation and Augmentation"
- For ViT LayerNorm: update gamma/beta parameters only
- Use style-split images for adaptation, test-split for evaluation
- Can be applied to ANY existing checkpoint with zero retraining
- Start with norm_adapt (simplest), then try TENT

---

## Evaluation Protocol (SAME for all options)
- Use SLAB score: `mean(2*arccos(|<q_pred,q_gt>|) + ||t_pred-t_gt||/||t_gt||)`
- Evaluate on `data/splits/sunlamp_test.txt` and `lightbox_test.txt` ONLY
- Also report: PCK@0.05, mean keypoint error, orientation error, position error
- Compare all options against baseline (current best model, no adaptation)

## How to Resume
1. Read this file: `domain_adaptation/PLAN.md`
2. Check which folders have been created and what's implemented
3. Check `domain_adaptation/STATUS.md` for progress
4. Continue implementing from where left off
