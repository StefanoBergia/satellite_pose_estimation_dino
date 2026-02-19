# Option 2: Self-Training with Pseudo-Labels

Iterative self-training: generate pseudo-labels on unlabeled real images, then fine-tune on synthetic + pseudo-labeled data.

## Usage

```bash
# Run self-training (3 iterations by default)
python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train.yaml

# Override checkpoint
python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train.yaml \
    --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
```

## Data Protocol (CRITICAL)
- Pseudo-labels generated on: `data/splits/{domain}_style.txt` (20% of real images)
- Evaluation on: `data/splits/{domain}_test.txt` (80% of real images)
- Ground-truth labels from sunlamp/lightbox are NEVER used for training
- Only YOLO bboxes from real images are used (for cropping)

## Key Config Options (config_self_train.yaml)
- `confidence_threshold`: min heatmap confidence to accept pseudo-label (0.3)
- `pseudo_label_weight`: loss weight for pseudo-labeled samples (0.5)
- `num_iterations`: self-training rounds (3)
- `epochs_per_iteration`: fine-tuning epochs per round (10)
