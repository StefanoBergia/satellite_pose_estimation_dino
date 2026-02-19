# Option 4: Test-Time Adaptation (TTA)

No retraining needed. Adapts a trained model at inference time.

## Methods
- **norm_adapt**: Update LayerNorm statistics from target domain
- **tent**: Entropy minimization on LayerNorm affine params (Wang et al., 2021)
- **memo**: Marginal entropy minimization with augmentations

## Usage

```bash
# TENT (recommended starting point)
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method tent --num_steps 5 --lr 1e-4

# Norm adaptation (simplest)
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method norm_adapt

# MEMO
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
    --method memo --memo_augmentations 8
```

## Data Protocol
- Adaptation: `data/splits/{domain}_style.txt` images (unlabeled)
- Evaluation: `data/splits/{domain}_test.txt` images (with GT)
- No training on test data
