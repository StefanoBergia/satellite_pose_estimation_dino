# Option 3: Domain Generalization (DSU + MixStyle)

Trains ONLY on synthetic data with feature-level perturbations to make the model robust to domain shifts. No target domain images needed.

## Methods
- **DSU**: Perturbs feature statistics (mean/std) with Gaussian noise during training
- **MixStyle**: Randomly mixes feature statistics between batch samples

## Usage

```bash
# Train from scratch with DSU + MixStyle
python -m domain_adaptation.option3_domain_generalization.train_dg \
    --config domain_adaptation/option3_domain_generalization/config_dg.yaml

# Warm-start from existing checkpoint
python -m domain_adaptation.option3_domain_generalization.train_dg \
    --config domain_adaptation/option3_domain_generalization/config_dg.yaml \
    --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
```

## How It Works
- DSU/MixStyle hooks are injected into the last N ViT transformer blocks
- During training: feature statistics are randomly perturbed/mixed
- During eval: pass-through (deterministic)
- The model learns to be invariant to style/domain-specific feature statistics

## Key Config Options (config_dg.yaml)
- `dsu_alpha/beta`: perturbation strength (0.1-0.5)
- `mixstyle_prob`: probability of applying MixStyle (0.3-0.7)
- `apply_to_last_n_blocks`: which ViT blocks to modify (default: last 4)
