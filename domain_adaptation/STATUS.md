# Domain Adaptation Implementation Status

## Overall Progress
- [x] Plan written (PLAN.md)
- [x] Option 2: Self-Training — IMPLEMENTED
- [x] Option 3: Domain Generalization (DSU) — IMPLEMENTED
- [x] Option 4: Test-Time Adaptation (TTA) — IMPLEMENTED
- [ ] Testing & validation on SLURM

## Files Created

### Option 2: Self-Training (`option2_self_training/`)
- `self_train.py` — iterative self-training with pseudo-labels
- `pseudo_label_dataset.py` — dataset class for pseudo-labeled real images
- `config_self_train.yaml` — configuration
- `README.md` — usage instructions

### Option 3: Domain Generalization (`option3_domain_generalization/`)
- `dsu_module.py` — Domain Shifting Uncertainty module
- `mixstyle.py` — MixStyle feature statistics mixing
- `train_dg.py` — training script with DSU/MixStyle injection
- `config_dg.yaml` — configuration
- `README.md` — usage instructions

### Option 4: Test-Time Adaptation (`option4_tta/`)
- `tta.py` — TTA engine (norm_adapt, TENT, MEMO)
- `evaluate_tta.py` — evaluation with TTA
- `config_tta.yaml` — configuration
- `README.md` — usage instructions

## Next Steps
1. Start with Option 4 (TTA) — no retraining, quickest to test
2. Then try Option 3 (DSU) — requires retraining but promising
3. Then Option 2 (self-training) — most complex, needs careful confidence tuning
4. Compare all against baseline on sunlamp_test + lightbox_test splits
