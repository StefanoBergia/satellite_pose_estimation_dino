"""Train DINOv3 with Domain Generalization (DSU + MixStyle).

Wraps ViT transformer blocks with DSU and/or MixStyle modules to make
the model robust to domain shifts. Trains ONLY on synthetic data.

Usage:
    python -m domain_adaptation.option3_domain_generalization.train_dg \
        --config domain_adaptation/option3_domain_generalization/config_dg.yaml

    python -m domain_adaptation.option3_domain_generalization.train_dg \
        --config domain_adaptation/option3_domain_generalization/config_dg.yaml \
        --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model import SatellitePoseModel
from src.trainer import Trainer
from src.utils import load_pnp_data
from train import build_dataset, maybe_subset
from evaluate_robust import (
    evaluate_split,
    print_selection_report,
    print_results_table,
)

from domain_adaptation.option3_domain_generalization.dsu_module import DSU
from domain_adaptation.option3_domain_generalization.mixstyle import MixStyle


def inject_dg_modules(model, dg_config):
    """Inject DSU and/or MixStyle into ViT transformer blocks.

    Uses forward hooks on LayerNorm modules to apply perturbations
    after normalization, without modifying the original model architecture.

    Args:
        model: SatellitePoseModel with DINOv3 backbone
        dg_config: domain_generalization section of config

    Returns:
        list of hooks (keep references to prevent garbage collection)
    """
    enable_dsu = dg_config.get("enable_dsu", True)
    enable_mixstyle = dg_config.get("enable_mixstyle", True)
    n_blocks = dg_config.get("apply_to_last_n_blocks", 4)
    injection = dg_config.get("injection_point", "after_ln1")

    dsu_alpha = dg_config.get("dsu_alpha", 0.3)
    dsu_beta = dg_config.get("dsu_beta", 0.3)
    ms_prob = dg_config.get("mixstyle_prob", 0.5)
    ms_alpha = dg_config.get("mixstyle_alpha", 0.1)

    # Get ViT layers
    # DINOv3 structure: model.backbone.encoder.layer[i]
    # Each layer has: layernorm_before (ln1), attention, layernorm_after (ln2), mlp
    # But HuggingFace Dinov2Model uses: model.backbone.layer[i]
    if hasattr(model.backbone, 'encoder'):
        layers = model.backbone.encoder.layer
    else:
        layers = model.backbone.layer

    num_layers = len(layers)
    target_layers = list(range(num_layers - n_blocks, num_layers))

    print(f"Domain Generalization setup:")
    print(f"  DSU: {'ON' if enable_dsu else 'OFF'} (alpha={dsu_alpha}, beta={dsu_beta})")
    print(f"  MixStyle: {'ON' if enable_mixstyle else 'OFF'} (p={ms_prob}, alpha={ms_alpha})")
    print(f"  Applying to blocks: {target_layers} (last {n_blocks} of {num_layers})")
    print(f"  Injection point: {injection}")

    # Create DSU/MixStyle modules (put on same device as model)
    device = next(model.parameters()).device
    modules = []
    hooks = []

    for layer_idx in target_layers:
        layer = layers[layer_idx]

        # Select target LayerNorm
        if injection == "after_ln1":
            target_ln = layer.layernorm_before if hasattr(layer, 'layernorm_before') else layer.norm1
        else:
            target_ln = layer.layernorm_after if hasattr(layer, 'layernorm_after') else layer.norm2

        # Create perturbation chain
        perturbations = []
        if enable_dsu:
            dsu = DSU(alpha=dsu_alpha, beta=dsu_beta).to(device)
            perturbations.append(dsu)
            modules.append(dsu)
        if enable_mixstyle:
            ms = MixStyle(p=ms_prob, alpha=ms_alpha).to(device)
            perturbations.append(ms)
            modules.append(ms)

        if not perturbations:
            continue

        # Register forward hook on the LayerNorm
        def make_hook(perturbs):
            def hook_fn(module, input, output):
                x = output
                for p in perturbs:
                    x = p(x)
                return x
            return hook_fn

        h = target_ln.register_forward_hook(make_hook(perturbations))
        hooks.append(h)

    print(f"  Registered {len(hooks)} hooks with {len(modules)} perturbation modules")

    # Store modules on model so they participate in .train()/.eval() mode switches
    model._dg_modules = torch.nn.ModuleList(modules)

    return hooks


def main():
    parser = argparse.ArgumentParser(description="Train with Domain Generalization")
    parser.add_argument("--config", type=str,
                        default="domain_adaptation/option3_domain_generalization/config_dg.yaml")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained checkpoint for warm-start")
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    # Load DG config
    with open(args.config) as f:
        dg_config = yaml.safe_load(f)

    # Load base config
    base_config_path = dg_config.get("base_config", "config.yaml")
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Override base config with DG-specific settings
    if "train" in dg_config:
        config["train"].update(dg_config["train"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]
    subset_size = args.subset_size or config["data"].get("subset_size")

    print(f"Device: {device}")
    print(f"Mode: {mode}")

    # Build datasets (synthetic train only + eval splits)
    print("Building datasets...")
    from torch.utils.data import DataLoader

    train_dataset = maybe_subset(
        build_dataset(config, "train", is_train=True, mode=mode), subset_size
    )
    print(f"  train: {len(train_dataset)} samples")

    eval_datasets = {}
    for split in ["val", "lightbox", "sunlamp"]:
        if split in config["data"]["splits"]:
            ds = maybe_subset(
                build_dataset(config, split, is_train=False, mode=mode), subset_size
            )
            eval_datasets[split] = ds
            print(f"  {split}: {len(ds)} samples")

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    eval_loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=config["train"]["num_workers"], pin_memory=True)
        for split, ds in eval_datasets.items()
    }

    # Build model
    print("Loading model...")
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})
    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=config["model"]["freeze_backbone"],
        unfreeze_last_n_blocks=config["model"].get("unfreeze_last_n_blocks", 0),
        head_hidden_dims=config["model"]["head_hidden"],
        num_keypoints=config["data"]["num_keypoints"],
        dropout=config["model"]["dropout"],
        mode=mode,
        points_3d_path=geo_cfg.get("points_3d") if mode == "keypoint_pnp" else None,
        camera_json_path=geo_cfg.get("camera") if mode == "keypoint_pnp" else None,
        pnp_iterations=geo_cfg.get("pnp_iterations", 10),
        keypoint_head_type=config["model"].get("keypoint_head_type", "mlp"),
        heatmap_size=pose_cfg.get("heatmap_size", 64),
    )

    # Load pretrained checkpoint
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Inject DSU + MixStyle into ViT blocks
    dg_cfg = dg_config.get("domain_generalization", {})
    hooks = inject_dg_modules(model, dg_cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        config=config,
        device=device,
    )
    trainer.train()

    # --- Final robust evaluation ---
    print("\n\nRunning final robust evaluation...")
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

    min_landmarks = 8
    reproj_error = 15.0
    confidence_threshold = 0.95
    settings = {
        "min_landmarks": min_landmarks,
        "reproj_error": reproj_error,
        "confidence": confidence_threshold,
    }

    all_results = {}
    for split, loader in eval_loaders.items():
        print(f"\n  Evaluating: {split} ({len(loader.dataset)} samples)")
        kp_metrics, pnp_results, pose_errors, method_counts = evaluate_split(
            model, loader, mode, device, pnp_data,
            min_landmarks=min_landmarks,
            reproj_error=reproj_error,
            confidence_threshold=confidence_threshold,
        )

        n_total = len(pnp_results)
        n_solved = sum(1 for r in pnp_results if r["success"])
        n_dropped = n_total - n_solved

        print_selection_report(method_counts, pnp_results, confidence_threshold,
                               min_landmarks, n_total)

        solved_errors = [e for e in pose_errors if e is not None]
        if solved_errors:
            mean_slab = np.mean([e["slab"] for e in solved_errors])
            mean_ori = np.mean([e["orient_score"] for e in solved_errors])
            mean_pos = np.mean([e["pos_score"] for e in solved_errors])
            mean_rot = np.mean([e["rot_deg"] for e in solved_errors])
            mean_t = np.mean([e["pos_abs"] for e in solved_errors])
        else:
            mean_slab = mean_ori = mean_pos = mean_rot = mean_t = 0.0

        all_results[split] = {
            **kp_metrics,
            "epnp_slab": mean_slab,
            "epnp_ori": mean_ori,
            "epnp_pos": mean_pos,
            "epnp_rot": mean_rot,
            "epnp_t": mean_t,
            "solved%": n_solved / max(n_total, 1),
            "dropped": n_dropped,
        }

    epoch = config["train"]["epochs"]
    print_results_table(all_results, mode, epoch, settings)

    # Clean up hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
