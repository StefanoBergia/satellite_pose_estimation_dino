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


def _make_perturbation_hook(perturbs):
    """Create a forward hook that applies a chain of perturbation modules."""
    def hook_fn(module, input, output):
        x = output
        for p in perturbs:
            x = p(x)
        return x
    return hook_fn


def _apply_chain(perturbs, x):
    """Apply a chain of perturbation modules to a single tensor."""
    for p in perturbs:
        x = p(x)
    return x


def _make_perturbation_hook_multi(perturbs):
    """Forward hook for HRNet stages that return list-of-tensors or single tensor."""
    def hook_fn(module, input, output):
        if isinstance(output, list):
            return [_apply_chain(perturbs, t) for t in output]
        return _apply_chain(perturbs, output)
    return hook_fn


def _inject_dg_dinov3(model, dg_config):
    """Inject DSU/MixStyle into DINOv3 ViT blocks via LayerNorm hooks."""
    enable_dsu = dg_config.get("enable_dsu", True)
    enable_mixstyle = dg_config.get("enable_mixstyle", True)
    n_blocks = dg_config.get("apply_to_last_n_blocks", 4)
    injection = dg_config.get("injection_point", "after_ln1")
    dsu_alpha = dg_config.get("dsu_alpha", 0.3)
    dsu_beta = dg_config.get("dsu_beta", 0.3)
    ms_prob = dg_config.get("mixstyle_prob", 0.5)
    ms_alpha = dg_config.get("mixstyle_alpha", 0.1)

    # HuggingFace DINOv3 structure: model.backbone.layer[i] or .encoder.layer[i]
    if hasattr(model.backbone, 'encoder'):
        layers = model.backbone.encoder.layer
    else:
        layers = model.backbone.layer

    num_layers = len(layers)
    target_layers = list(range(num_layers - n_blocks, num_layers))

    print(f"Domain Generalization setup (DINOv3):")
    print(f"  DSU: {'ON' if enable_dsu else 'OFF'} (alpha={dsu_alpha}, beta={dsu_beta})")
    print(f"  MixStyle: {'ON' if enable_mixstyle else 'OFF'} (p={ms_prob}, alpha={ms_alpha})")
    print(f"  Applying to ViT blocks: {target_layers} (last {n_blocks} of {num_layers})")
    print(f"  Injection point: {injection}")

    device = next(model.parameters()).device
    modules = []
    hooks = []

    for layer_idx in target_layers:
        layer = layers[layer_idx]

        if injection == "after_ln1":
            target_ln = layer.layernorm_before if hasattr(layer, 'layernorm_before') else layer.norm1
        else:
            target_ln = layer.layernorm_after if hasattr(layer, 'layernorm_after') else layer.norm2

        perturbations = []
        if enable_dsu:
            dsu = DSU(alpha=dsu_alpha, beta=dsu_beta).to(device)
            perturbations.append(dsu)
            modules.append(dsu)
        if enable_mixstyle:
            ms = MixStyle(p=ms_prob, alpha=ms_alpha).to(device)
            perturbations.append(ms)
            modules.append(ms)

        if perturbations:
            h = target_ln.register_forward_hook(_make_perturbation_hook(perturbations))
            hooks.append(h)

    print(f"  Registered {len(hooks)} hooks with {len(modules)} perturbation modules")
    return hooks, modules


def _inject_dg_hrnet(model, dg_config):
    """Inject DSU/MixStyle into HRNet stages via stage-level hooks.

    Hooks once per stage (not per BN2d layer) to avoid cascading perturbations
    that destroy pretrained features. HRNet stages 2-4 return lists of
    multi-resolution tensors, so we use _make_perturbation_hook_multi.
    """
    enable_dsu = dg_config.get("enable_dsu", True)
    enable_mixstyle = dg_config.get("enable_mixstyle", True)
    n_stages = dg_config.get("apply_to_last_n_blocks", 2)  # HRNet has 4 stages
    dsu_alpha = dg_config.get("dsu_alpha", 0.3)
    dsu_beta = dg_config.get("dsu_beta", 0.3)
    ms_prob = dg_config.get("mixstyle_prob", 0.5)
    ms_alpha = dg_config.get("mixstyle_alpha", 0.1)

    # timm HRNet-W32 stage names (NOT "stage1"!)
    stage_names = ["layer1", "stage2", "stage3", "stage4"]
    target_stages = stage_names[-n_stages:] if n_stages > 0 else []

    print(f"Domain Generalization setup (HRNet):")
    print(f"  DSU: {'ON' if enable_dsu else 'OFF'} (alpha={dsu_alpha}, beta={dsu_beta})")
    print(f"  MixStyle: {'ON' if enable_mixstyle else 'OFF'} (p={ms_prob}, alpha={ms_alpha})")
    print(f"  Applying to HRNet stages: {target_stages}")

    device = next(model.parameters()).device
    modules = []
    hooks = []

    for stage_name in target_stages:
        stage = getattr(model.backbone, stage_name, None)
        if stage is None:
            print(f"  WARNING: stage '{stage_name}' not found on backbone, skipping")
            continue

        perturbations = []
        if enable_dsu:
            dsu = DSU(alpha=dsu_alpha, beta=dsu_beta).to(device)
            perturbations.append(dsu)
            modules.append(dsu)
        if enable_mixstyle:
            ms = MixStyle(p=ms_prob, alpha=ms_alpha).to(device)
            perturbations.append(ms)
            modules.append(ms)

        if perturbations:
            h = stage.register_forward_hook(_make_perturbation_hook_multi(perturbations))
            hooks.append(h)

    print(f"  Registered {len(hooks)} hooks with {len(modules)} perturbation modules")
    return hooks, modules


def inject_dg_modules(model, dg_config, backbone_type="dinov3"):
    """Inject DSU and/or MixStyle into backbone normalization layers.

    Dispatches to the appropriate backbone-specific injection function:
      - DINOv3: hooks onto LayerNorm after selected ViT transformer blocks
      - HRNet:  hooks onto BatchNorm2d layers in selected stages

    Args:
        model: SatellitePoseModel
        dg_config: domain_generalization section of config
        backbone_type: "dinov3" or "hrnet"

    Returns:
        list of hooks (keep references to prevent garbage collection)
    """
    if backbone_type in ("hrnet", "hrnet_w32"):
        hooks, modules = _inject_dg_hrnet(model, dg_config)
    else:
        hooks, modules = _inject_dg_dinov3(model, dg_config)

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

    # Evaluation parameters (matching evaluate_robust.py)
    parser.add_argument("--gt_crop", action="store_true",
                        help="Crop around GT keypoints instead of YOLO bbox")
    parser.add_argument("--crop_pnp", action="store_true",
                        help="Run PnP in crop-resized space with adjusted K")
    parser.add_argument("--resize_first", action="store_true",
                        help="Resize full image to image_size before cropping")
    parser.add_argument("--no_crop", action="store_true",
                        help="Feed full images (no YOLO bbox crop)")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax",
                        choices=["softargmax", "argmax"],
                        help="Keypoint extraction method")
    parser.add_argument("--reproj_error", type=float, default=15.0,
                        help="RANSAC reprojection error in pixels")
    parser.add_argument("--ransac_confidence", type=float, default=0.99,
                        help="RANSAC confidence parameter")
    parser.add_argument("--ransac_iterations", type=int, default=200,
                        help="RANSAC max iterations")
    parser.add_argument("--t_ratio_max", type=float, default=0.0,
                        help="Max ||t_est||/||t_gt|| ratio to accept (0=disabled)")
    parser.add_argument("--min_kpt_area", type=float, default=0.0,
                        help="Min bbox area of visible keypoints to accept PnP (0=disabled)")
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0,
                        help="Reject PnP solutions with inlier RMSE > threshold (0=disabled)")
    parser.add_argument("--no_conf_filter", action="store_true",
                        help="Skip confidence pre-filtering; feed all visible to RANSAC")
    parser.add_argument("--min_inliers_schedule", type=str, default="",
                        help="Comma-separated descending min inlier thresholds")
    parser.add_argument("--refine_lm", type=int, default=0,
                        help="Enable LM refinement after EPnP (default: 0)")
    parser.add_argument("--refine_retrim", type=int, default=0,
                        help="Enable 2-pass LM with outlier retrimming (default: 0)")
    args = parser.parse_args()

    # Parse cascading schedule
    schedule_str = (args.min_inliers_schedule or "").strip()
    if schedule_str:
        args.min_inliers_schedule_list = sorted(
            [int(x.strip()) for x in schedule_str.split(",") if x.strip()],
            reverse=True,
        )
    else:
        args.min_inliers_schedule_list = None

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
        backbone_type=config["model"].get("backbone_type", "dinov3"),
        hrnet_pretrained=config["model"].get("hrnet_pretrained", None),
    )

    # Load pretrained checkpoint
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")

    # Inject DSU + MixStyle into backbone normalization layers
    dg_cfg = dg_config.get("domain_generalization", {})
    backbone_type = config["model"].get("backbone_type", "dinov3")
    hooks = inject_dg_modules(model, dg_cfg, backbone_type=backbone_type)

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
    reproj_error = args.reproj_error
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
            crop_pnp=args.crop_pnp,
            image_size=config["data"]["image_size"],
            ransac_iterations=args.ransac_iterations,
            ransac_confidence=args.ransac_confidence,
            min_kpt_area=args.min_kpt_area,
            t_ratio_max=args.t_ratio_max,
            resize_first=args.resize_first,
            heatmap_size=pose_cfg.get("heatmap_size", 128),
            kpt_extractor=args.kpt_extractor,
            rmse_inliers_thr=args.rmse_inliers_thr,
            no_conf_filter=args.no_conf_filter,
            min_inliers_schedule=args.min_inliers_schedule_list,
            refine_lm=bool(args.refine_lm),
            refine_retrim=bool(args.refine_retrim),
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
