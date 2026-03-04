"""Train DINOv3 with Domain Adversarial Neural Network (DANN).

Implements Option 5 domain adaptation using a Gradient Reversal Layer to make
backbone features domain-invariant across synthetic, lightbox, and sunlamp domains.

Architecture:
  - Task branch: synthetic images → backbone → keypoint head → keypoint/heatmap loss
  - Domain branch: balanced domain images → backbone → GRL → domain classifier → CE loss
  - Total loss: task_loss + lambda * domain_loss
  - Lambda follows a progressive schedule from 0 to lambda_max

Usage:
    python -m domain_adaptation.option5_dann.train_dann \\
        --config domain_adaptation/option5_dann/config_dann.yaml \\
        --pretrained outputs_dino_msssim/best_model.pth
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.losses import (
    heatmap_mse_loss,
    heatmap_msssim_loss,
    visibility_weighted_mse,
)
from src.model import SatellitePoseModel
from src.utils import compute_pixel_error, compute_pck, load_pnp_data
from train import build_dataset, maybe_subset
from evaluate_robust import evaluate_split, print_results_table, print_selection_report

from domain_adaptation.option5_dann.dann_modules import DomainClassifier, GRL
from domain_adaptation.option5_dann.domain_dataset import DomainDataset


def compute_lambda(step: int, total_steps: int, lambda_max: float) -> float:
    """Progressive DANN lambda schedule: 0 → lambda_max over total_steps."""
    p = step / max(total_steps, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def compute_task_loss(
    model_out: dict,
    batch: dict,
    device: torch.device,
    epoch: int,
    lambda_kp: float,
    lambda_heatmap: float,
    lambda_msssim: float,
    heatmap_size: int,
    heatmap_sigma: float,
    occluded_weight: float,
    msssim_warmup_epochs: int,
    msssim_rampup_epochs: int,
    msssim_win_size: int,
    msssim_num_scales,
) -> tuple[torch.Tensor, dict]:
    """Compute task loss (keypoint coord MSE + heatmap MSE + MS-SSIM)."""
    gt_kp = batch["keypoints"].to(device)
    vis = batch["visibility"].to(device)

    kp_loss = visibility_weighted_mse(model_out["keypoints"], gt_kp, vis, occluded_weight)
    total = lambda_kp * kp_loss
    log = {"kp_loss": kp_loss.item()}

    if "heatmaps" in model_out and lambda_heatmap > 0:
        hm_loss = heatmap_mse_loss(
            model_out["heatmaps"], gt_kp, vis,
            heatmap_size, heatmap_sigma, occluded_weight,
        )
        total = total + lambda_heatmap * hm_loss
        log["heatmap_loss"] = hm_loss.item()

    if "heatmaps" in model_out and lambda_msssim > 0:
        if epoch <= msssim_warmup_epochs:
            msssim_w = 0.0
        else:
            msssim_w = lambda_msssim * min(
                1.0, (epoch - msssim_warmup_epochs) / max(msssim_rampup_epochs, 1)
            )
        if msssim_w > 0:
            msssim_loss = heatmap_msssim_loss(
                model_out["heatmaps"], gt_kp, vis,
                heatmap_size, heatmap_sigma,
                win_size=msssim_win_size,
                num_scales=msssim_num_scales,
                occluded_weight=occluded_weight,
            )
            total = total + msssim_w * msssim_loss
            log["msssim_loss"] = msssim_loss.item()

    log["task_loss"] = total.item()
    return total, log


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    occluded_weight: float,
    pck_threshold: float,
) -> dict:
    """Quick keypoint-loss evaluation for checkpointing decisions."""
    model.eval()
    total_loss = total_px_err = total_pck = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        gt_kp = batch["keypoints"].to(device)
        vis = batch["visibility"].to(device)
        crop_box = batch["crop_box"].to(device)

        model_out = model(pixel_values=images)
        kp_loss = visibility_weighted_mse(model_out["keypoints"], gt_kp, vis, occluded_weight)
        px_err = compute_pixel_error(model_out["keypoints"], gt_kp, vis, crop_box)
        pck = compute_pck(model_out["keypoints"], gt_kp, vis, crop_box, pck_threshold)

        bs = images.shape[0]
        total_loss += kp_loss.item() * bs
        total_px_err += px_err.item() * bs
        total_pck += pck.item() * bs
        n += bs

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "pixel_error": total_px_err / n,
        "pck": total_pck / n,
    }


def save_checkpoint(model: torch.nn.Module, config: dict, path: Path, epoch: int):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, path)
    print(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="DANN domain adaptation training")
    parser.add_argument(
        "--config", type=str,
        default="domain_adaptation/option5_dann/config_dann.yaml",
    )
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained checkpoint for warm-start")
    parser.add_argument("--subset_size", type=int, default=None)

    # Evaluation arguments (matching evaluate_robust.py / train_dg.py)
    parser.add_argument("--gt_crop", action="store_true")
    parser.add_argument("--crop_pnp", action="store_true")
    parser.add_argument("--resize_first", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax",
                        choices=["softargmax", "argmax"])
    parser.add_argument("--reproj_error", type=float, default=15.0)
    parser.add_argument("--ransac_confidence", type=float, default=0.99)
    parser.add_argument("--ransac_iterations", type=int, default=200)
    parser.add_argument("--t_ratio_max", type=float, default=0.0)
    parser.add_argument("--min_kpt_area", type=float, default=0.0)
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0)
    parser.add_argument("--no_conf_filter", action="store_true")
    parser.add_argument("--min_inliers_schedule", type=str, default="")
    parser.add_argument("--refine_lm", type=int, default=0)
    parser.add_argument("--refine_retrim", type=int, default=0)
    args = parser.parse_args()

    schedule_str = (args.min_inliers_schedule or "").strip()
    args.min_inliers_schedule_list = (
        sorted([int(x.strip()) for x in schedule_str.split(",") if x.strip()], reverse=True)
        if schedule_str else None
    )

    # Load configs
    with open(args.config) as f:
        dann_config = yaml.safe_load(f)

    base_config_path = dann_config.get("base_config", "config.yaml")
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Override base config with DANN train settings
    if "train" in dann_config:
        config["train"].update(dann_config["train"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]
    subset_size = args.subset_size or config["data"].get("subset_size")

    print(f"Device: {device}")
    print(f"Mode: {mode}")

    # DANN hyperparameters
    dann_cfg = dann_config.get("dann", {})
    lambda_max = dann_cfg.get("lambda_max", 0.3)
    domain_classifier_hidden = dann_cfg.get("domain_classifier_hidden", 256)
    apply_schedule = dann_cfg.get("apply_lambda_schedule", True)
    feature_dim = dann_cfg.get("feature_dim", 1024)
    splits_dir = dann_cfg.get("splits_dir", "data/splits")
    classifier_warmup_epochs = dann_cfg.get("classifier_warmup_epochs", 5)

    # Loss hyperparameters from base config
    pose_cfg = config.get("pose", {})
    lambda_kp = pose_cfg.get("lambda_keypoint", 1.0)
    lambda_heatmap = pose_cfg.get("lambda_heatmap", 0.0)
    lambda_msssim = pose_cfg.get("lambda_msssim", 0.0)
    heatmap_size = pose_cfg.get("heatmap_size", 64)
    heatmap_sigma = pose_cfg.get("heatmap_sigma", 1.5)
    occluded_weight = config["train"].get("occluded_weight", 0.5)
    msssim_warmup_epochs = pose_cfg.get("msssim_warmup_epochs", 5)
    msssim_rampup_epochs = pose_cfg.get("msssim_rampup_epochs", 10)
    msssim_win_size = pose_cfg.get("msssim_win_size", 7)
    msssim_num_scales = pose_cfg.get("msssim_num_scales", None)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    # --- Datasets ---
    print("Building datasets...")
    train_dataset = maybe_subset(
        build_dataset(config, "train", is_train=True, mode=mode), subset_size
    )
    print(f"  synthetic train: {len(train_dataset)} samples")

    eval_datasets = {}
    for split in ["val", "lightbox", "sunlamp"]:
        if split in config["data"]["splits"]:
            ds = maybe_subset(
                build_dataset(config, split, is_train=False, mode=mode), subset_size
            )
            eval_datasets[split] = ds
            print(f"  {split}: {len(ds)} samples")

    domain_dataset = DomainDataset(config, style_lists_dir=splits_dir)

    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    domain_batch_size = max(batch_size // 2, 8)

    synthetic_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    domain_loader = DataLoader(
        domain_dataset, batch_size=domain_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    eval_loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
        for split, ds in eval_datasets.items()
    }

    # --- Model ---
    print("Loading model...")
    geo_cfg = config.get("geometry", {})
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
    ).to(device)

    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")
        if not missing and not unexpected:
            print("  All keys matched.")

    # Domain classifier (adversarial head, NOT saved in final checkpoint)
    domain_classifier = DomainClassifier(
        in_dim=feature_dim,
        hidden_dim=domain_classifier_hidden,
        num_domains=3,
    ).to(device)
    grl = GRL()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dc_params = sum(p.numel() for p in domain_classifier.parameters())
    print(f"  Model: {total_params:,} total params, {trainable_params:,} trainable")
    print(f"  Domain classifier: {dc_params:,} params")

    # --- Optimizer ---
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.keypoint_head.parameters())
    dc_param_list = list(domain_classifier.parameters())

    param_groups = [{"params": head_params, "lr": config["train"]["lr"]}]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": config["train"]["lr_backbone"],
        })
    param_groups.append({
        "params": dc_param_list,
        "lr": config["train"]["lr"],
    })

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config["train"]["weight_decay"]
    )

    epochs = config["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    log_interval = config["train"].get("log_interval", 50)

    # --- Training loop ---
    domain_iter = iter(domain_loader)
    total_steps = epochs * len(synthetic_loader)
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nStarting DANN training: {epochs} epochs, {len(synthetic_loader)} steps/epoch")
    print(f"Lambda: {'scheduled 0→' + str(lambda_max) if apply_schedule else str(lambda_max)}, "
          f"{total_steps} total steps")

    for epoch in range(1, epochs + 1):
        model.train()
        domain_classifier.train()
        accum = {}
        num_batches = 0

        class_names = ["synthetic", "lightbox", "sunlamp"]
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]

        pbar = tqdm(synthetic_loader, desc=f"Epoch {epoch}/{epochs}", file=sys.stdout)
        for batch in pbar:
            lambda_ = (
                compute_lambda(global_step, total_steps, lambda_max)
                if apply_schedule else lambda_max
            )
            # Freeze adversarial signal during classifier warm-up
            effective_lambda = 0.0 if epoch <= classifier_warmup_epochs else lambda_

            # --- Task branch: synthetic → keypoint loss ---
            images = batch["image"].to(device)
            model_out = model(pixel_values=images)
            task_loss, task_log = compute_task_loss(
                model_out, batch, device, epoch,
                lambda_kp, lambda_heatmap, lambda_msssim,
                heatmap_size, heatmap_sigma, occluded_weight,
                msssim_warmup_epochs, msssim_rampup_epochs,
                msssim_win_size, msssim_num_scales,
            )

            # --- Domain branch: balanced domain images → domain loss ---
            try:
                dom_batch = next(domain_iter)
            except StopIteration:
                domain_iter = iter(domain_loader)
                dom_batch = next(domain_iter)

            dom_images = dom_batch["image"].to(device)
            dom_labels = dom_batch["domain_label"].to(device)

            # CLS token features via backbone pooler_output
            dom_features = model.backbone(pixel_values=dom_images).pooler_output  # (B, 1024)
            dom_features_rev = grl(dom_features, effective_lambda)
            dom_logits = domain_classifier(dom_features_rev)
            domain_loss = F.cross_entropy(dom_logits, dom_labels)

            with torch.no_grad():
                preds = dom_logits.argmax(1)
                for c in range(3):
                    mask = dom_labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()

            total_loss = task_loss + domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            log = {
                **task_log,
                "domain_loss": domain_loss.item(),
                "total_loss": total_loss.item(),
                "lambda": effective_lambda,
            }
            for k, v in log.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({
                    "task": f"{task_loss.item():.4f}",
                    "dom": f"{domain_loss.item():.4f}",
                    "λ": f"{effective_lambda:.3f}",
                })

        scheduler.step()

        # Log epoch averages
        for k, v in accum.items():
            writer.add_scalar(f"train/{k}", v / max(num_batches, 1), epoch)
        writer.add_scalar("train/lr_head", optimizer.param_groups[0]["lr"], epoch)

        # Domain classifier accuracy (per class + overall)
        total_correct = sum(class_correct)
        total_samples = sum(class_total)
        writer.add_scalar("domain/acc_overall", total_correct / max(total_samples, 1), epoch)
        for c, name in enumerate(class_names):
            acc = class_correct[c] / max(class_total[c], 1)
            writer.add_scalar(f"domain/acc_{name}", acc, epoch)

        # Evaluation
        all_metrics = {}
        for split_name, loader in eval_loaders.items():
            metrics = evaluate(model, loader, device, occluded_weight, pck_threshold)
            all_metrics[split_name] = metrics
            for k, v in metrics.items():
                writer.add_scalar(f"{split_name}/{k}", v, epoch)
            print(
                f"  [{split_name}] loss={metrics['loss']:.5f}  "
                f"px_err={metrics['pixel_error']:.2f}  pck={metrics['pck']:.4f}"
            )

        # Save best model (task loss only, not domain loss)
        primary_split = "val" if "val" in all_metrics else next(iter(all_metrics))
        val_loss = all_metrics[primary_split]["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config, output_dir / "best_model.pth", epoch)
            print(f"  * New best model (val_loss={best_val_loss:.5f})")

        if epoch % 10 == 0:
            save_checkpoint(
                model, config,
                output_dir / f"checkpoint_epoch{epoch:03d}.pth", epoch
            )

    # Always save final model
    save_checkpoint(model, config, output_dir / "final_model.pth", epoch)
    writer.close()
    print("Training complete.")

    # --- Final robust evaluation using best checkpoint ---
    print("\n\nRunning final robust evaluation...")
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    min_landmarks = 8
    all_results = {}
    for split, loader in eval_loaders.items():
        print(f"\n  Evaluating: {split} ({len(loader.dataset)} samples)")
        kp_metrics, pnp_results, pose_errors, method_counts = evaluate_split(
            model, loader, mode, device, pnp_data,
            min_landmarks=min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=0.95,
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

        print_selection_report(method_counts, pnp_results, 0.95, min_landmarks, n_total)

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

    print_results_table(
        all_results, mode, epochs,
        {"reproj_error": args.reproj_error},
    )


if __name__ == "__main__":
    main()
