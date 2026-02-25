"""Evaluate DINOv3 with Test-Time Adaptation on SPEED+ real domains.

Runs TTA adaptation on style-split images, then evaluates on test-split images.
Compares baseline (no TTA) vs adapted results.

Usage:
    python -m domain_adaptation.option4_tta.evaluate_tta \
        --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
        --method tent --num_steps 5

    python -m domain_adaptation.option4_tta.evaluate_tta \
        --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
        --method norm_adapt

    python -m domain_adaptation.option4_tta.evaluate_tta \
        --checkpoint outputs_keypoints_heatmap_FDA/best_model.pth \
        --method memo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.utils import load_pnp_data
from evaluate_robust import (
    evaluate_split,
    solve_pnp_robust,
    compute_pose_errors,
    print_selection_report,
    print_results_table,
)
from domain_adaptation.option4_tta.tta import TENT, NormAdapt, MEMO


def load_split_list(splits_dir, domain, split_type):
    """Load filename list from splits file."""
    path = Path(splits_dir) / f"{domain}_{split_type}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def build_dataset(config, split, include_list=None, no_crop=False,
                   gt_crop=False, gt_crop_margin=0.3, gt_crop_min_size=100,
                   resize_first=False):
    """Build evaluation dataset for a split."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][split]

    transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )

    pose_json = None
    pose_labels = config["data"].get("pose_labels", {})
    if split in pose_labels:
        pose_json = pose_labels[split]

    return SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=include_list,
        no_crop=no_crop,
        gt_crop=gt_crop,
        resize_first=config["data"]["image_size"] if resize_first else 0,
    )


def evaluate_robust_metrics(model, loader, mode, device, pnp_data,
                            min_landmarks=8, reproj_error=15.0,
                            confidence_threshold=0.95, label="",
                            occluded_weight=0.5, pck_threshold=0.05,
                            crop_pnp=False, image_size=512,
                            ransac_iterations=200, ransac_confidence=0.99,
                            min_kpt_area=0.0, t_ratio_max=0.0,
                            resize_first=False, heatmap_size=128,
                            kpt_extractor="softargmax",
                            rmse_inliers_thr=0.0, no_conf_filter=False,
                            min_inliers_schedule=None,
                            refine_lm=True, refine_retrim=True,
                            refine_keep_frac=0.8, refine_min_keep=6):
    """Run robust evaluation and return aggregated metrics dict."""
    kp_metrics, pnp_results, pose_errors, method_counts, _, _ = evaluate_split(
        model, loader, mode, device, pnp_data,
        min_landmarks=min_landmarks,
        reproj_error=reproj_error,
        confidence_threshold=confidence_threshold,
        occluded_weight=occluded_weight,
        pck_threshold=pck_threshold,
        crop_pnp=crop_pnp,
        image_size=image_size,
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence,
        min_kpt_area=min_kpt_area,
        t_ratio_max=t_ratio_max,
        resize_first=resize_first,
        heatmap_size=heatmap_size,
        kpt_extractor=kpt_extractor,
        rmse_inliers_thr=rmse_inliers_thr,
        no_conf_filter=no_conf_filter,
        min_inliers_schedule=min_inliers_schedule,
        refine_lm=refine_lm,
        refine_retrim=refine_retrim,
        refine_keep_frac=refine_keep_frac,
        refine_min_keep=refine_min_keep,
    )

    n_total = len(pnp_results)
    n_solved = sum(1 for r in pnp_results if r["success"])
    n_dropped = n_total - n_solved

    if label:
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

    result = {
        **kp_metrics,
        "epnp_slab": mean_slab,
        "epnp_ori": mean_ori,
        "epnp_pos": mean_pos,
        "epnp_rot": mean_rot,
        "epnp_t": mean_t,
        "solved%": n_solved / max(n_total, 1),
        "dropped": n_dropped,
    }
    return result


def print_comparison(baseline, adapted, domain, method):
    """Print baseline vs adapted metrics side-by-side."""
    print(f"\n{'='*60}")
    print(f"  {domain.upper()} — {method}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Adapted':>12} {'Delta':>12}")
    print(f"  {'-'*56}")

    all_keys = list(baseline.keys())
    for k in adapted:
        if k not in all_keys:
            all_keys.append(k)

    for k in all_keys:
        bv = baseline.get(k, float("nan"))
        av = adapted.get(k, float("nan"))
        delta = av - bv
        sign = "+" if delta > 0 else ""
        # For SLAB, lower is better; for PCK, higher is better
        print(f"  {k:<20} {bv:>12.4f} {av:>12.4f} {sign}{delta:>11.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Test-Time Adaptation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default="tent",
                        choices=["tent", "norm_adapt", "memo"])
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--domains", type=str, nargs="+",
                        default=["sunlamp", "lightbox"])
    parser.add_argument("--memo_augmentations", type=int, default=8)
    # Robust PnP parameters
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence threshold for keypoint selection")
    parser.add_argument("--min_landmarks", type=int, default=8,
                        help="Min keypoints for PnP; top-N fallback if fewer pass threshold")
    parser.add_argument("--reproj_error", type=float, default=15.0,
                        help="RANSAC reprojection error in pixels")
    parser.add_argument("--no_crop", action="store_true",
                        help="Feed full images (no YOLO bbox crop). Use for models "
                             "trained on full images (e.g. colleague's HRNet).")

    # GT crop + crop-space PnP (matching colleague's pipeline)
    parser.add_argument("--gt_crop", action="store_true",
                        help="Crop around GT keypoints instead of YOLO bbox")
    parser.add_argument("--crop_pnp", action="store_true",
                        help="Run PnP in crop-resized space with adjusted K")
    parser.add_argument("--resize_first", action="store_true",
                        help="Resize full image to image_size before cropping")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax",
                        choices=["softargmax", "argmax"],
                        help="Keypoint extraction method")
    parser.add_argument("--ransac_confidence", type=float, default=0.99,
                        help="RANSAC confidence parameter")
    parser.add_argument("--ransac_iterations", type=int, default=200,
                        help="RANSAC max iterations")
    parser.add_argument("--min_kpt_area", type=float, default=0.0,
                        help="Min bbox area of visible keypoints to accept PnP (0=disabled)")
    parser.add_argument("--t_ratio_max", type=float, default=0.0,
                        help="Max ||t_est||/||t_gt|| ratio to accept (0=disabled)")
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0,
                        help="Reject PnP solutions with inlier RMSE > threshold (0=disabled)")
    parser.add_argument("--no_conf_filter", action="store_true",
                        help="Skip confidence pre-filtering; feed all visible to RANSAC")

    # Cascading inlier schedule
    parser.add_argument("--min_inliers_schedule", type=str, default="",
                        help="Comma-separated descending min inlier thresholds (e.g. '11,9,8,6,4')")

    # LM refinement
    parser.add_argument("--refine_lm", type=int, default=0,
                        help="Enable LM refinement after EPnP (default: 0)")
    parser.add_argument("--refine_retrim", type=int, default=0,
                        help="Enable 2-pass LM with outlier retrimming (default: 0)")
    parser.add_argument("--refine_keep_frac", type=float, default=0.8,
                        help="Fraction of inliers to keep after retrimming")
    parser.add_argument("--refine_min_keep", type=int, default=6,
                        help="Minimum points to keep after retrimming")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    mode = config["model"]["mode"]
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})

    print(f"  Mode: {mode}, Epoch: {ckpt.get('epoch', '?')}")

    occluded_weight = config["train"].get("occluded_weight", 0.5)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    # Load PnP data
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

    # Build model once
    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=True,
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
    _, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  Ignoring {len(unexpected)} unexpected keys (e.g. DSU/MixStyle modules)")
    model.to(device)
    model.eval()

    # Build per-domain datasets
    domain_data = {}
    style_datasets = []
    for domain in args.domains:
        style_list = load_split_list(args.splits_dir, domain, "style")
        test_list = load_split_list(args.splits_dir, domain, "test")
        style_ds = build_dataset(config, domain, include_list=style_list,
                                  no_crop=args.no_crop, gt_crop=args.gt_crop,
                                  resize_first=args.resize_first)
        test_ds = build_dataset(config, domain, include_list=test_list,
                                no_crop=args.no_crop, gt_crop=args.gt_crop,
                                resize_first=args.resize_first)
        domain_data[domain] = {
            "style_ds": style_ds,
            "test_ds": test_ds,
            "style_list": style_list,
            "test_list": test_list,
        }
        style_datasets.append(style_ds)
        print(f"  {domain}: style={len(style_ds)}, test={len(test_ds)}")

    # Pool all style sets into a single loader for adaptation
    from torch.utils.data import ConcatDataset
    combined_style_dataset = ConcatDataset(style_datasets)
    combined_style_loader = DataLoader(
        combined_style_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"\n  Combined style set: {len(combined_style_dataset)} images "
          f"({' + '.join(str(len(d)) for d in style_datasets)})")

    # --- Baseline evaluation (no TTA) on each domain ---
    print(f"\n{'#'*60}")
    print(f"  Baseline evaluation (before TTA)")
    print(f"{'#'*60}")
    baseline_metrics = {}
    for domain in args.domains:
        test_loader = DataLoader(
            domain_data[domain]["test_ds"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"\n  [{domain}] baseline on test split...")
        baseline_metrics[domain] = evaluate_robust_metrics(
            model, test_loader, mode, device, pnp_data,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            label=f"{domain}_baseline",
            occluded_weight=occluded_weight,
            pck_threshold=pck_threshold,
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
            refine_keep_frac=args.refine_keep_frac,
            refine_min_keep=args.refine_min_keep,
        )

    # --- TTA adaptation on combined style set ---
    print(f"\n{'#'*60}")
    print(f"  Running {args.method} adaptation on combined style set...")
    print(f"{'#'*60}")

    if args.method == "tent":
        adapter = TENT(model, lr=args.lr, num_steps=args.num_steps)
        adapter.adapt(combined_style_loader, device, num_steps=args.num_steps)
    elif args.method == "norm_adapt":
        adapter = NormAdapt(model)
        adapter.adapt(combined_style_loader, device)
    elif args.method == "memo":
        # For MEMO: first do a TENT pass for global adaptation,
        # then per-sample MEMO during evaluation
        adapter = MEMO(
            model, lr=args.lr,
            num_augmentations=args.memo_augmentations,
            num_steps=1,
        )
        tent_adapter = TENT(model, lr=args.lr, num_steps=args.num_steps)
        tent_adapter.adapt(combined_style_loader, device, num_steps=args.num_steps)

    # --- Adapted evaluation on each domain separately ---
    print(f"\n{'#'*60}")
    print(f"  Adapted evaluation (after TTA)")
    print(f"{'#'*60}")
    for domain in args.domains:
        test_loader = DataLoader(
            domain_data[domain]["test_ds"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"\n  [{domain}] adapted evaluation on test split...")
        adapted_metrics = evaluate_robust_metrics(
            model, test_loader, mode, device, pnp_data,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            label=f"{domain}_adapted",
            occluded_weight=occluded_weight,
            pck_threshold=pck_threshold,
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
            refine_keep_frac=args.refine_keep_frac,
            refine_min_keep=args.refine_min_keep,
        )
        print_comparison(baseline_metrics[domain], adapted_metrics, domain, args.method)

    # Save adapted checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    save_path = ckpt_dir / f"tta_{args.method}_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "tta_method": args.method,
        "epoch": ckpt.get("epoch", "?"),
    }, save_path)
    print(f"\nAdapted checkpoint saved to: {save_path}")

    print("Done!")


if __name__ == "__main__":
    main()
