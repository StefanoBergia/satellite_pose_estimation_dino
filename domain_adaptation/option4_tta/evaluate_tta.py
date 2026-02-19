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


def build_dataset(config, split, include_list=None):
    """Build evaluation dataset for a split."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][split]

    transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=False
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
    )


def evaluate_robust_metrics(model, loader, mode, device, pnp_data,
                            min_landmarks=8, reproj_error=15.0,
                            confidence_threshold=0.95, label="",
                            occluded_weight=0.5, pck_threshold=0.05):
    """Run robust evaluation and return aggregated metrics dict."""
    kp_metrics, pnp_results, pose_errors, method_counts = evaluate_split(
        model, loader, mode, device, pnp_data,
        min_landmarks=min_landmarks,
        reproj_error=reproj_error,
        confidence_threshold=confidence_threshold,
        occluded_weight=occluded_weight,
        pck_threshold=pck_threshold,
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
    args = parser.parse_args()

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
        style_ds = build_dataset(config, domain, include_list=style_list)
        test_ds = build_dataset(config, domain, include_list=test_list)
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
        )
        print_comparison(baseline_metrics[domain], adapted_metrics, domain, args.method)

    print("\nDone!")


if __name__ == "__main__":
    main()
