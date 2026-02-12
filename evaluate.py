"""Standalone evaluation script.

Loads a checkpoint, infers the mode and config from the checkpoint itself,
and evaluates on val, lightbox, and sunlamp splits.

Usage:
    python evaluate.py --checkpoint outputs/best_model.pth
    python evaluate.py --checkpoint outputs_head_pose_pnp/best_model.pth
    python evaluate.py --checkpoint outputs/best_model.pth --splits val lightbox
    python evaluate.py --checkpoint outputs/best_model.pth --stats
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.losses import visibility_weighted_mse
from src.utils import (
    compute_pixel_error,
    compute_pixel_rmse,
    compute_pck,
    compute_rotation_error,
    compute_translation_error,
    compute_slab_score,
    load_pnp_data,
    compute_cv_pnp_slab,
    solve_pnp_batch,
)


def _to_pixel_coords(pred, gt, crop_box):
    """Convert crop-relative [0,1] keypoints to pixel coordinates.

    Returns:
        pred_px, gt_px: both (B, K, 2) in pixel space
    """
    x1 = crop_box[:, 0:1].unsqueeze(-1)
    y1 = crop_box[:, 1:2].unsqueeze(-1)
    x2 = crop_box[:, 2:3].unsqueeze(-1)
    y2 = crop_box[:, 3:4].unsqueeze(-1)
    scale = torch.cat([x2 - x1, y2 - y1], dim=-1)
    offset = torch.cat([x1, y1], dim=-1)
    return pred * scale + offset, gt * scale + offset


def _compute_single_pnp_errors(R, t, gt_q, gt_t):
    """Compute EPnP errors for a single sample.

    Args:
        R: (3, 3) predicted rotation matrix (body-to-camera, Rodrigues convention)
        t: (3,) predicted translation
        gt_q: (4,) GT quaternion (w, x, y, z) SPEED+ convention
        gt_t: (3,) GT translation

    Returns:
        rot_err_deg, trans_err_rel, slab_score (all float)
    """
    rvec, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(rvec)
    if angle < 1e-10:
        pred_q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = rvec.flatten() / angle
        pred_q = np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])
    pred_q /= np.linalg.norm(pred_q) + 1e-10

    dot = np.abs(np.dot(pred_q, gt_q)).clip(0.0, 1.0)
    err_orient = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0 - 1e-7))
    orient_score = 0.0 if err_orient < 0.002949 else float(err_orient)

    err_pos_rel = np.linalg.norm(t - gt_t) / max(np.linalg.norm(gt_t), 1e-8)
    pos_score = 0.0 if err_pos_rel < 0.002173 else float(err_pos_rel)

    rot_err_deg = float(np.degrees(err_orient))
    trans_err_rel = float(err_pos_rel)
    slab_score = orient_score + pos_score

    return rot_err_deg, trans_err_rel, slab_score


@torch.no_grad()
def evaluate_split(model, loader, mode, device, pnp_data,
                   occluded_weight=0.5, pck_threshold=0.05,
                   collect_stats=False,
                   confidence_threshold=0.0, reproj_error=8.0,
                   min_inliers=4):
    """Run evaluation on a single split. Returns dict of averaged metrics.

    When collect_stats=True, also returns a dict of per-sample arrays for
    statistical analysis (threshold sweeps, distance-binned breakdowns).
    """
    model.eval()
    accum = {}
    num_batches = 0

    # Separate accumulators for EPnP (sum-based, not mean-of-means)
    epnp_accum = {}

    # Per-sample collectors (only when collect_stats=True)
    if collect_stats:
        all_pixel_errors = []
        all_gt_distances = []
        all_has_pose = []
        all_epnp_rot_errors = []
        all_epnp_trans_rel_errors = []
        all_epnp_slab_scores = []
        all_epnp_valid = []
        all_epnp_n_inliers = []
        all_kp_norm_dists = []
        all_kp_vis_masks = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        gt_kp = batch["keypoints"].to(device)
        vis = batch["visibility"].to(device)
        crop_box = batch["crop_box"].to(device)

        fwd_kwargs = {"pixel_values": images}
        if mode == "keypoint_pose_pnp":
            fwd_kwargs["crop_box"] = crop_box
            fwd_kwargs["img_size"] = batch["img_size"].to(device)
            fwd_kwargs["visibility"] = vis

        model_out = model(**fwd_kwargs)

        # Keypoint metrics
        kp_loss = visibility_weighted_mse(model_out["keypoints"], gt_kp, vis, occluded_weight)
        px_err = compute_pixel_error(model_out["keypoints"], gt_kp, vis, crop_box)
        px_rmse = compute_pixel_rmse(model_out["keypoints"], gt_kp, vis, crop_box)
        pck = compute_pck(model_out["keypoints"], gt_kp, vis, crop_box, pck_threshold)

        accum["loss"] = accum.get("loss", 0.0) + kp_loss.item()
        accum["px_err"] = accum.get("px_err", 0.0) + px_err.item()
        accum["px_rmse"] = accum.get("px_rmse", 0.0) + px_rmse.item()
        accum["pck"] = accum.get("pck", 0.0) + pck.item()

        # Pose metrics (direct head)
        has_pose = batch["has_pose"].to(device)
        if "direct_rotation" in model_out and has_pose.any():
            gt_q = batch["quaternion"].to(device)
            gt_t = batch["translation"].to(device)

            rot_err = compute_rotation_error(model_out["direct_rotation"], gt_q, has_pose)
            trans_err = compute_translation_error(model_out["direct_translation"], gt_t, has_pose)
            accum["rot_deg"] = accum.get("rot_deg", 0.0) + rot_err.item()
            accum["t_err"] = accum.get("t_err", 0.0) + trans_err.item()

            slab = compute_slab_score(
                model_out["direct_rotation"], model_out["direct_translation"],
                gt_q, gt_t, has_pose,
            )
            accum["slab"] = accum.get("slab", 0.0) + slab["slab_score"].item()
            accum["slab_ori"] = accum.get("slab_ori", 0.0) + slab["orientation_score"].item()
            accum["slab_pos"] = accum.get("slab_pos", 0.0) + slab["position_score"].item()

        if "pnp_rotation" in model_out and has_pose.any():
            gt_q = batch["quaternion"].to(device)
            gt_t = batch["translation"].to(device)

            pnp_rot_err = compute_rotation_error(model_out["pnp_rotation"], gt_q, has_pose)
            pnp_trans_err = compute_translation_error(model_out["pnp_translation"], gt_t, has_pose)
            accum["dpnp_rot"] = accum.get("dpnp_rot", 0.0) + pnp_rot_err.item()
            accum["dpnp_t"] = accum.get("dpnp_t", 0.0) + pnp_trans_err.item()

            pnp_slab = compute_slab_score(
                model_out["pnp_rotation"], model_out["pnp_translation"],
                gt_q, gt_t, has_pose,
            )
            accum["dpnp_slab"] = accum.get("dpnp_slab", 0.0) + pnp_slab["slab_score"].item()

        # Extract heatmap confidence (used by both EPnP metrics and stats)
        confidence = None
        if "heatmaps" in model_out:
            hm = model_out["heatmaps"].detach().cpu().numpy()  # (B, K, H, W)
            confidence = hm.max(axis=(2, 3))  # (B, K)

        # OpenCV EPnP SLAB score (works for ALL modes, including keypoint_only)
        if pnp_data is not None and has_pose.any():
            gt_q = batch["quaternion"].to(device)
            gt_t = batch["translation"].to(device)

            cv_pnp = compute_cv_pnp_slab(
                model_out["keypoints"], crop_box, vis,
                gt_q, gt_t, has_pose,
                pnp_data["points_3d"], pnp_data["camera_matrix"], pnp_data["dist_coeffs"],
                confidence=confidence,
                confidence_threshold=confidence_threshold,
                reproj_error=reproj_error,
                min_inliers=min_inliers,
            )
            for k, v in cv_pnp.items():
                epnp_accum[k] = epnp_accum.get(k, 0.0) + v

        # Collect per-sample data for statistical analysis
        if collect_stats:
            B = images.size(0)
            has_pose_np = batch["has_pose"].cpu().numpy()
            gt_t_np = batch["translation"].cpu().numpy()

            # Per-sample pixel errors and normalized keypoint distances
            pred_px, gt_px = _to_pixel_coords(model_out["keypoints"], gt_kp, crop_box)
            kp_dists = torch.norm(pred_px - gt_px, dim=-1)  # (B, K)
            crop_w = crop_box[:, 2] - crop_box[:, 0]
            crop_h = crop_box[:, 3] - crop_box[:, 1]
            diag = torch.sqrt(crop_w ** 2 + crop_h ** 2)  # (B,)
            norm_dists = kp_dists / diag.unsqueeze(1)  # (B, K)
            vis_mask = (vis > 0)  # (B, K)

            kp_dists_np = kp_dists.cpu().numpy()
            norm_dists_np = norm_dists.cpu().numpy()
            vis_mask_np = vis_mask.cpu().numpy()

            for b_idx in range(B):
                m = vis_mask_np[b_idx]
                if m.sum() > 0:
                    all_pixel_errors.append(float(kp_dists_np[b_idx][m].mean()))
                else:
                    all_pixel_errors.append(0.0)
                all_kp_norm_dists.append(norm_dists_np[b_idx])
                all_kp_vis_masks.append(vis_mask_np[b_idx])

            all_gt_distances.extend(np.linalg.norm(gt_t_np, axis=-1).tolist())
            all_has_pose.extend(has_pose_np.tolist())

            # Per-sample EPnP errors
            if pnp_data is not None:
                rotations, translations, pnp_success, batch_n_inliers = solve_pnp_batch(
                    model_out["keypoints"], crop_box, vis,
                    pnp_data["points_3d"], pnp_data["camera_matrix"],
                    pnp_data["dist_coeffs"],
                    confidence=confidence,
                    confidence_threshold=confidence_threshold,
                    reproj_error=reproj_error,
                    min_inliers=min_inliers,
                )
                gt_q_np = batch["quaternion"].cpu().numpy()

                for b_idx in range(B):
                    all_epnp_n_inliers.append(int(batch_n_inliers[b_idx]))
                    if has_pose_np[b_idx] and pnp_success[b_idx]:
                        rot_err, trans_rel, slab_s = _compute_single_pnp_errors(
                            rotations[b_idx], translations[b_idx],
                            gt_q_np[b_idx], gt_t_np[b_idx],
                        )
                        all_epnp_rot_errors.append(rot_err)
                        all_epnp_trans_rel_errors.append(trans_rel)
                        all_epnp_slab_scores.append(slab_s)
                        all_epnp_valid.append(True)
                    else:
                        all_epnp_rot_errors.append(float("nan"))
                        all_epnp_trans_rel_errors.append(float("nan"))
                        all_epnp_slab_scores.append(float("nan"))
                        all_epnp_valid.append(False)
            else:
                for _ in range(B):
                    all_epnp_rot_errors.append(float("nan"))
                    all_epnp_trans_rel_errors.append(float("nan"))
                    all_epnp_slab_scores.append(float("nan"))
                    all_epnp_valid.append(False)
                    all_epnp_n_inliers.append(0)

        num_batches += 1

    # Average per-batch metrics
    result = {k: v / max(num_batches, 1) for k, v in accum.items()}

    # Compute proper global means for EPnP metrics from accumulated sums
    n_valid = epnp_accum.get("epnp_n_valid", 0.0)
    n_total = epnp_accum.get("epnp_n_total", 0.0)
    if n_valid > 0:
        result["epnp_slab"] = epnp_accum["epnp_slab_sum"] / n_valid
        result["epnp_ori"] = epnp_accum["epnp_orient_sum"] / n_valid
        result["epnp_pos"] = epnp_accum["epnp_pos_sum"] / n_valid
        result["epnp_rot"] = epnp_accum["epnp_rot_sum"] / n_valid
        result["epnp_t"] = epnp_accum["epnp_terr_sum"] / n_valid
    if n_total > 0:
        result["epnp_ok%"] = epnp_accum["epnp_n_ok"] / n_total
        result["epnp_drop"] = int(n_total - n_valid)

    per_sample = None
    if collect_stats:
        per_sample = {
            "pixel_errors": np.array(all_pixel_errors),
            "gt_distances": np.array(all_gt_distances),
            "has_pose": np.array(all_has_pose, dtype=bool),
            "epnp_rot_errors": np.array(all_epnp_rot_errors),
            "epnp_trans_rel_errors": np.array(all_epnp_trans_rel_errors),
            "epnp_slab_scores": np.array(all_epnp_slab_scores),
            "epnp_valid": np.array(all_epnp_valid, dtype=bool),
            "epnp_n_inliers": np.array(all_epnp_n_inliers, dtype=int),
            "kp_norm_dists": np.array(all_kp_norm_dists),   # (N, K)
            "kp_vis_masks": np.array(all_kp_vis_masks),      # (N, K)
        }

    return result, per_sample


def print_results(all_results, mode, epoch):
    """Print results as a formatted table."""
    all_keys = []
    for metrics in all_results.values():
        for k in metrics:
            if k not in all_keys:
                all_keys.append(k)

    col_w = 14
    split_w = 10
    header = f"{'Split':<{split_w}}" + "".join(f"{k:>{col_w}}" for k in all_keys)
    sep = "-" * len(header)

    print(f"\nMode: {mode} | Epoch: {epoch}")
    print(sep)
    print(header)
    print(sep)

    for split, metrics in all_results.items():
        row = f"{split:<{split_w}}"
        for k in all_keys:
            if k in metrics:
                v = metrics[k]
                if k == "epnp_drop":
                    row += f"{int(v):>{col_w}d}"
                elif "ok%" in k or "pck" in k:
                    row += f"{v:>{col_w}.4f}"
                elif "deg" in k or "rot" in k:
                    row += f"{v:>{col_w}.2f}"
                elif "slab" in k:
                    row += f"{v:>{col_w}.4f}"
                else:
                    row += f"{v:>{col_w}.4f}"
            else:
                row += f"{'--':>{col_w}}"
        print(row)

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on all splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stats", action="store_true",
                        help="Generate statistical analysis plots and tables")
    parser.add_argument("--stats_dir", type=str, default=None,
                        help="Directory for stats output (default: <checkpoint_dir>/stats)")
    # PnP parameters
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Minimum heatmap confidence to include a keypoint "
                             "(default: 0.0 = disabled; heatmap softmax peaks are ~0.001-0.01)")
    parser.add_argument("--reproj_error", type=float, default=8.0,
                        help="RANSAC reprojection error threshold in pixels (default: 8.0)")
    parser.add_argument("--min_inliers", type=int, default=4,
                        help="Minimum RANSAC inliers to accept a PnP solution (default: 4)")
    parser.add_argument("--fda_test_split", action="store_true",
                        help="Use FDA test-only split for lightbox/sunlamp "
                             "(only if model was trained with FDA)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and extract config
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    epoch = ckpt.get("epoch", "?")
    mode = config["model"]["mode"]
    root = Path(config["data"]["root"])
    geo_cfg = config.get("geometry", {})

    print(f"  Mode: {mode}")
    print(f"  Epoch: {epoch}")
    print(f"  Device: {device}")

    # Stats output directory
    if args.stats:
        if args.stats_dir:
            stats_dir = Path(args.stats_dir)
        else:
            stats_dir = Path(args.checkpoint).parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Stats output: {stats_dir}")

    # Load PnP data for OpenCV-based SLAB score (works for all modes)
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        print(f"  Loading PnP data: {geo_cfg['points_3d']}, {geo_cfg['camera']}")
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
    else:
        print("  Warning: No geometry config found, OpenCV PnP SLAB score will be skipped")

    # Build model from saved config
    pose_cfg = config.get("pose", {})
    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=True,
        head_hidden_dims=config["model"]["head_hidden"],
        num_keypoints=config["data"]["num_keypoints"],
        dropout=config["model"]["dropout"],
        mode=mode,
        points_3d_path=geo_cfg.get("points_3d") if mode == "keypoint_pose_pnp" else None,
        camera_json_path=geo_cfg.get("camera") if mode == "keypoint_pose_pnp" else None,
        pnp_iterations=geo_cfg.get("pnp_iterations", 10),
        keypoint_head_type=config["model"].get("keypoint_head_type", "mlp"),
        heatmap_size=pose_cfg.get("heatmap_size", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = KeypointTransform(image_size=config["data"]["image_size"], is_train=False)
    occluded_weight = config["train"].get("occluded_weight", 0.5)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    all_results = {}
    all_per_sample = {}
    for split in args.splits:
        if split not in config["data"]["splits"]:
            print(f"  Skipping {split} (not in config)")
            continue

        split_cfg = config["data"]["splits"][split]

        # Always load pose labels for SLAB score computation via OpenCV PnP
        pose_json = None
        pose_labels = config["data"].get("pose_labels", {})
        if split in pose_labels:
            pose_json = pose_labels[split]

        # Only use FDA test split if explicitly requested
        include_list = None
        if split in ("lightbox", "sunlamp") and args.fda_test_split:
            fda_cfg = config.get("fda", {})
            splits_dir = Path(fda_cfg.get("splits_dir", "data/splits"))
            test_list_path = splits_dir / f"{split}_test.txt"
            if test_list_path.exists():
                with open(test_list_path) as f:
                    include_list = set(line.strip() for line in f if line.strip())
                print(f"  Using FDA test split: {len(include_list)} images")

        dataset = SpeedPlusKeypointDataset(
            image_dir=str(root / split_cfg["images"]),
            label_dir=str(root / split_cfg["labels"]),
            num_keypoints=config["data"]["num_keypoints"],
            bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
            transform=transform,
            pose_json=pose_json,
            include_list=include_list,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"\nEvaluating {split} ({len(dataset)} samples)...")
        metrics, per_sample = evaluate_split(
            model, loader, mode, device, pnp_data,
            occluded_weight, pck_threshold,
            collect_stats=args.stats,
            confidence_threshold=args.confidence_threshold,
            reproj_error=args.reproj_error,
            min_inliers=args.min_inliers,
        )
        all_results[split] = metrics
        if per_sample is not None:
            all_per_sample[split] = per_sample

    print_results(all_results, mode, epoch)

    if args.stats and all_per_sample:
        from src.stats import run_stats_analysis
        run_stats_analysis(all_per_sample, stats_dir)


if __name__ == "__main__":
    main()
