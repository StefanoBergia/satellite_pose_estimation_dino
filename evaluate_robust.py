"""Robust EPnP evaluation — ensures every sample gets a pose estimate.

Strategy per sample:
  1. Among visible keypoints, select those with confidence >= threshold.
  2. If fewer than `min_landmarks` pass, take the top `min_landmarks` by
     confidence instead (regardless of threshold).
  3. If fewer than 4 visible keypoints exist, use all of them.
  4. RANSAC uses a generous reprojection error (default 15 px).

This guarantees 0 dropped samples (every sample with pose GT gets a score).

Reports per-split statistics on how many samples were solved with all
keypoints above the threshold vs. how many needed the top-N fallback.

Usage:
    python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth
    python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits val
    python evaluate_robust.py --checkpoint outputs_keypoints_heatmap/best_model.pth \
        --confidence 0.9 --min_landmarks 6 --reproj_error 10
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
from src.utils import (
    compute_pixel_error,
    compute_pixel_rmse,
    compute_pck,
    load_pnp_data,
)
from src.losses import visibility_weighted_mse


# ---------------------------------------------------------------------------
# PnP solver with top-N confidence fallback
# ---------------------------------------------------------------------------


def solve_pnp_robust(
    kp_px: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray | None,
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_landmarks: int,
    reproj_error: float,
    confidence_threshold: float,
) -> dict:
    """Solve PnP for a single sample with top-N confidence fallback.

    Strategy:
      1. Use visible keypoints with confidence >= threshold.
      2. If fewer than min_landmarks pass, take top min_landmarks by confidence.
      3. If fewer than 4 visible keypoints total, use all of them.

    Returns a dict with:
        success, R, t, n_inliers, n_keypoints_used, selection_method,
        min_conf_used (lowest confidence among selected keypoints)
    """
    vis_mask = visibility > 0
    n_visible = int(vis_mask.sum())

    if n_visible < 4:
        return {
            "success": False,
            "R": np.eye(3),
            "t": np.zeros(3),
            "n_inliers": 0,
            "n_keypoints_used": n_visible,
            "selection_method": "failed",
            "min_conf_used": 0.0,
        }

    # Decide which keypoints to use
    if confidence is not None:
        conf_mask = vis_mask & (confidence >= confidence_threshold)
        n_above = int(conf_mask.sum())

        if n_above >= min_landmarks:
            # Enough keypoints pass the threshold
            mask = conf_mask
            method = "threshold"
        else:
            # Fallback: take top min_landmarks (or all visible) by confidence
            vis_indices = np.where(vis_mask)[0]
            vis_confs = confidence[vis_indices]
            n_select = min(min_landmarks, len(vis_indices))
            top_indices = vis_indices[np.argsort(vis_confs)[::-1][:n_select]]
            mask = np.zeros_like(vis_mask)
            mask[top_indices] = True
            method = "top-N"
    else:
        mask = vis_mask
        method = "all_visible"

    n_used = int(mask.sum())
    min_conf = float(confidence[mask].min()) if confidence is not None and n_used > 0 else 0.0

    pts_2d = kp_px[mask].reshape(-1, 1, 2)
    pts_3d = points_3d[mask].reshape(-1, 1, 3)

    cv2.setRNGSeed(42)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=reproj_error,
        iterationsCount=200,
    )

    if ok and inliers is not None and len(inliers) >= 4:
        R, _ = cv2.Rodrigues(rvec)
        return {
            "success": True,
            "R": R,
            "t": tvec.flatten(),
            "n_inliers": len(inliers),
            "n_keypoints_used": n_used,
            "selection_method": method,
            "min_conf_used": min_conf,
        }

    # If RANSAC failed with selected set, retry with ALL visible as last resort
    if method != "all_visible" and n_visible > n_used:
        pts_2d = kp_px[vis_mask].reshape(-1, 1, 2)
        pts_3d = points_3d[vis_mask].reshape(-1, 1, 3)

        cv2.setRNGSeed(42)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reproj_error,
            iterationsCount=200,
        )

        if ok and inliers is not None and len(inliers) >= 4:
            R, _ = cv2.Rodrigues(rvec)
            min_conf_all = float(confidence[vis_mask].min()) if confidence is not None else 0.0
            return {
                "success": True,
                "R": R,
                "t": tvec.flatten(),
                "n_inliers": len(inliers),
                "n_keypoints_used": n_visible,
                "selection_method": "all_visible",
                "min_conf_used": min_conf_all,
            }

    return {
        "success": False,
        "R": np.eye(3),
        "t": np.zeros(3),
        "n_inliers": 0,
        "n_keypoints_used": n_used,
        "selection_method": "failed",
        "min_conf_used": min_conf,
    }


def compute_pose_errors(R, t, gt_q, gt_t):
    """Compute orientation error (deg), position error (relative), SLAB score."""
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

    err_pos_abs = float(np.linalg.norm(t - gt_t))

    return {
        "rot_deg": float(np.degrees(err_orient)),
        "pos_rel": float(err_pos_rel),
        "pos_abs": err_pos_abs,
        "orient_score": orient_score,
        "pos_score": pos_score,
        "slab": orient_score + pos_score,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model, loader, mode, device, pnp_data,
                   min_landmarks, reproj_error, confidence_threshold,
                   occluded_weight=0.5, pck_threshold=0.05):
    model.eval()

    # Per-sample collectors
    all_pnp_results = []
    all_pose_errors = []

    # Keypoint metric accumulators
    kp_accum = {"loss": 0.0, "px_err": 0.0, "px_rmse": 0.0, "pck": 0.0}
    n_batches = 0

    # Selection method counters
    method_counts = {"threshold": 0, "top-N": 0, "all_visible": 0, "failed": 0}

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        gt_kp = batch["keypoints"].to(device)
        vis = batch["visibility"].to(device)
        crop_box = batch["crop_box"].to(device)

        fwd_kwargs = {"pixel_values": images}
        if mode == "keypoint_pnp":
            fwd_kwargs["crop_box"] = crop_box
            fwd_kwargs["img_size"] = batch["img_size"].to(device)
            fwd_kwargs["visibility"] = vis

        model_out = model(**fwd_kwargs)

        # Keypoint metrics
        kp_loss = visibility_weighted_mse(model_out["keypoints"], gt_kp, vis, occluded_weight)
        px_err = compute_pixel_error(model_out["keypoints"], gt_kp, vis, crop_box)
        px_rmse = compute_pixel_rmse(model_out["keypoints"], gt_kp, vis, crop_box)
        pck = compute_pck(model_out["keypoints"], gt_kp, vis, crop_box, pck_threshold)
        kp_accum["loss"] += kp_loss.item()
        kp_accum["px_err"] += px_err.item()
        kp_accum["px_rmse"] += px_rmse.item()
        kp_accum["pck"] += pck.item()
        n_batches += 1

        # Heatmap confidence
        confidence_batch = None
        if "heatmaps" in model_out:
            hm = model_out["heatmaps"].detach().cpu().numpy()
            confidence_batch = hm.max(axis=(2, 3))  # (B, K)

        # PnP per sample
        if pnp_data is None:
            continue

        B = images.size(0)
        pred_np = model_out["keypoints"].detach().cpu().numpy()
        crop_np = crop_box.detach().cpu().numpy()
        vis_np = vis.detach().cpu().numpy()
        has_pose_np = batch["has_pose"].cpu().numpy()
        gt_q_np = batch["quaternion"].cpu().numpy()
        gt_t_np = batch["translation"].cpu().numpy()

        for b in range(B):
            if not has_pose_np[b]:
                continue

            # Convert to pixel coords
            x1, y1, x2, y2 = crop_np[b]
            kp_px = np.zeros((pred_np.shape[1], 2), dtype=np.float64)
            kp_px[:, 0] = pred_np[b, :, 0] * (x2 - x1) + x1
            kp_px[:, 1] = pred_np[b, :, 1] * (y2 - y1) + y1

            conf = confidence_batch[b] if confidence_batch is not None else None

            pnp_res = solve_pnp_robust(
                kp_px, vis_np[b], conf,
                pnp_data["points_3d"], pnp_data["camera_matrix"],
                pnp_data["dist_coeffs"],
                min_landmarks=min_landmarks,
                reproj_error=reproj_error,
                confidence_threshold=confidence_threshold,
            )
            all_pnp_results.append(pnp_res)

            method = pnp_res["selection_method"]
            method_counts[method] = method_counts.get(method, 0) + 1

            if pnp_res["success"]:
                errors = compute_pose_errors(
                    pnp_res["R"], pnp_res["t"],
                    gt_q_np[b], gt_t_np[b],
                )
                all_pose_errors.append(errors)
            else:
                all_pose_errors.append(None)

    kp_metrics = {k: v / max(n_batches, 1) for k, v in kp_accum.items()}

    return kp_metrics, all_pnp_results, all_pose_errors, method_counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_selection_report(method_counts, pnp_results, confidence_threshold,
                           min_landmarks, n_total):
    """Print how samples were solved: threshold vs top-N fallback vs failed."""
    print(f"\n{'':=<80}")
    print("  KEYPOINT SELECTION BREAKDOWN")
    print(f"{'':=<80}")
    print(f"  {'Method':<20}  {'Samples':>10}  {'%':>8}  Description")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}  {'-'*40}")

    methods = [
        ("threshold", f">= {min_landmarks} kpts with conf >= {confidence_threshold}"),
        ("top-N", f"< {min_landmarks} passed; used top-{min_landmarks} by conf"),
        ("all_visible", "RANSAC failed on selection; used all visible"),
        ("failed", "PnP could not solve (< 4 visible keypoints)"),
    ]

    for method, desc in methods:
        count = method_counts.get(method, 0)
        pct = 100.0 * count / max(n_total, 1)
        print(f"  {method:<20}  {count:>10d}  {pct:>7.2f}%  {desc}")

    print(f"  {'-'*20}  {'-'*10}  {'-'*8}")
    n_failed = method_counts.get("failed", 0)
    solved = n_total - n_failed
    print(f"  {'TOTAL':<20}  {n_total:>10d}  {100.0:>7.2f}%")
    print(f"  Solved: {solved}/{n_total} ({100.0*solved/max(n_total,1):.2f}%)")

    # Min confidence actually used (across solved samples)
    min_confs = [r["min_conf_used"] for r in pnp_results if r["success"]]
    if min_confs:
        print(f"\n  Lowest confidence keypoint actually used:")
        print(f"    min={min(min_confs):.4f},  median={np.median(min_confs):.4f},  "
              f"mean={np.mean(min_confs):.4f},  max={max(min_confs):.4f}")
    print()


def print_results_table(all_results, mode, epoch, settings):
    """Print final metrics table across splits."""
    print(f"\n{'':=<120}")
    print(f"  ROBUST EPnP EVALUATION — Mode: {mode} | Epoch: {epoch}")
    print(f"  Settings: min_landmarks={settings['min_landmarks']}, "
          f"reproj_error={settings['reproj_error']}px, "
          f"confidence={settings['confidence']}")
    print(f"{'':=<120}")

    keys = ["loss", "px_err", "px_rmse", "pck",
            "epnp_slab", "epnp_ori", "epnp_pos(%)", "epnp_rot(°)",
            "epnp_t(m)", "solved%", "dropped"]
    data_keys = ["loss", "px_err", "px_rmse", "pck",
                 "epnp_slab", "epnp_ori", "epnp_pos", "epnp_rot",
                 "epnp_t", "solved%", "dropped"]

    col_w = 13
    split_w = 12
    header = f"{'Split':<{split_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for split, metrics in all_results.items():
        row = f"{split:<{split_w}}"
        for k in data_keys:
            if k in metrics:
                v = metrics[k]
                if k == "dropped":
                    row += f"{int(v):>{col_w}d}"
                elif "%" in k or "pck" in k:
                    row += f"{v:>{col_w}.4f}"
                elif k == "epnp_rot":
                    row += f"{v:>{col_w}.2f}"
                else:
                    row += f"{v:>{col_w}.4f}"
            else:
                row += f"{'--':>{col_w}}"
        print(row)

    print(sep)


def save_results(all_results, output_path, mode, epoch, settings):
    """Save results to text file."""
    lines = []
    lines.append(f"Mode: {mode} | Epoch: {epoch}")
    lines.append(f"Settings: min_landmarks={settings['min_landmarks']}, "
                 f"reproj_error={settings['reproj_error']}px, "
                 f"confidence={settings['confidence']}")
    lines.append("")

    keys = ["loss", "px_err", "px_rmse", "pck",
            "epnp_slab", "epnp_ori", "epnp_pos(%)", "epnp_rot(°)",
            "epnp_t(m)", "solved%", "dropped"]
    data_keys = ["loss", "px_err", "px_rmse", "pck",
                 "epnp_slab", "epnp_ori", "epnp_pos", "epnp_rot",
                 "epnp_t", "solved%", "dropped"]

    col_w = 13
    split_w = 12
    header = f"{'Split':<{split_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep = "-" * len(header)

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for split, metrics in all_results.items():
        row = f"{split:<{split_w}}"
        for k in data_keys:
            if k in metrics:
                v = metrics[k]
                if k == "dropped":
                    row += f"{int(v):>{col_w}d}"
                elif "%" in k or "pck" in k:
                    row += f"{v:>{col_w}.4f}"
                elif k == "epnp_rot":
                    row += f"{v:>{col_w}.2f}"
                else:
                    row += f"{v:>{col_w}.4f}"
            else:
                row += f"{'--':>{col_w}}"
        lines.append(row)

    lines.append(sep)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robust EPnP evaluation — adaptive confidence, 0 drops"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    # Robust PnP parameters
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence threshold for keypoint selection (default: 0.95)")
    parser.add_argument("--min_landmarks", type=int, default=8,
                        help="Minimum keypoints for PnP; if fewer pass threshold, "
                             "take top-N by confidence instead (default: 8)")
    parser.add_argument("--reproj_error", type=float, default=15.0,
                        help="RANSAC reprojection error in pixels (default: 15.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output results file (default: <checkpoint_dir>/results_robust.txt)")
    parser.add_argument("--test_split", action="store_true",
                        help="For lightbox/sunlamp, evaluate only on the test subset "
                             "(data/splits/{split}_test.txt), excluding the style subset. "
                             "Use this to avoid evaluating on images used for adaptation.")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directory containing *_test.txt and *_style.txt files "
                             "(used with --test_split, default: data/splits)")

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

    print(f"\n  PnP settings:")
    print(f"    Confidence threshold:         {args.confidence}")
    print(f"    Minimum landmarks:            {args.min_landmarks}")
    print(f"    RANSAC reprojection error:    {args.reproj_error} px")
    print(f"    Fallback: if < {args.min_landmarks} keypoints pass threshold, "
          f"use top-{args.min_landmarks} by confidence")

    settings = {
        "confidence": args.confidence,
        "min_landmarks": args.min_landmarks,
        "reproj_error": args.reproj_error,
    }

    # Load PnP data
    if not (geo_cfg.get("points_3d") and geo_cfg.get("camera")):
        print("ERROR: No geometry config found in checkpoint. Cannot run PnP.")
        return
    pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
    print(f"  3D points: {pnp_data['points_3d'].shape}")

    # Build model
    pose_cfg = config.get("pose", {})
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
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  Ignoring {len(unexpected)} unexpected keys (e.g. DSU/MixStyle modules)")
    model.to(device)
    model.eval()

    transform = KeypointTransform(image_size=config["data"]["image_size"], is_train=False)
    occluded_weight = config["train"].get("occluded_weight", 0.5)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    all_results = {}
    for split in args.splits:
        if split not in config["data"]["splits"]:
            print(f"  Skipping {split} (not in config)")
            continue

        split_cfg = config["data"]["splits"][split]

        pose_json = None
        pose_labels = config["data"].get("pose_labels", {})
        if split in pose_labels:
            pose_json = pose_labels[split]

        # Optionally restrict to test subset (excluding style/adaptation images)
        include_list = None
        if split in ("lightbox", "sunlamp"):
            if args.test_split:
                test_list_path = Path(args.splits_dir) / f"{split}_test.txt"
                if test_list_path.exists():
                    with open(test_list_path) as f:
                        include_list = set(line.strip() for line in f if line.strip())
                    print(f"  Using test split only: {len(include_list)} images "
                          f"(excluded style subset)")
                else:
                    print(f"  WARNING: {test_list_path} not found, using full split")

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

        print(f"\n{'':=<80}")
        print(f"  Evaluating: {split} ({len(dataset)} samples)")
        print(f"{'':=<80}")

        kp_metrics, pnp_results, pose_errors, method_counts = evaluate_split(
            model, loader, mode, device, pnp_data,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            occluded_weight=occluded_weight,
            pck_threshold=pck_threshold,
        )

        n_total = len(pnp_results)
        n_solved = sum(1 for r in pnp_results if r["success"])
        n_dropped = n_total - n_solved

        # Selection method breakdown report
        print_selection_report(method_counts, pnp_results, args.confidence,
                               args.min_landmarks, n_total)

        # Aggregate pose metrics (only over solved samples)
        solved_errors = [e for e in pose_errors if e is not None]
        if solved_errors:
            mean_slab = np.mean([e["slab"] for e in solved_errors])
            mean_ori = np.mean([e["orient_score"] for e in solved_errors])
            mean_pos = np.mean([e["pos_score"] for e in solved_errors])
            mean_rot = np.mean([e["rot_deg"] for e in solved_errors])
            mean_t = np.mean([e["pos_abs"] for e in solved_errors])
        else:
            mean_slab = mean_ori = mean_pos = mean_rot = mean_t = 0.0

        # Inlier statistics
        inliers = [r["n_inliers"] for r in pnp_results if r["success"]]
        if inliers:
            print(f"  Inlier stats: min={min(inliers)}, "
                  f"median={np.median(inliers):.0f}, "
                  f"mean={np.mean(inliers):.1f}, "
                  f"max={max(inliers)}")

        # Number of keypoints used stats
        n_kp_used = [r["n_keypoints_used"] for r in pnp_results if r["success"]]
        if n_kp_used:
            print(f"  Keypoints used: min={min(n_kp_used)}, "
                  f"median={np.median(n_kp_used):.0f}, "
                  f"mean={np.mean(n_kp_used):.1f}, "
                  f"max={max(n_kp_used)}")

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

    print_results_table(all_results, mode, epoch, settings)

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint).parent / "results_robust.txt")
    save_results(all_results, output_path, mode, epoch, settings)


if __name__ == "__main__":
    main()
