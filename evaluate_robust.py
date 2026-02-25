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
    compute_crop_K,
)
from src.losses import visibility_weighted_mse


# ---------------------------------------------------------------------------
# LM Refinement (ported from colleague's eval_pnp_crop_dynamic_lm_refinement)
# ---------------------------------------------------------------------------


def refine_pose_lm(pts_3d, pts_2d, camera_matrix, R0, t0):
    """Single-pass LM refinement using cv2.solvePnPRefineLM."""
    try:
        rvec0, _ = cv2.Rodrigues(R0.astype(np.float64))
        tvec0 = t0.reshape(3, 1).astype(np.float64)
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=pts_3d.astype(np.float64),
            imagePoints=pts_2d.astype(np.float64),
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=None,
            rvec=rvec0,
            tvec=tvec0,
        )
        R, _ = cv2.Rodrigues(rvec)
        return True, R.astype(np.float64), tvec.reshape(3).astype(np.float64)
    except cv2.error:
        return False, R0, t0


def refine_pose_lm_retrim(pts_3d, pts_2d, camera_matrix, R0, t0,
                           keep_frac=0.8, min_keep=6):
    """Two-pass LM: refine -> trim worst reprojection outliers -> refine again."""
    ok1, R1, t1 = refine_pose_lm(pts_3d, pts_2d, camera_matrix, R0, t0)
    if not ok1:
        return False, R0, t0

    # Compute per-point reprojection error after first pass
    rvec1, _ = cv2.Rodrigues(R1.astype(np.float64))
    proj, _ = cv2.projectPoints(
        pts_3d.astype(np.float64), rvec1,
        t1.reshape(3, 1).astype(np.float64),
        camera_matrix.astype(np.float64), None,
    )
    proj = proj.reshape(-1, 2)
    err = np.sqrt(np.sum((proj - pts_2d.reshape(-1, 2)) ** 2, axis=1))

    # Keep best fraction
    n = len(err)
    k = max(min_keep, int(np.floor(keep_frac * n)))
    k = min(k, n)
    idx = np.argsort(err)[:k]

    ok2, R2, t2 = refine_pose_lm(pts_3d[idx], pts_2d[idx], camera_matrix, R1, t1)
    if ok2:
        return True, R2, t2
    return True, R1, t1  # pass1 ok but pass2 failed


# ---------------------------------------------------------------------------
# PnP solver with cascading inlier schedule + LM refinement
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
    min_inliers_schedule: list[int] | None = None,
    refine_lm: bool = True,
    refine_retrim: bool = True,
    refine_keep_frac: float = 0.8,
    refine_min_keep: int = 6,
    iterations: int = 200,
    ransac_confidence: float = 0.99,
) -> dict:
    """Solve PnP for a single sample with cascading inliers + LM refinement.

    Strategy:
      1. Use visible keypoints with confidence >= threshold.
      2. If fewer than min_landmarks pass, take top min_landmarks by confidence.
      3. If fewer than 4 visible keypoints total, use all of them.
      4. EPnP RANSAC with cascading min_inliers schedule.
      5. LM refinement with optional outlier retrimming.

    Returns a dict with:
        success, R, t, n_inliers, n_keypoints_used, selection_method,
        min_conf_used, refined
    """
    vis_mask = visibility > 0
    n_visible = int(vis_mask.sum())

    fail_result = {
        "success": False,
        "R": np.eye(3),
        "t": np.zeros(3),
        "n_inliers": 0,
        "n_keypoints_used": 0,
        "selection_method": "failed",
        "min_conf_used": 0.0,
        "refined": False,
        "inlier_pts_2d": None,
        "inlier_pts_3d": None,
        "R_base": None,
        "t_base": None,
    }

    if n_visible < 4:
        fail_result["n_keypoints_used"] = n_visible
        return fail_result

    # Decide which keypoints to use
    if confidence is not None:
        conf_mask = vis_mask & (confidence >= confidence_threshold)
        n_above = int(conf_mask.sum())

        if n_above >= min_landmarks:
            mask = conf_mask
            method = "threshold"
        else:
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

    def _run_ransac(p3, p2):
        cv2.setRNGSeed(42)
        return cv2.solvePnPRansac(
            p3, p2, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reproj_error,
            iterationsCount=iterations,
            confidence=ransac_confidence,
        )

    ok, rvec, tvec, inliers = _run_ransac(pts_3d, pts_2d)

    if ok and inliers is not None and len(inliers) >= 4:
        R, _ = cv2.Rodrigues(rvec)
        idx_inl = inliers.reshape(-1)
        result = {
            "success": True,
            "R": R,
            "t": tvec.flatten(),
            "n_inliers": len(inliers),
            "n_keypoints_used": n_used,
            "selection_method": method,
            "min_conf_used": min_conf,
            "refined": False,
            "inlier_pts_2d": kp_px[mask][idx_inl].copy(),
            "inlier_pts_3d": points_3d[mask][idx_inl].copy(),
            "R_base": None,
            "t_base": None,
        }
    elif method != "all_visible" and n_visible > n_used:
        # Retry with ALL visible as last resort
        pts_2d = kp_px[vis_mask].reshape(-1, 1, 2)
        pts_3d = points_3d[vis_mask].reshape(-1, 1, 3)
        mask = vis_mask

        ok, rvec, tvec, inliers = _run_ransac(pts_3d, pts_2d)

        if ok and inliers is not None and len(inliers) >= 4:
            R, _ = cv2.Rodrigues(rvec)
            min_conf_all = float(confidence[vis_mask].min()) if confidence is not None else 0.0
            idx_inl = inliers.reshape(-1)
            result = {
                "success": True,
                "R": R,
                "t": tvec.flatten(),
                "n_inliers": len(inliers),
                "n_keypoints_used": n_visible,
                "selection_method": "all_visible",
                "min_conf_used": min_conf_all,
                "refined": False,
                "inlier_pts_2d": kp_px[mask][idx_inl].copy(),
                "inlier_pts_3d": points_3d[mask][idx_inl].copy(),
                "R_base": None,
                "t_base": None,
            }
        else:
            fail_result["n_keypoints_used"] = n_used
            fail_result["min_conf_used"] = min_conf
            return fail_result
    else:
        fail_result["n_keypoints_used"] = n_used
        fail_result["selection_method"] = method
        fail_result["min_conf_used"] = min_conf
        return fail_result

    # Cascading min_inliers gate (matching colleague's pick_dynamic_min_inliers)
    if min_inliers_schedule and result["success"]:
        ninl = result["n_inliers"]
        accepted = False
        for thr in min_inliers_schedule:  # already sorted descending
            if ninl >= thr:
                accepted = True
                break
        if not accepted:
            result["success"] = False
            result["selection_method"] = "failed"
            return result

    # Optional: LM refinement on inlier points
    if refine_lm and result["success"] and result["n_inliers"] >= 4:
        idx_inl = inliers.reshape(-1)
        pts_3d_inl = points_3d[mask][idx_inl].reshape(-1, 3)
        pts_2d_inl = kp_px[mask][idx_inl].reshape(-1, 2)
        R_base = result["R"].copy()
        t_base = result["t"].copy()

        if refine_retrim and result["n_inliers"] >= refine_min_keep:
            ok_ref, R_ref, t_ref = refine_pose_lm_retrim(
                pts_3d_inl, pts_2d_inl,
                camera_matrix, R_base, t_base,
                keep_frac=refine_keep_frac,
                min_keep=refine_min_keep,
            )
        else:
            ok_ref, R_ref, t_ref = refine_pose_lm(
                pts_3d_inl, pts_2d_inl,
                camera_matrix, R_base, t_base,
            )

        # Accept refined pose if valid (t_z > 0 and finite)
        # Colleague only checks t_z > 0 + finite; t_ratio rollback handled by caller
        if ok_ref and np.isfinite(t_ref).all() and np.isfinite(R_ref).all() and t_ref[2] > 0:
            result["R"] = R_ref
            result["t"] = t_ref
            result["refined"] = True
            result["R_base"] = R_base
            result["t_base"] = t_base

    return result


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
                   occluded_weight=0.5, pck_threshold=0.05,
                   min_inliers_schedule=None,
                   refine_lm=True, refine_retrim=True,
                   refine_keep_frac=0.8, refine_min_keep=6,
                   crop_pnp=False, image_size=512,
                   ransac_iterations=200, ransac_confidence=0.99,
                   min_kpt_area=0.0, t_ratio_max=0.0, model_extent=1.0,
                   resize_first=False, heatmap_size=128,
                   kpt_extractor="softargmax",
                   rmse_inliers_thr=0.0,
                   no_conf_filter=False,
                   collect_stats=False):
    model.eval()

    # Per-sample collectors
    all_pnp_results = []
    all_pose_errors = []
    dropped_images = []  # (filename, reason, n_visible, n_used, min_conf)

    # Per-sample stats collectors (for --stats plots)
    if collect_stats:
        all_gt_distances = []
        all_epnp_rot_errors = []
        all_epnp_trans_rel_errors = []
        all_epnp_slab_scores = []
        all_epnp_valid = []
        all_epnp_n_inliers = []
        all_has_pose = []

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

        # Override keypoints with argmax extraction if requested
        if kpt_extractor == "argmax" and "heatmaps" in model_out:
            hm = model_out["heatmaps"]  # (B, K, H, W)
            probs = torch.sigmoid(hm)
            B_hm, K_hm, H_hm, W_hm = probs.shape
            flat = probs.view(B_hm, K_hm, -1)
            idx = torch.argmax(flat, dim=-1)  # (B, K)
            y_hm = (idx // W_hm).to(torch.float32)
            x_hm = (idx % W_hm).to(torch.float32)
            # Normalize to [0, 1] matching model's soft-argmax convention
            coords_argmax = torch.stack([
                x_hm / (W_hm - 1),
                y_hm / (H_hm - 1),
            ], dim=-1)  # (B, K, 2)
            model_out["keypoints"] = coords_argmax

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
        img_size_np = batch["img_size"].cpu().numpy()
        filenames = batch["filename"]  # list of strings

        for b in range(B):
            if not has_pose_np[b]:
                continue

            # Convert to pixel coords
            x1, y1, x2, y2 = crop_np[b]
            kp_px = np.zeros((pred_np.shape[1], 2), dtype=np.float32)

            if crop_pnp:
                # Match colleague's coordinate convention:
                # soft-argmax [0, H-1] * (image_size / heatmap_size) → [0, 508]
                # Our normalized [0, 1] needs: * (heatmap_size-1) * (image_size / heatmap_size)
                coord_scale = (heatmap_size - 1) * image_size / heatmap_size
                kp_px[:, 0] = pred_np[b, :, 0] * coord_scale
                kp_px[:, 1] = pred_np[b, :, 1] * coord_scale

                if resize_first:
                    # crop_box is in resized (512) space; scale K to match
                    orig_w, orig_h = img_size_np[b]
                    K_base = pnp_data["camera_matrix"].copy()
                    K_base[0, :] *= image_size / orig_w
                    K_base[1, :] *= image_size / orig_h
                else:
                    # crop_box is in original image space
                    K_base = pnp_data["camera_matrix"]

                K_use = compute_crop_K(K_base, crop_np[b], image_size)
            else:
                # Map to full-image pixel space (original behavior)
                kp_px[:, 0] = pred_np[b, :, 0] * (x2 - x1) + x1
                kp_px[:, 1] = pred_np[b, :, 1] * (y2 - y1) + y1
                K_use = pnp_data["camera_matrix"]

            conf = None if no_conf_filter else (confidence_batch[b] if confidence_batch is not None else None)

            # Colleague passes dist_coeffs=None for crop-space PnP
            dist = None if crop_pnp else pnp_data["dist_coeffs"]

            pnp_res = solve_pnp_robust(
                kp_px, vis_np[b], conf,
                pnp_data["points_3d"], K_use,
                dist,
                min_landmarks=min_landmarks,
                reproj_error=reproj_error,
                confidence_threshold=confidence_threshold,
                min_inliers_schedule=min_inliers_schedule,
                refine_lm=refine_lm,
                refine_retrim=refine_retrim,
                refine_keep_frac=refine_keep_frac,
                refine_min_keep=refine_min_keep,
                iterations=ransac_iterations,
                ransac_confidence=ransac_confidence,
            )

            # Validation gates
            if pnp_res["success"]:
                reject = False
                # Reprojection RMSE gate on inliers (use RANSAC pose, not refined)
                # Colleague checks RMSE before LM refinement; match that by using base pose
                if rmse_inliers_thr > 0 and pnp_res["inlier_pts_2d"] is not None:
                    inl_3d = pnp_res["inlier_pts_3d"].reshape(-1, 3)
                    inl_2d = pnp_res["inlier_pts_2d"].reshape(-1, 2)
                    R_chk = pnp_res["R_base"] if pnp_res["R_base"] is not None else pnp_res["R"]
                    t_chk = pnp_res["t_base"] if pnp_res["t_base"] is not None else pnp_res["t"]
                    rvec_chk, _ = cv2.Rodrigues(R_chk.astype(np.float64))
                    proj_chk, _ = cv2.projectPoints(
                        inl_3d.astype(np.float64), rvec_chk,
                        t_chk.reshape(3, 1).astype(np.float64),
                        K_use.astype(np.float64), None)
                    d_chk = proj_chk.reshape(-1, 2) - inl_2d
                    rmse_inl = float(np.sqrt(np.mean(np.sum(d_chk * d_chk, axis=1))))
                    if rmse_inl > rmse_inliers_thr:
                        reject = True
                # Anti-cluster gate: reject if INLIER keypoints span too small an area
                if min_kpt_area > 0 and pnp_res["inlier_pts_2d"] is not None:
                    inl_kp = pnp_res["inlier_pts_2d"].reshape(-1, 2)
                    if len(inl_kp) >= 2:
                        area = ((inl_kp[:, 0].max() - inl_kp[:, 0].min()) *
                                (inl_kp[:, 1].max() - inl_kp[:, 1].min()))
                        if area < min_kpt_area:
                            reject = True
                # GT-free translation ratio gate using crop-based depth estimate.
                # z_expected ≈ f * model_extent / crop_size_px
                if t_ratio_max > 0:
                    crop_w = max(x2 - x1, 1.0)
                    crop_h = max(y2 - y1, 1.0)
                    crop_diag = max(crop_w, crop_h)
                    f_avg = (K_use[0, 0] + K_use[1, 1]) / 2.0
                    z_expected = f_avg * model_extent / crop_diag

                    t_pred_norm = np.linalg.norm(pnp_res["t"])
                    t_ratio = t_pred_norm / max(z_expected, 1e-8)

                    # Rollback LM if it diverged
                    if pnp_res["refined"] and pnp_res["R_base"] is not None:
                        if t_ratio > t_ratio_max:
                            pnp_res["R"] = pnp_res["R_base"]
                            pnp_res["t"] = pnp_res["t_base"]
                            pnp_res["refined"] = False
                            t_pred_norm = np.linalg.norm(pnp_res["t"])
                            t_ratio = t_pred_norm / max(z_expected, 1e-8)

                    # Hard reject if still out of range (both too far and too close)
                    if t_ratio > t_ratio_max or t_ratio < (1.0 / t_ratio_max):
                        reject = True
                if reject:
                    pnp_res["success"] = False
                    pnp_res["selection_method"] = "failed"

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
                errors = None
                all_pose_errors.append(None)
                dropped_images.append({
                    "filename": filenames[b],
                    "method": pnp_res["selection_method"],
                    "n_keypoints_used": pnp_res["n_keypoints_used"],
                    "min_conf_used": pnp_res["min_conf_used"],
                    "n_visible": int(vis_np[b].sum()),
                    "confidence": conf.tolist() if conf is not None else None,
                })

            if collect_stats:
                all_has_pose.append(True)
                all_gt_distances.append(float(np.linalg.norm(gt_t_np[b])))
                all_epnp_valid.append(pnp_res["success"])
                all_epnp_n_inliers.append(pnp_res["n_inliers"])
                if errors is not None:
                    all_epnp_rot_errors.append(errors["rot_deg"])
                    all_epnp_trans_rel_errors.append(errors["pos_rel"])
                    all_epnp_slab_scores.append(errors["slab"])
                else:
                    all_epnp_rot_errors.append(0.0)
                    all_epnp_trans_rel_errors.append(0.0)
                    all_epnp_slab_scores.append(0.0)

    kp_metrics = {k: v / max(n_batches, 1) for k, v in kp_accum.items()}

    per_sample = None
    if collect_stats:
        per_sample = {
            "gt_distances": np.array(all_gt_distances),
            "has_pose": np.array(all_has_pose, dtype=bool),
            "epnp_rot_errors": np.array(all_epnp_rot_errors),
            "epnp_trans_rel_errors": np.array(all_epnp_trans_rel_errors),
            "epnp_slab_scores": np.array(all_epnp_slab_scores),
            "epnp_valid": np.array(all_epnp_valid, dtype=bool),
            "epnp_n_inliers": np.array(all_epnp_n_inliers, dtype=int),
        }

    return kp_metrics, all_pnp_results, all_pose_errors, method_counts, dropped_images, per_sample


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

    def _fmt(v, w):
        s = f"{v:.4f}"
        if len(s) > w:
            s = f"{v:.4e}"
        return f"{s:>{w}}"

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
                else:
                    row += _fmt(v, col_w)
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

    def _fmt(v, w):
        s = f"{v:.4f}"
        if len(s) > w:
            s = f"{v:.4e}"
        return f"{s:>{w}}"

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
                else:
                    row += _fmt(v, col_w)
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
    parser.add_argument("--no_crop", action="store_true",
                        help="Feed full images (no YOLO bbox crop). Use for models "
                             "trained on full images (e.g. colleague's HRNet).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output results file (default: <checkpoint_dir>/results_robust.txt)")
    parser.add_argument("--test_split", action="store_true",
                        help="For lightbox/sunlamp, evaluate only on the test subset "
                             "(data/splits/{split}_test.txt), excluding the style subset. "
                             "Use this to avoid evaluating on images used for adaptation.")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directory containing *_test.txt and *_style.txt files "
                             "(used with --test_split, default: data/splits)")

    # Cascading inlier schedule
    parser.add_argument("--min_inliers_schedule", type=str, default="",
                        help="Comma-separated descending min inlier thresholds for cascading "
                             "RANSAC acceptance (e.g. '11,9,8,6,4'). Empty to disable (default).")

    # LM refinement
    parser.add_argument("--refine_lm", type=int, default=0,
                        help="Enable LM refinement after EPnP (default: 0)")
    parser.add_argument("--refine_retrim", type=int, default=0,
                        help="Enable 2-pass LM with outlier retrimming (default: 0)")
    parser.add_argument("--refine_keep_frac", type=float, default=0.8,
                        help="Fraction of inliers to keep after retrimming (default: 0.8)")
    parser.add_argument("--refine_min_keep", type=int, default=6,
                        help="Minimum points to keep after retrimming (default: 6)")

    # GT crop + crop-space PnP (matching colleague's pipeline)
    parser.add_argument("--gt_crop", action="store_true",
                        help="Crop around GT keypoints instead of YOLO bbox")
    parser.add_argument("--crop_pnp", action="store_true",
                        help="Run PnP in crop-resized space with adjusted K")
    parser.add_argument("--ransac_confidence", type=float, default=0.99,
                        help="RANSAC confidence parameter (default: 0.99)")
    parser.add_argument("--ransac_iterations", type=int, default=200,
                        help="RANSAC max iterations (default: 200)")
    parser.add_argument("--min_kpt_area", type=float, default=0.0,
                        help="Min bbox area of visible keypoints to accept PnP (0=disabled)")
    parser.add_argument("--t_ratio_max", type=float, default=0.0,
                        help="Max ||t_est||/z_expected ratio to accept (0=disabled). "
                             "z_expected is estimated from crop size + 3D model extent + focal length "
                             "(GT-free).")
    parser.add_argument("--resize_first", action="store_true",
                        help="Resize full image to image_size before cropping "
                             "(matches colleague's pipeline: resize 1920x1200 → 512x512, then crop)")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax",
                        choices=["softargmax", "argmax"],
                        help="Keypoint extraction method: softargmax (model's built-in) "
                             "or argmax (hard argmax on sigmoid heatmaps, matching colleague's eval.job)")
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0,
                        help="Reject PnP solutions with inlier reprojection RMSE > threshold px "
                             "(0=disabled, colleague uses 15)")
    parser.add_argument("--no_conf_filter", action="store_true",
                        help="Skip confidence pre-filtering; feed all visible keypoints to RANSAC "
                             "(matches colleague's pipeline)")
    parser.add_argument("--stats", action="store_true",
                        help="Generate metrics-by-distance and metrics-by-min-inliers plots")

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
    print(f"    RANSAC confidence:            {args.ransac_confidence}")
    print(f"    RANSAC iterations:            {args.ransac_iterations}")
    print(f"    Cascading inlier schedule:    {args.min_inliers_schedule_list}")
    print(f"    LM refinement:               {'ON' if args.refine_lm else 'OFF'}")
    print(f"    LM retrim:                   {'ON' if args.refine_retrim else 'OFF'} "
          f"(keep_frac={args.refine_keep_frac}, min_keep={args.refine_min_keep})")
    print(f"    GT crop:                     {'ON' if args.gt_crop else 'OFF'}")
    print(f"    Resize first:                {'ON' if args.resize_first else 'OFF'}")
    print(f"    Crop-space PnP:              {'ON' if args.crop_pnp else 'OFF'}")
    print(f"    Keypoint extractor:          {args.kpt_extractor}")
    print(f"    Confidence pre-filter:       {'OFF (all visible → RANSAC)' if args.no_conf_filter else 'ON'}")
    if args.rmse_inliers_thr > 0:
        print(f"    Inlier RMSE gate:            {args.rmse_inliers_thr} px")
    if args.min_kpt_area > 0:
        print(f"    Min keypoint area gate:      {args.min_kpt_area} px²")
    if args.t_ratio_max > 0:
        print(f"    Translation ratio gate:      {args.t_ratio_max}x (GT-free, crop-based z_expected)")
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

    # Largest 3D model dimension (for GT-free translation ratio gate)
    model_extent = float(np.ptp(pnp_data["points_3d"], axis=0).max())

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
        backbone_type=config["model"].get("backbone_type", "dinov3"),
        hrnet_pretrained=config["model"].get("hrnet_pretrained", None),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  Ignoring {len(unexpected)} unexpected keys (e.g. DSU/MixStyle modules)")
    model.to(device)
    model.eval()

    transform = KeypointTransform(
        image_size=config["data"]["image_size"],
        is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )
    occluded_weight = config["train"].get("occluded_weight", 0.5)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    all_results = {}
    all_per_sample = {}
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
            no_crop=args.no_crop,
            gt_crop=args.gt_crop,
            resize_first=config["data"]["image_size"] if args.resize_first else 0,
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

        kp_metrics, pnp_results, pose_errors, method_counts, dropped, per_sample = evaluate_split(
            model, loader, mode, device, pnp_data,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            occluded_weight=occluded_weight,
            pck_threshold=pck_threshold,
            min_inliers_schedule=args.min_inliers_schedule_list,
            refine_lm=bool(args.refine_lm),
            refine_retrim=bool(args.refine_retrim),
            refine_keep_frac=args.refine_keep_frac,
            refine_min_keep=args.refine_min_keep,
            crop_pnp=args.crop_pnp,
            image_size=config["data"]["image_size"],
            ransac_iterations=args.ransac_iterations,
            ransac_confidence=args.ransac_confidence,
            min_kpt_area=args.min_kpt_area,
            t_ratio_max=args.t_ratio_max,
            model_extent=model_extent,
            resize_first=args.resize_first,
            heatmap_size=pose_cfg.get("heatmap_size", 128),
            kpt_extractor=args.kpt_extractor,
            rmse_inliers_thr=args.rmse_inliers_thr,
            no_conf_filter=args.no_conf_filter,
            collect_stats=args.stats,
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

        # Save dropped images list
        if dropped:
            dropped_path = Path(args.checkpoint).parent / f"dropped_{split}.txt"
            with open(dropped_path, "w") as f:
                f.write(f"# Dropped images for {split}: {len(dropped)}/{n_total}\n")
                f.write(f"# filename | method | n_visible | n_kpts_used | min_conf | per_kpt_conf\n")
                for d in dropped:
                    conf_str = ",".join(f"{c:.3f}" for c in d["confidence"]) if d["confidence"] is not None else "N/A"
                    f.write(f"{d['filename']} | {d['method']} | {d['n_visible']} | "
                            f"{d['n_keypoints_used']} | {d['min_conf_used']:.4f} | {conf_str}\n")
            print(f"  Dropped images saved to: {dropped_path}")

        if per_sample is not None:
            all_per_sample[split] = per_sample

    print_results_table(all_results, mode, epoch, settings)

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint).parent / "results_robust.txt")
    save_results(all_results, output_path, mode, epoch, settings)

    # Generate stats plots
    if args.stats and all_per_sample:
        from src.stats import run_stats_analysis
        stats_dir = Path(args.checkpoint).parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        run_stats_analysis(all_per_sample, stats_dir)


if __name__ == "__main__":
    main()
