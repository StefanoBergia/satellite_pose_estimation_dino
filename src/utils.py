import json

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw

from .model import quaternion_to_matrix


def load_pnp_data(points_3d_path: str, camera_json_path: str) -> dict:
    """Load 3D model points and camera intrinsics for PnP solving.

    Returns:
        dict with 'points_3d' (K, 3), 'camera_matrix' (3, 3), 'dist_coeffs' (5,)
    """
    with open(points_3d_path, "r") as f:
        pts_data = json.load(f)
    points_3d = np.array(pts_data["points"], dtype=np.float64)

    with open(camera_json_path, "r") as f:
        cam_data = json.load(f)
    camera_matrix = np.array(cam_data["cameraMatrix"], dtype=np.float64)
    dist_coeffs = np.array(cam_data["distCoeffs"], dtype=np.float64)

    return {
        "points_3d": points_3d,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }


def solve_pnp_batch(
    pred_kp: torch.Tensor,
    crop_box: torch.Tensor,
    visibility: torch.Tensor,
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.0,
    reproj_error: float = 8.0,
    min_inliers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run OpenCV solvePnP on a batch of predicted keypoints.

    Converts crop-relative [0,1] keypoints to full-image pixel coordinates,
    then solves PnP per sample using EPnP with RANSAC.

    Args:
        pred_kp: (B, K, 2) predicted keypoints in crop-relative [0,1]
        crop_box: (B, 4) crop boxes (x1, y1, x2, y2) in pixels
        visibility: (B, K) visibility flags
        points_3d: (K, 3) 3D model keypoints
        camera_matrix: (3, 3) camera intrinsic matrix
        dist_coeffs: (5,) distortion coefficients
        confidence: (B, K) optional per-keypoint confidence (e.g. heatmap peak)
        confidence_threshold: minimum confidence to include a keypoint (default 0.0 = disabled)
        reproj_error: RANSAC reprojection error threshold in pixels
        min_inliers: minimum number of RANSAC inliers to accept a solution

    Returns:
        rotations: (B, 3, 3) rotation matrices (identity if PnP fails)
        translations: (B, 3) translation vectors (zeros if PnP fails)
        success: (B,) bool array indicating PnP success
        n_inliers: (B,) int array with RANSAC inlier count per sample (0 if failed)
    """
    B, K, _ = pred_kp.shape
    pred_np = pred_kp.detach().cpu().numpy()
    crop_np = crop_box.detach().cpu().numpy()
    vis_np = visibility.detach().cpu().numpy()

    rotations = np.zeros((B, 3, 3), dtype=np.float64)
    translations = np.zeros((B, 3), dtype=np.float64)
    success = np.zeros(B, dtype=bool)
    n_inliers = np.zeros(B, dtype=int)

    for i in range(B):
        # Convert crop-relative to full-image pixel coordinates
        x1, y1, x2, y2 = crop_np[i]
        crop_w = x2 - x1
        crop_h = y2 - y1

        kp_px = np.zeros((K, 2), dtype=np.float64)
        kp_px[:, 0] = pred_np[i, :, 0] * crop_w + x1
        kp_px[:, 1] = pred_np[i, :, 1] * crop_h + y1

        # Filter to visible keypoints (vis > 0) and confident keypoints
        mask = vis_np[i] > 0
        if confidence is not None:
            mask = mask & (confidence[i] >= confidence_threshold)
        if mask.sum() < 4:
            # Need at least 4 points for PnP
            rotations[i] = np.eye(3)
            continue

        pts_2d = kp_px[mask].reshape(-1, 1, 2)
        pts_3d = points_3d[mask].reshape(-1, 1, 3)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reproj_error,
            iterationsCount=100,
        )

        if ok and inliers is not None and len(inliers) >= min_inliers:
            R, _ = cv2.Rodrigues(rvec)
            rotations[i] = R
            translations[i] = tvec.flatten()
            success[i] = True
            n_inliers[i] = len(inliers)
        else:
            rotations[i] = np.eye(3)

    return rotations, translations, success, n_inliers


def compute_cv_pnp_slab(
    pred_kp: torch.Tensor,
    crop_box: torch.Tensor,
    visibility: torch.Tensor,
    gt_q: torch.Tensor,
    gt_t: torch.Tensor,
    has_pose: torch.Tensor,
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.0,
    reproj_error: float = 8.0,
    min_inliers: int = 4,
) -> dict[str, float]:
    """Compute SLAB score using OpenCV PnP on predicted keypoints.

    Works for ALL modes including keypoint_only.

    Args:
        pred_kp: (B, K, 2) predicted keypoints in crop-relative [0,1]
        crop_box: (B, 4) crop boxes (x1, y1, x2, y2) in pixels
        visibility: (B, K) visibility flags
        gt_q: (B, 4) ground truth quaternion (w, x, y, z) SPEED+ convention
        gt_t: (B, 3) ground truth translation
        has_pose: (B,) bool mask for samples with pose labels
        points_3d: (K, 3) 3D model keypoints
        camera_matrix: (3, 3) camera intrinsic matrix
        dist_coeffs: (5,) distortion coefficients
        confidence: (B, K) optional per-keypoint confidence (e.g. heatmap peak)
        confidence_threshold: minimum confidence to include a keypoint (default 0.0 = disabled)
        reproj_error: RANSAC reprojection error threshold in pixels
        min_inliers: minimum number of RANSAC inliers to accept a solution

    Returns:
        dict with epnp_slab_sum, epnp_orient_sum, epnp_pos_sum,
        epnp_rot_sum, epnp_terr_sum, epnp_n_ok, epnp_n_valid, epnp_n_total
    """
    _zeros = {
        "epnp_slab_sum": 0.0, "epnp_orient_sum": 0.0, "epnp_pos_sum": 0.0,
        "epnp_rot_sum": 0.0, "epnp_terr_sum": 0.0,
        "epnp_n_ok": 0.0, "epnp_n_valid": 0.0, "epnp_n_total": 0.0,
    }

    has_np = has_pose.detach().cpu().numpy()
    if has_np.sum() == 0:
        return _zeros

    rotations, translations, pnp_success, _ = solve_pnp_batch(
        pred_kp, crop_box, visibility, points_3d, camera_matrix, dist_coeffs,
        confidence=confidence, confidence_threshold=confidence_threshold,
        reproj_error=reproj_error, min_inliers=min_inliers,
    )

    # Only evaluate where both has_pose and PnP succeeded
    valid = has_np & pnp_success
    if valid.sum() == 0:
        return {**_zeros, "epnp_n_total": float(has_np.sum())}

    gt_q_np = gt_q.detach().cpu().numpy()[valid]   # (N, 4) wxyz (SPEED+ convention)
    gt_t_np = gt_t.detach().cpu().numpy()[valid]    # (N, 3)
    pred_R = rotations[valid]                        # (N, 3, 3) body-to-camera
    pred_t = translations[valid]                     # (N, 3)

    # Convert predicted rotation matrices to quaternions (wxyz).
    # pred_R is body-to-camera from solvePnP (Rodrigues convention).
    # The SPEED+ quat2dcm formula builds the DCM in transposed form relative to
    # Rodrigues, so Rodrigues(R_body2cam) directly gives a quaternion matching
    # the gt_q convention — no transpose needed.
    pred_q = np.zeros((pred_R.shape[0], 4), dtype=np.float64)
    for i in range(pred_R.shape[0]):
        rvec, _ = cv2.Rodrigues(pred_R[i])
        angle = np.linalg.norm(rvec)
        if angle < 1e-10:
            pred_q[i] = [1, 0, 0, 0]  # wxyz identity
        else:
            axis = rvec.flatten() / angle
            pred_q[i, 0] = np.cos(angle / 2)        # w
            pred_q[i, 1:] = axis * np.sin(angle / 2)  # x, y, z
        # Normalize
        pred_q[i] /= np.linalg.norm(pred_q[i]) + 1e-10

    # Orientation error: 2 * arccos(|<q_pred, q_gt>|)
    dot = np.abs(np.sum(pred_q * gt_q_np, axis=-1)).clip(0.0, 1.0)
    err_orient = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0 - 1e-7))  # radians
    orient_score = np.where(err_orient < 0.002949, 0.0, err_orient)

    # Position error: ||t_pred - t_gt|| / ||t_gt||
    err_pos = np.linalg.norm(pred_t - gt_t_np, axis=-1) / np.maximum(np.linalg.norm(gt_t_np, axis=-1), 1e-8)
    pos_score = np.where(err_pos < 0.002173, 0.0, err_pos)

    # Rotation error in degrees (for logging)
    rot_err_deg = np.degrees(err_orient)

    # Translation error absolute
    trans_err = np.linalg.norm(pred_t - gt_t_np, axis=-1)

    n_valid = int(valid.sum())
    n_has_pose = int(has_np.sum())

    return {
        "epnp_slab_sum": float((orient_score + pos_score).sum()),
        "epnp_orient_sum": float(orient_score.sum()),
        "epnp_pos_sum": float(pos_score.sum()),
        "epnp_rot_sum": float(rot_err_deg.sum()),
        "epnp_terr_sum": float(trans_err.sum()),
        "epnp_n_ok": float(pnp_success[has_np].sum()),
        "epnp_n_valid": float(n_valid),
        "epnp_n_total": float(n_has_pose),
    }


def compute_pixel_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visibility: torch.Tensor,
    crop_box: torch.Tensor,
) -> torch.Tensor:
    """Compute mean Euclidean error in original image pixel coordinates.

    Args:
        pred: (B, K, 2) predicted keypoints in crop-relative [0,1]
        gt: (B, K, 2) ground truth keypoints in crop-relative [0,1]
        visibility: (B, K) visibility flags
        crop_box: (B, 4) crop boxes (x1, y1, x2, y2) in pixels

    Returns:
        Scalar mean pixel error over visible keypoints
    """
    x1 = crop_box[:, 0:1].unsqueeze(-1)
    y1 = crop_box[:, 1:2].unsqueeze(-1)
    x2 = crop_box[:, 2:3].unsqueeze(-1)
    y2 = crop_box[:, 3:4].unsqueeze(-1)
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale = torch.cat([crop_w, crop_h], dim=-1)
    offset = torch.cat([x1, y1], dim=-1)

    pred_px = pred * scale + offset
    gt_px = gt * scale + offset

    dist = torch.norm(pred_px - gt_px, dim=-1)

    mask = visibility > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    return dist[mask].mean()


def compute_pixel_rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visibility: torch.Tensor,
    crop_box: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE of Euclidean distances in original image pixel coordinates.

    Unlike compute_pixel_error (which returns mean of distances), this returns
    sqrt(mean(distance²)), penalizing outlier keypoints more heavily.

    Args:
        pred: (B, K, 2) predicted keypoints in crop-relative [0,1]
        gt: (B, K, 2) ground truth keypoints in crop-relative [0,1]
        visibility: (B, K) visibility flags
        crop_box: (B, 4) crop boxes (x1, y1, x2, y2) in pixels

    Returns:
        Scalar RMSE pixel error over visible keypoints
    """
    x1 = crop_box[:, 0:1].unsqueeze(-1)
    y1 = crop_box[:, 1:2].unsqueeze(-1)
    x2 = crop_box[:, 2:3].unsqueeze(-1)
    y2 = crop_box[:, 3:4].unsqueeze(-1)
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale = torch.cat([crop_w, crop_h], dim=-1)
    offset = torch.cat([x1, y1], dim=-1)

    pred_px = pred * scale + offset
    gt_px = gt * scale + offset

    dist_sq = torch.sum((pred_px - gt_px) ** 2, dim=-1)

    mask = visibility > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.sqrt(dist_sq[mask].mean())


def compute_pck(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visibility: torch.Tensor,
    crop_box: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Compute PCK (Percentage of Correct Keypoints).

    A keypoint is correct if its pixel error is within `threshold` * bbox_diagonal.

    Args:
        pred: (B, K, 2) predicted keypoints in crop-relative [0,1]
        gt: (B, K, 2) ground truth keypoints in crop-relative [0,1]
        visibility: (B, K) visibility flags
        crop_box: (B, 4) crop boxes (x1, y1, x2, y2) in pixels
        threshold: fraction of bbox diagonal

    Returns:
        Scalar PCK value in [0, 1]
    """
    x1 = crop_box[:, 0:1].unsqueeze(-1)
    y1 = crop_box[:, 1:2].unsqueeze(-1)
    x2 = crop_box[:, 2:3].unsqueeze(-1)
    y2 = crop_box[:, 3:4].unsqueeze(-1)
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale = torch.cat([crop_w, crop_h], dim=-1)
    offset = torch.cat([x1, y1], dim=-1)

    pred_px = pred * scale + offset
    gt_px = gt * scale + offset

    dist = torch.norm(pred_px - gt_px, dim=-1)

    diag = torch.sqrt(crop_w.squeeze(-1).squeeze(-1) ** 2 + crop_h.squeeze(-1).squeeze(-1) ** 2)
    thresh_px = threshold * diag

    mask = visibility > 0
    correct = (dist < thresh_px.unsqueeze(1)) & mask

    if mask.sum() == 0:
        return torch.tensor(1.0, device=pred.device)

    return correct.sum().float() / mask.sum().float()


def compute_rotation_error(
    pred_R: torch.Tensor,
    gt_q: torch.Tensor,
    has_pose: torch.Tensor,
) -> torch.Tensor:
    """Compute mean rotation error in degrees.

    Args:
        pred_R: (B, 3, 3) predicted rotation matrix
        gt_q: (B, 4) ground truth quaternion (w, x, y, z) SPEED+ convention
        has_pose: (B,) bool mask

    Returns:
        Scalar mean rotation error in degrees
    """
    if has_pose.sum() == 0:
        return torch.tensor(0.0, device=pred_R.device)

    pred_R = pred_R[has_pose]
    gt_q = gt_q[has_pose]

    gt_R = quaternion_to_matrix(gt_q)
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * (180.0 / torch.pi)

    return angle_deg.mean()


def compute_translation_error(
    pred_t: torch.Tensor,
    gt_t: torch.Tensor,
    has_pose: torch.Tensor,
) -> torch.Tensor:
    """Compute mean translation error (Euclidean distance).

    Args:
        pred_t: (B, 3) predicted translation
        gt_t: (B, 3) ground truth translation
        has_pose: (B,) bool mask

    Returns:
        Scalar mean translation error
    """
    if has_pose.sum() == 0:
        return torch.tensor(0.0, device=pred_t.device)

    pred_t = pred_t[has_pose]
    gt_t = gt_t[has_pose]

    return torch.norm(pred_t - gt_t, dim=-1).mean()


def compute_slab_score(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_q: torch.Tensor,
    gt_t: torch.Tensor,
    has_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the SLAB/SPEC2021 pose score used in the SPEED+ competition.

    Per-image score = orientation_score + position_score, averaged over N images.

    Orientation error: e_q = 2 * arccos(|<q_pred, q_gt>|)  (radians)
    Position error:    e_t = ||t_pred - t_gt|| / ||t_gt||   (relative)

    Errors below machine-precision thresholds are zeroed:
        orientation < 0.169 deg (0.00295 rad) -> 0
        position    < 0.002173                -> 0

    Args:
        pred_R: (B, 3, 3) predicted rotation matrix
        pred_t: (B, 3) predicted translation
        gt_q: (B, 4) ground truth quaternion (w, x, y, z) SPEED+ convention
        gt_t: (B, 3) ground truth translation
        has_pose: (B,) bool mask

    Returns:
        dict with slab_score, orientation_score, position_score (all scalar)
    """
    zero = torch.tensor(0.0, device=pred_R.device)
    if has_pose.sum() == 0:
        return {"slab_score": zero, "orientation_score": zero, "position_score": zero}

    pred_R = pred_R[has_pose]
    pred_t = pred_t[has_pose]
    gt_q = gt_q[has_pose]
    gt_t = gt_t[has_pose]

    # pred_R is body-to-camera (Rodrigues convention). The SPEED+ quat2dcm
    # formula builds the DCM transposed relative to the standard convention,
    # so matrix_to_quaternion(R_body2cam) directly matches gt_q — no transpose.
    pred_q = matrix_to_quaternion(pred_R)

    # Orientation error: 2 * arccos(|<q_pred, q_gt>|)
    dot = torch.abs((pred_q * gt_q).sum(dim=-1)).clamp(0.0, 1.0)
    err_orientation = 2.0 * torch.acos(dot.clamp(max=1.0 - 1e-7))  # radians

    # Threshold: 0.169 deg = 0.002949 rad
    orientation_score = torch.where(
        err_orientation < 0.002949,
        torch.zeros_like(err_orientation),
        err_orientation,
    )

    # Position error: ||t_pred - t_gt|| / ||t_gt||
    err_position = torch.norm(pred_t - gt_t, dim=-1) / torch.norm(gt_t, dim=-1).clamp(min=1e-8)

    # Threshold: 0.002173
    position_score = torch.where(
        err_position < 0.002173,
        torch.zeros_like(err_position),
        err_position,
    )

    # Per-image pose score = orientation + position, averaged
    pose_scores = orientation_score + position_score

    return {
        "slab_score": pose_scores.mean(),
        "orientation_score": orientation_score.mean(),
        "position_score": position_score.mean(),
    }


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion in SPEED+ convention (w, x, y, z).

    Args:
        R: (B, 3, 3) rotation matrices

    Returns:
        (B, 4) quaternions in (w, x, y, z) scalar-first format
    """
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2  # s = 4*w
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10)) * 2
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]

    # Normalize
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return q


def visualize_heatmaps(
    image: Image.Image,
    heatmaps: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay predicted heatmaps on an image using JET colormap.

    Args:
        image: PIL Image (the crop or full image)
        heatmaps: (K, H, W) predicted heatmaps (raw logits)
        alpha: blending factor for the heatmap overlay

    Returns:
        PIL Image with heatmap overlay
    """
    w, h = image.size

    # Max across keypoints to get a single activation map
    combined = heatmaps.max(axis=0)  # (H, W)

    # Normalize to [0, 1]
    vmin, vmax = combined.min(), combined.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)
    else:
        combined = np.zeros_like(combined)

    # Convert to uint8 and apply JET colormap
    hm_uint8 = (combined * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)  # (H, W, 3) BGR
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Resize to match image
    hm_pil = Image.fromarray(hm_color).resize((w, h), Image.BILINEAR)

    # Alpha blend
    img_rgb = image.convert("RGB")
    blended = Image.blend(img_rgb, hm_pil, alpha)

    return blended


def visualize_keypoints(
    image: Image.Image,
    pred_kp: np.ndarray,
    gt_kp: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    radius: int = 4,
) -> Image.Image:
    """Draw predicted (and optionally GT) keypoints on an image.

    Args:
        image: PIL Image (the crop)
        pred_kp: (K, 2) in [0,1] crop-relative coords
        gt_kp: (K, 2) in [0,1] crop-relative coords, optional
        visibility: (K,) visibility flags, optional
        radius: circle radius in pixels

    Returns:
        PIL Image with keypoints drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i in range(len(pred_kp)):
        if visibility is not None and visibility[i] == 0:
            continue

        # Predicted: red
        px, py = pred_kp[i, 0] * w, pred_kp[i, 1] * h
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill="red",
            outline="red",
        )

        # Ground truth: green
        if gt_kp is not None:
            gx, gy = gt_kp[i, 0] * w, gt_kp[i, 1] * h
            draw.ellipse(
                [gx - radius, gy - radius, gx + radius, gy + radius],
                fill="green",
                outline="green",
            )
            draw.line([(px, py), (gx, gy)], fill="yellow", width=1)

    return img
