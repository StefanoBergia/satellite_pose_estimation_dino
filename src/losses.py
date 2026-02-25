import math

import torch
import torch.nn.functional as F

from .model import quaternion_to_matrix


def visibility_weighted_mse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visibility: torch.Tensor,
    occluded_weight: float = 0.5,
) -> torch.Tensor:
    """MSE loss weighted by keypoint visibility.

    Args:
        pred: (B, K, 2)
        gt: (B, K, 2)
        visibility: (B, K) — 0=ignore, 1=occluded, 2=visible
        occluded_weight: weight for occluded (vis=1) keypoints

    Returns:
        Scalar loss
    """
    mse = ((pred - gt) ** 2).sum(dim=-1)  # (B, K)

    weight = torch.zeros_like(mse)
    weight[visibility == 2] = 1.0
    weight[visibility == 1] = occluded_weight

    if weight.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    loss = (mse * weight).sum() / weight.sum()
    return loss


def generate_heatmaps(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    """Generate normalized ground truth Gaussian heatmaps from keypoint coordinates.

    Each heatmap is a 2D Gaussian normalized to sum to 1 (probability distribution),
    matching the spatial-softmax output of the heatmap head.

    Args:
        keypoints: (B, K, 2) in [0, 1] crop-relative coordinates
        visibility: (B, K) — 0=ignore, 1=occluded, 2=visible
        heatmap_size: spatial size H=W of the heatmap grid
        sigma: Gaussian standard deviation in heatmap pixel space

    Returns:
        (B, K, H, W) ground truth heatmaps, each summing to 1 for visible keypoints
    """
    B, K, _ = keypoints.shape

    # Scale keypoints to heatmap pixel space
    kp_hm = keypoints * (heatmap_size - 1)  # (B, K, 2) in [0, heatmap_size-1]
    mu_x = kp_hm[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    mu_y = kp_hm[:, :, 1].unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)

    # Coordinate grids
    coords = torch.arange(heatmap_size, device=keypoints.device, dtype=keypoints.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (H, W) each
    xx = xx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    yy = yy.unsqueeze(0).unsqueeze(0)

    # 2D Gaussian
    gauss = torch.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * sigma ** 2))

    # Normalize each heatmap to sum to 1 (proper probability distribution)
    gauss = gauss / (gauss.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    # Zero out invisible keypoints
    mask = (visibility > 0).float().unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    return gauss * mask


def heatmap_mse_loss(
    pred_heatmaps: torch.Tensor,
    gt_keypoints: torch.Tensor,
    visibility: torch.Tensor,
    heatmap_size: int,
    sigma: float,
    occluded_weight: float = 0.5,
) -> torch.Tensor:
    """MSE loss between predicted and GT heatmaps, weighted by visibility.

    Both predicted and GT heatmaps are normalized probability distributions
    (sum to 1 per keypoint channel).

    Args:
        pred_heatmaps: (B, K, H, W) predicted heatmaps (spatial-softmax normalized)
        gt_keypoints: (B, K, 2) in [0, 1]
        visibility: (B, K) — 0=ignore, 1=occluded, 2=visible
        heatmap_size: spatial size of heatmaps
        sigma: Gaussian sigma for GT generation
        occluded_weight: weight for occluded (vis=1) keypoints

    Returns:
        Scalar loss
    """
    gt_heatmaps = generate_heatmaps(gt_keypoints, visibility, heatmap_size, sigma)

    # Per-keypoint MSE averaged over spatial dims
    mse_per_kp = ((pred_heatmaps - gt_heatmaps) ** 2).mean(dim=(-2, -1))  # (B, K)

    # Visibility weighting
    weight = torch.zeros_like(mse_per_kp)
    weight[visibility == 2] = 1.0
    weight[visibility == 1] = occluded_weight

    if weight.sum() == 0:
        return torch.tensor(0.0, device=pred_heatmaps.device, requires_grad=True)

    return (mse_per_kp * weight).sum() / weight.sum()


def _ssim_single_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    win_size: int,
    data_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SSIM components for a single scale using Gaussian-windowed statistics.

    Args:
        x, y: (N, 1, H, W) tensors
        win_size: Gaussian window size (must be odd)
        data_range: dynamic range of the input (max pixel value)

    Returns:
        ssim_map: (N, 1, H', W') full SSIM (luminance * contrast * structure)
        cs_map: (N, 1, H', W') contrast-structure component only
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 1D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(win_size, dtype=x.dtype, device=x.device) - win_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()

    # 2D separable Gaussian window: (1, 1, win_size, win_size)
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)

    pad = win_size // 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    # Clamp variances to zero (numerical artifacts can produce tiny negatives)
    sigma_x_sq = sigma_x_sq.clamp(min=0)
    sigma_y_sq = sigma_y_sq.clamp(min=0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)

    return ssim_map, cs_map


def heatmap_msssim_loss(
    pred_heatmaps: torch.Tensor,
    gt_keypoints: torch.Tensor,
    visibility: torch.Tensor,
    heatmap_size: int,
    sigma: float,
    win_size: int = 7,
    num_scales: int | None = None,
    occluded_weight: float = 0.5,
) -> torch.Tensor:
    """Multi-Scale SSIM loss between predicted and GT heatmaps, weighted by visibility.

    Args:
        pred_heatmaps: (B, K, H, W) predicted heatmaps (spatial-softmax normalized)
        gt_keypoints: (B, K, 2) in [0, 1]
        visibility: (B, K) — 0=ignore, 1=occluded, 2=visible
        heatmap_size: spatial size of heatmaps
        sigma: Gaussian sigma for GT heatmap generation
        win_size: SSIM Gaussian window size
        num_scales: number of MS-SSIM scales (auto-detected if None)
        occluded_weight: weight for occluded (vis=1) keypoints

    Returns:
        Scalar loss = 1 - weighted_mean_ms_ssim
    """
    gt_heatmaps = generate_heatmaps(gt_keypoints, visibility, heatmap_size, sigma)

    B, K, H, W = pred_heatmaps.shape

    # Auto-detect number of scales: floor(log2(H / win_size)) + 1, clamped to [2, 5]
    if num_scales is None:
        num_scales = int(math.floor(math.log2(H / win_size))) + 1
    num_scales = max(2, min(5, num_scales))

    # MS-SSIM paper weights, truncated to num_scales
    all_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(all_weights[:num_scales], dtype=pred_heatmaps.dtype, device=pred_heatmaps.device)
    weights = weights / weights.sum()  # re-normalize after truncation

    # Reshape to (B*K, 1, H, W) — each keypoint independently
    pred = pred_heatmaps.reshape(B * K, 1, H, W)
    gt = gt_heatmaps.reshape(B * K, 1, H, W)

    # Dynamic data range
    data_range = max(pred.max().item(), gt.max().item())
    if data_range < 1e-8:
        return torch.tensor(0.0, device=pred_heatmaps.device, requires_grad=True)

    # Multi-scale loop: collect CS at each scale, full SSIM at final scale
    mcs_list = []
    for i in range(num_scales):
        ssim_map, cs_map = _ssim_single_scale(pred, gt, win_size, data_range)

        if i < num_scales - 1:
            # Intermediate scales: store mean CS
            mcs_list.append(cs_map.mean(dim=(-2, -1)).clamp(min=1e-8))
            # Downsample for next scale
            pred = F.avg_pool2d(pred, kernel_size=2)
            gt = F.avg_pool2d(gt, kernel_size=2)
        else:
            # Final scale: store mean full SSIM
            mcs_list.append(ssim_map.mean(dim=(-2, -1)).clamp(min=1e-8))

    # Stack: (num_scales, B*K, 1)
    mcs_stack = torch.stack(mcs_list, dim=0)

    # Weighted product: prod(mcs_i ^ weight_i) across scales
    # weights: (num_scales,) -> (num_scales, 1, 1)
    w = weights.reshape(-1, 1, 1)
    ms_ssim = torch.prod(mcs_stack ** w, dim=0).squeeze(-1)  # (B*K,)

    # Reshape back to (B, K)
    ms_ssim = ms_ssim.reshape(B, K)

    # Visibility weighting (same pattern as heatmap_mse_loss)
    weight = torch.zeros_like(ms_ssim)
    weight[visibility == 2] = 1.0
    weight[visibility == 1] = occluded_weight

    if weight.sum() == 0:
        return torch.tensor(0.0, device=pred_heatmaps.device, requires_grad=True)

    # Loss = 1 - weighted mean MS-SSIM
    return 1.0 - (ms_ssim * weight).sum() / weight.sum()


def geodesic_rotation_loss(
    pred_R: torch.Tensor,
    gt_q: torch.Tensor,
    has_pose: torch.Tensor,
) -> torch.Tensor:
    """Geodesic distance between predicted rotation matrix and GT quaternion.

    L = arccos((trace(R_pred^T @ R_gt) - 1) / 2)

    Args:
        pred_R: (B, 3, 3) predicted rotation matrix
        gt_q: (B, 4) ground truth quaternion (w, x, y, z) SPEED+ convention
        has_pose: (B,) bool mask for samples with pose labels

    Returns:
        Scalar mean geodesic loss (in radians)
    """
    if has_pose.sum() == 0:
        return torch.tensor(0.0, device=pred_R.device, requires_grad=True)

    pred_R = pred_R[has_pose]
    gt_q = gt_q[has_pose]

    gt_R = quaternion_to_matrix(gt_q)

    # R_diff = R_pred^T @ R_gt
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)

    # trace
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

    # Clamp for numerical stability
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    return angle.mean()


def translation_loss(
    pred_t: torch.Tensor,
    gt_t: torch.Tensor,
    has_pose: torch.Tensor,
) -> torch.Tensor:
    """L1 loss on translation vector.

    Args:
        pred_t: (B, 3) predicted translation
        gt_t: (B, 3) ground truth translation
        has_pose: (B,) bool mask

    Returns:
        Scalar mean L1 loss
    """
    if has_pose.sum() == 0:
        return torch.tensor(0.0, device=pred_t.device, requires_grad=True)

    pred_t = pred_t[has_pose]
    gt_t = gt_t[has_pose]

    return F.l1_loss(pred_t, gt_t)


def combined_pose_loss(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_q: torch.Tensor,
    gt_t: torch.Tensor,
    has_pose: torch.Tensor,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined rotation + translation loss.

    Returns individual losses and weighted total for logging.
    """
    rot_loss = geodesic_rotation_loss(pred_R, gt_q, has_pose)
    trans_loss = translation_loss(pred_t, gt_t, has_pose)
    total = rotation_weight * rot_loss + translation_weight * trans_loss

    return {
        "rotation_loss": rot_loss,
        "translation_loss": trans_loss,
        "pose_loss": total,
    }
