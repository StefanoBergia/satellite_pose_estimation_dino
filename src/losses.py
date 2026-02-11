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
