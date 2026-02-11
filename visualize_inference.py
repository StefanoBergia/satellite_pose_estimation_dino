"""Visualize inference results across val, lightbox, and sunlamp splits.

Draws predicted and ground truth keypoints on the FULL original image
(not just the crop), with the crop bounding box shown. Also draws pose
axes when available. Generates individual images and a summary grid per split.

Usage:
    python visualize_inference.py --checkpoint outputs/best_model.pth
    python visualize_inference.py --checkpoint outputs/best_model.pth --num_samples 10 --out_dir outputs/viz_inference
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel, quaternion_to_matrix
from src.utils import (
    compute_pixel_error,
    compute_rotation_error,
    compute_translation_error,
    load_pnp_data,
    solve_pnp_batch,
    visualize_heatmaps,
)


def crop_to_full_keypoints(kp_crop, crop_box):
    """Convert crop-relative [0,1] keypoints to full-image pixel coords.

    Args:
        kp_crop: (K, 2) in [0,1] crop-relative
        crop_box: (4,) [x1, y1, x2, y2] in pixels

    Returns:
        (K, 2) in full-image pixel coordinates
    """
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1
    kp_px = np.zeros_like(kp_crop)
    kp_px[:, 0] = kp_crop[:, 0] * crop_w + x1
    kp_px[:, 1] = kp_crop[:, 1] * crop_h + y1
    return kp_px


def draw_axes_fullimg(
    draw: ImageDraw.ImageDraw,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    scale: float,
    axis_length: float = 0.3,
    line_width: int = 3,
    label: str | None = None,
    colors: tuple[str, str, str] = ("red", "green", "blue"),
):
    """Draw projected 3D coordinate axes directly in full-image pixel coords."""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    origin = translation
    axes_3d = [
        translation + rotation[:, 0] * axis_length,
        translation + rotation[:, 1] * axis_length,
        translation + rotation[:, 2] * axis_length,
    ]

    def project(pt3d):
        if pt3d[2] <= 0:
            return None
        u = (fx * pt3d[0] / pt3d[2] + cx) * scale
        v = (fy * pt3d[1] / pt3d[2] + cy) * scale
        return (u, v)

    origin_2d = project(origin)
    if origin_2d is None:
        return

    for ax, color in zip(axes_3d, colors):
        end_2d = project(ax)
        if end_2d is not None:
            draw.line([origin_2d, end_2d], fill=color, width=line_width)

    if label is not None:
        draw.text((origin_2d[0] + 5, origin_2d[1] - 12), label,
                  fill="white")


# Tango spacecraft wireframe edges (indices into the 11 keypoints)
# 0-3: top face, 4-7: bottom face, 8: left antenna, 9: right antenna, 10: bottom arm
WIREFRAME_EDGES = [
    # Top face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Vertical struts
    (0, 4), (1, 5), (2, 6), (3, 7),
    # Appendages
    (1, 8), (5, 8),   # left antenna to nearest top/bottom corners
    (2, 9), (6, 9),   # right antenna
    (3, 10), (7, 10),  # bottom arm
]


def project_points(points_3d, rotation, translation, camera_matrix, scale=1.0):
    """Project 3D points to 2D pixel coordinates using pose and camera intrinsics.

    Args:
        points_3d: (K, 3) 3D points in object frame
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
        camera_matrix: (3, 3) camera intrinsic matrix
        scale: display scale factor

    Returns:
        pts_2d: (K, 2) pixel coordinates (u, v), NaN if behind camera
    """
    # Transform to camera frame: P_cam = R @ P_obj + t
    pts_cam = (rotation @ points_3d.T).T + translation  # (K, 3)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    pts_2d = np.full((len(points_3d), 2), np.nan)
    valid = pts_cam[:, 2] > 0
    pts_2d[valid, 0] = (fx * pts_cam[valid, 0] / pts_cam[valid, 2] + cx) * scale
    pts_2d[valid, 1] = (fy * pts_cam[valid, 1] / pts_cam[valid, 2] + cy) * scale
    return pts_2d


def draw_wireframe(
    draw: ImageDraw.ImageDraw,
    points_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    scale: float = 1.0,
    color: str = "lime",
    line_width: int = 2,
    label: str | None = None,
    dot_radius: int = 3,
):
    """Project 3D wireframe model onto the image and draw edges + vertices."""
    pts_2d = project_points(points_3d, rotation, translation, camera_matrix, scale)

    # Draw edges
    for i, j in WIREFRAME_EDGES:
        if np.isnan(pts_2d[i]).any() or np.isnan(pts_2d[j]).any():
            continue
        p1 = (pts_2d[i, 0], pts_2d[i, 1])
        p2 = (pts_2d[j, 0], pts_2d[j, 1])
        draw.line([p1, p2], fill=color, width=line_width)

    # Draw vertices
    for k in range(len(pts_2d)):
        if np.isnan(pts_2d[k]).any():
            continue
        x, y = pts_2d[k]
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
            fill=color, outline="white",
        )

    # Label at centroid of visible projected points
    if label is not None:
        valid = ~np.isnan(pts_2d[:, 0])
        if valid.any():
            cx = pts_2d[valid, 0].mean()
            cy = pts_2d[valid, 1].min() - 14
            draw.text((cx, cy), label, fill=color)


def draw_wireframe_comparison(
    full_image: Image.Image,
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    scale: float,
    pred_R=None, pred_t=None,
    gt_R=None, gt_t=None,
) -> Image.Image:
    """Create a wireframe overlay image comparing GT and predicted poses."""
    img = full_image.copy()
    draw = ImageDraw.Draw(img)

    if gt_R is not None:
        draw_wireframe(draw, points_3d, gt_R, gt_t, camera_matrix, scale,
                       color="#00FF00", line_width=2, label="GT", dot_radius=4)
    if pred_R is not None:
        draw_wireframe(draw, points_3d, pred_R, pred_t, camera_matrix, scale,
                       color="#FF4444", line_width=2, label="Pred", dot_radius=3)
    return img


def draw_on_full_image(
    full_image, pred_kp_crop, gt_kp_crop, visibility, crop_box,
    pred_R=None, pred_t=None, gt_R=None, gt_t=None,
    camera_matrix=None, scale=1.0, radius=5,
):
    """Draw keypoints, bbox, and pose axes on the full image.

    Args:
        full_image: PIL Image (full frame)
        pred_kp_crop: (K, 2) predicted keypoints in crop-relative [0,1]
        gt_kp_crop: (K, 2) GT keypoints in crop-relative [0,1]
        visibility: (K,) visibility flags
        crop_box: (4,) [x1, y1, x2, y2] in original pixel coords
        scale: scaling factor applied to the display image
        radius: keypoint circle radius in display pixels
    """
    img = full_image.copy()
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = crop_box * scale

    # Draw crop bounding box
    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)

    # Convert keypoints to full-image pixel coords (scaled)
    pred_px = crop_to_full_keypoints(pred_kp_crop, crop_box) * scale
    gt_px = crop_to_full_keypoints(gt_kp_crop, crop_box) * scale

    for i in range(len(pred_kp_crop)):
        if visibility[i] == 0:
            continue

        px, py = pred_px[i, 0], pred_px[i, 1]
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     fill="red", outline="white")

        gx, gy = gt_px[i, 0], gt_px[i, 1]
        draw.ellipse([gx - radius, gy - radius, gx + radius, gy + radius],
                     fill="lime", outline="white")
        draw.line([(px, py), (gx, gy)], fill="yellow", width=1)

    # Draw pose axes in full-image coordinates
    # GT axes: thin, pastel colors, labeled "GT"
    if camera_matrix is not None and gt_R is not None:
        draw_axes_fullimg(draw, gt_R, gt_t, camera_matrix, scale,
                          axis_length=0.25, line_width=2, label="GT",
                          colors=("#FF9999", "#99FF99", "#9999FF"))
    # Pred axes: thick, saturated colors, labeled "Pred"/"EPnP"
    if camera_matrix is not None and pred_R is not None:
        draw_axes_fullimg(draw, pred_R, pred_t, camera_matrix, scale,
                          axis_length=0.3, line_width=3, label="Pred")

    return img


def compute_sample_metrics(model_out, sample, points_3d=None, camera_matrix=None, dist_coeffs=None):
    """Compute per-sample metrics. Returns a dict of metric strings."""
    metrics = {}

    pred_kp = model_out["keypoints"].cpu()
    gt_kp = sample["keypoints"].unsqueeze(0)
    vis = sample["visibility"].unsqueeze(0)
    crop_box = sample["crop_box"].unsqueeze(0)

    px_err = compute_pixel_error(pred_kp, gt_kp, vis, crop_box)
    metrics["px_err"] = f"{px_err.item():.1f}px"

    if "direct_rotation" in model_out and sample["has_pose"]:
        pred_R = model_out["direct_rotation"].cpu()
        pred_t = model_out["direct_translation"].cpu()
        gt_q = sample["quaternion"].unsqueeze(0)
        gt_t = sample["translation"].unsqueeze(0)
        has_pose = torch.tensor([True])

        rot_err = compute_rotation_error(pred_R, gt_q, has_pose)
        trans_err = compute_translation_error(pred_t, gt_t, has_pose)
        metrics["rot"] = f"{rot_err.item():.1f}\u00b0"
        metrics["t_err"] = f"{trans_err.item():.3f}m"

    elif points_3d is not None and sample["has_pose"]:
        # Extract per-keypoint confidence from heatmap peaks
        confidence = None
        if "heatmaps" in model_out:
            hm = model_out["heatmaps"].cpu().squeeze(0).numpy()  # (K, H, W)
            confidence = hm.max(axis=(1, 2)).reshape(1, -1)  # (1, K)

        # EPnP-based pose from predicted keypoints
        rotations, translations, success, _ = solve_pnp_batch(
            pred_kp, crop_box, vis, points_3d, camera_matrix, dist_coeffs,
            confidence=confidence,
        )
        if success[0]:
            pred_R = torch.from_numpy(rotations[:1]).float()
            pred_t_vec = torch.from_numpy(translations[:1]).float()
            gt_q = sample["quaternion"].unsqueeze(0)
            gt_t = sample["translation"].unsqueeze(0)
            has_pose = torch.tensor([True])

            rot_err = compute_rotation_error(pred_R, gt_q, has_pose)
            trans_err = compute_translation_error(pred_t_vec, gt_t, has_pose)
            metrics["rot(epnp)"] = f"{rot_err.item():.1f}\u00b0"
            metrics["t(epnp)"] = f"{trans_err.item():.3f}m"

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize inference across splits")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per split")
    parser.add_argument("--out_dir", type=str, default="outputs/viz_inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display_width", type=int, default=960,
                        help="Width of the output images (full frame is scaled to this)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]
    root = Path(config["data"]["root"])

    # Load camera matrix, 3D points, and distortion coefficients for EPnP
    camera_matrix = None
    points_3d = None
    dist_coeffs = None
    geo_cfg = config.get("geometry", {})
    if geo_cfg.get("camera") and geo_cfg.get("points_3d"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
        camera_matrix = pnp_data["camera_matrix"].astype(np.float32)
        points_3d = pnp_data["points_3d"]
        dist_coeffs = pnp_data["dist_coeffs"]

    # Load model
    print(f"Loading model from {args.checkpoint}...")
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint epoch: {epoch}, mode: {mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = KeypointTransform(image_size=config["data"]["image_size"], is_train=False)

    for split in args.splits:
        if split not in config["data"]["splits"]:
            print(f"  Skipping {split} (not in config)")
            continue

        split_cfg = config["data"]["splits"][split]

        pose_json = None
        pose_labels = config["data"].get("pose_labels", {})
        if split in pose_labels:
            pose_json = pose_labels[split]

        # Dataset with transforms (for model input)
        dataset = SpeedPlusKeypointDataset(
            image_dir=str(root / split_cfg["images"]),
            label_dir=str(root / split_cfg["labels"]),
            num_keypoints=config["data"]["num_keypoints"],
            bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
            transform=transform,
            pose_json=pose_json,
        )

        random.seed(args.seed)
        n_samples = min(args.num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), n_samples)

        print(f"\n[{split}] Visualizing {n_samples} samples...")

        tiles = []
        for idx in indices:
            sample = dataset[idx]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            gt_kp = sample["keypoints"].numpy()
            vis = sample["visibility"].numpy()
            crop_box = sample["crop_box"].numpy()
            img_size = sample["img_size"].numpy()  # [W, H]

            fwd_kwargs = {"pixel_values": image_tensor}
            if mode == "keypoint_pose_pnp":
                fwd_kwargs["crop_box"] = sample["crop_box"].unsqueeze(0).to(device)
                fwd_kwargs["img_size"] = sample["img_size"].unsqueeze(0).to(device)
                fwd_kwargs["visibility"] = sample["visibility"].unsqueeze(0).to(device)

            with torch.no_grad():
                model_out = model(**fwd_kwargs)

            pred_kp = model_out["keypoints"].cpu().squeeze(0).numpy()

            # Per-sample metrics
            metrics = compute_sample_metrics(
                model_out, sample, points_3d, camera_matrix, dist_coeffs,
            )

            # Load original full image
            img_path = dataset.samples[idx][0]
            full_image = Image.open(img_path)
            if full_image.mode == "L":
                full_image = full_image.convert("RGB")

            # Scale full image for display
            orig_w, orig_h = full_image.size
            scale = args.display_width / orig_w
            display_h = int(orig_h * scale)
            full_display = full_image.resize((args.display_width, display_h), Image.BILINEAR)

            # Get pose data
            pred_R = pred_t = gt_R = gt_t = None
            if "direct_rotation" in model_out:
                pred_R = model_out["direct_rotation"].cpu().squeeze(0).numpy()
                pred_t = model_out["direct_translation"].cpu().squeeze(0).numpy()
            elif points_3d is not None:
                # Solve pose via EPnP from predicted keypoints
                kp_tensor = model_out["keypoints"].cpu()
                cb_tensor = sample["crop_box"].unsqueeze(0)
                vis_tensor = sample["visibility"].unsqueeze(0)
                confidence = None
                if "heatmaps" in model_out:
                    hm = model_out["heatmaps"].cpu().squeeze(0).numpy()  # (K, H, W)
                    confidence = hm.max(axis=(1, 2)).reshape(1, -1)  # (1, K)
                rotations, translations, success, _ = solve_pnp_batch(
                    kp_tensor, cb_tensor, vis_tensor,
                    points_3d, camera_matrix, dist_coeffs,
                    confidence=confidence,
                )
                if success[0]:
                    pred_R = rotations[0]
                    pred_t = translations[0]
            if sample["has_pose"]:
                gt_q = sample["quaternion"].numpy()
                gt_R = quaternion_to_matrix(
                    torch.tensor(gt_q).unsqueeze(0)
                ).squeeze(0).numpy()
                gt_t = sample["translation"].numpy()

            # Draw on full image
            viz = draw_on_full_image(
                full_display, pred_kp, gt_kp, vis, crop_box,
                pred_R=pred_R, pred_t=pred_t,
                gt_R=gt_R, gt_t=gt_t,
                camera_matrix=camera_matrix, scale=scale, radius=5,
            )

            # Add metric text overlay
            draw = ImageDraw.Draw(viz)
            metric_text = "  ".join(f"{k}={v}" for k, v in metrics.items())
            draw.rectangle([0, display_h - 20, args.display_width, display_h],
                           fill=(0, 0, 0))
            draw.text((4, display_h - 18), f"[{split}] {metric_text}", fill="white")

            # Save individual image
            viz.save(out_dir / f"{split}_{idx:05d}.png")
            tiles.append(viz)

            # Save heatmap overlay on the crop region
            if "heatmaps" in model_out:
                # Extract and resize the crop region from the full display image
                sx1, sy1, sx2, sy2 = (crop_box * scale).astype(int)
                sx1, sy1 = max(0, sx1), max(0, sy1)
                sx2 = min(args.display_width, sx2)
                sy2 = min(display_h, sy2)
                crop_region = full_display.crop((sx1, sy1, sx2, sy2))
                hm_viz = visualize_heatmaps(
                    crop_region,
                    model_out["heatmaps"].cpu().squeeze(0).numpy(),
                )
                hm_viz.save(out_dir / f"{split}_{idx:05d}_heatmap.png")

            # Save wireframe comparison (GT green vs Pred red)
            if points_3d is not None and (pred_R is not None or gt_R is not None):
                wf_viz = draw_wireframe_comparison(
                    full_display, points_3d, camera_matrix, scale,
                    pred_R=pred_R, pred_t=pred_t,
                    gt_R=gt_R, gt_t=gt_t,
                )
                # Add metric text overlay
                wf_draw = ImageDraw.Draw(wf_viz)
                wf_draw.rectangle([0, display_h - 20, args.display_width, display_h],
                                  fill=(0, 0, 0))
                wf_draw.text((4, display_h - 18),
                             f"[{split}] wireframe: green=GT  red=Pred  {metric_text}",
                             fill="white")
                wf_viz.save(out_dir / f"{split}_{idx:05d}_wireframe.png")

        # Build summary grid (2 columns for full-frame images)
        n_cols = min(2, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        tile_w = args.display_width
        tile_h = display_h
        header_h = 28
        grid_w = n_cols * tile_w
        grid_h = n_rows * tile_h + header_h

        grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid)
        draw.text((8, 6), f"{split.upper()} -- epoch {epoch} -- {mode}", fill="white")

        for i, tile in enumerate(tiles):
            row, col = divmod(i, n_cols)
            grid.paste(tile, (col * tile_w, row * tile_h + header_h))

        grid_path = out_dir / f"grid_{split}.png"
        grid.save(grid_path)
        print(f"  Saved grid to {grid_path}")

    print(f"\nDone. All outputs in {out_dir}/")
    print("Legend: red=predicted, green=GT keypoints, yellow=error line, cyan=crop bbox")
    if camera_matrix is not None:
        print("        'Pred' thick axes=predicted pose, 'GT' thin pastel axes=ground truth pose")


if __name__ == "__main__":
    main()
