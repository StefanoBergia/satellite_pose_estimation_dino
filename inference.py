"""Run inference on images and visualize predicted keypoints and pose.

Usage:
    python inference.py --config config.yaml --checkpoint outputs/best_model.pth --split val --num_samples 20 --out_dir outputs/viz
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import ImageDraw

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel, quaternion_to_matrix
from src.utils import visualize_keypoints, visualize_heatmaps


def draw_axes(
    draw: ImageDraw.ImageDraw,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    crop_box: np.ndarray,
    viz_w: int,
    viz_h: int,
    axis_length: float = 0.3,
    line_width: int = 3,
):
    """Draw projected 3D coordinate axes on the visualization image.

    Colors: X=red, Y=green, Z=blue
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    origin = translation
    axes_3d = [
        translation + rotation[:, 0] * axis_length,  # X
        translation + rotation[:, 1] * axis_length,  # Y
        translation + rotation[:, 2] * axis_length,  # Z
    ]

    def project(pt3d):
        if pt3d[2] <= 0:
            return None
        u = fx * pt3d[0] / pt3d[2] + cx
        v = fy * pt3d[1] / pt3d[2] + cy
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        u_viz = (u - x1) / crop_w * viz_w
        v_viz = (v - y1) / crop_h * viz_h
        return (u_viz, v_viz)

    origin_2d = project(origin)
    if origin_2d is None:
        return

    colors = ["red", "green", "blue"]
    for ax, color in zip(axes_3d, colors):
        end_2d = project(ax)
        if end_2d is not None:
            draw.line([origin_2d, end_2d], fill=color, width=line_width)


def main():
    parser = argparse.ArgumentParser(description="Visualize keypoint and pose predictions")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to visualize")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of images to visualize")
    parser.add_argument("--out_dir", type=str, default="outputs/viz", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]

    # Build dataset (no augmentation)
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][args.split]
    transform = KeypointTransform(image_size=config["data"]["image_size"], is_train=False)

    pose_json = None
    if mode != "keypoint_only":
        pose_labels = config["data"].get("pose_labels", {})
        if args.split in pose_labels:
            pose_json = pose_labels[args.split]

    # Use test-only split if FDA split files exist
    include_list = None
    if args.split in ("lightbox", "sunlamp"):
        fda_cfg = config.get("fda", {})
        splits_dir = Path(fda_cfg.get("splits_dir", "data/splits"))
        test_list_path = splits_dir / f"{args.split}_test.txt"
        if test_list_path.exists():
            with open(test_list_path) as f:
                include_list = set(line.strip() for line in f if line.strip())
            print(f"  Using test split: {len(include_list)} images")

    dataset = SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=include_list,
    )

    # Raw dataset for visualization images
    raw_dataset = SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=None,
        pose_json=pose_json,
        include_list=include_list,
    )

    # Load camera matrix for pose axis drawing
    camera_matrix = None
    geo_cfg = config.get("geometry", {})
    if mode != "keypoint_only" and geo_cfg.get("camera"):
        with open(geo_cfg["camera"], "r") as f:
            cam_data = json.load(f)
        camera_matrix = np.array(cam_data["cameraMatrix"], dtype=np.float32)

    # Load model
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

    # Sample indices
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(indices)} samples from '{args.split}' split (mode={mode})...")

    viz_size = 448

    for idx in indices:
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        gt_kp = sample["keypoints"].numpy()
        vis = sample["visibility"].numpy()
        crop_box = sample["crop_box"].numpy()

        # Build forward kwargs
        fwd_kwargs = {"pixel_values": image_tensor}
        if mode == "keypoint_pose_pnp":
            fwd_kwargs["crop_box"] = sample["crop_box"].unsqueeze(0).to(device)
            fwd_kwargs["img_size"] = sample["img_size"].unsqueeze(0).to(device)
            fwd_kwargs["visibility"] = sample["visibility"].unsqueeze(0).to(device)

        with torch.no_grad():
            model_out = model(**fwd_kwargs)

        pred_kp = model_out["keypoints"].cpu().squeeze(0).numpy()

        # Get raw crop for drawing
        raw_sample = raw_dataset[idx]
        raw_crop = raw_sample["image"]
        raw_crop = raw_crop.resize((viz_size, viz_size))

        # Draw keypoints
        viz = visualize_keypoints(raw_crop, pred_kp, gt_kp, vis, radius=5)

        # Draw predicted pose axes
        if camera_matrix is not None and "direct_rotation" in model_out:
            pred_R = model_out["direct_rotation"].cpu().squeeze(0).numpy()
            pred_t = model_out["direct_translation"].cpu().squeeze(0).numpy()
            draw = ImageDraw.Draw(viz)
            draw_axes(draw, pred_R, pred_t, camera_matrix, crop_box, viz_size, viz_size,
                      axis_length=0.3, line_width=3)

            # Draw GT pose axes (thinner, shorter) if available
            if sample["has_pose"]:
                gt_q = sample["quaternion"].numpy()
                gt_R = quaternion_to_matrix(
                    torch.tensor(gt_q).unsqueeze(0)
                ).squeeze(0).numpy()
                gt_t = sample["translation"].numpy()
                draw_axes(draw, gt_R, gt_t, camera_matrix, crop_box, viz_size, viz_size,
                          axis_length=0.2, line_width=1)

        viz.save(out_dir / f"{args.split}_{idx:05d}.png")

        # Save heatmap overlay if available
        if "heatmaps" in model_out:
            hm_viz = visualize_heatmaps(
                raw_crop,
                model_out["heatmaps"].cpu().squeeze(0).numpy(),
            )
            hm_viz.save(out_dir / f"{args.split}_{idx:05d}_heatmap.png")

    print(f"Saved {len(indices)} visualizations to {out_dir}/")


if __name__ == "__main__":
    main()
