"""Run inference on images and visualize predicted keypoints and pose.

Usage:
    python inference.py --config config.yaml --checkpoint outputs/best_model.pth --split val --num_samples 20 --out_dir outputs/viz
"""

import argparse
import random
from pathlib import Path

import torch
import yaml

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.utils import visualize_keypoints, visualize_heatmaps


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

    # Load model
    geo_cfg = config.get("geometry", {})
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
        if mode == "keypoint_pnp":
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
