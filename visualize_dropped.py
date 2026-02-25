"""Visualize images that were dropped (PnP failed) by evaluate_robust.py.

Loads dropped_{split}.txt files, runs inference on those images, and produces
side-by-side annotated visualizations:
  LEFT:  crop inference (the run that failed)
  RIGHT: full-image inference (fallback — satellite always fully in frame)

Usage:
    python visualize_dropped.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits lightbox sunlamp
    python visualize_dropped.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits val --max_images 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.utils import load_pnp_data, visualize_heatmaps


KEYPOINT_NAMES = [
    "TL_top", "TR_top", "BR_top", "BL_top",
    "TL_bot", "TR_bot", "BR_bot", "BL_bot",
    "ant_L", "ant_R", "arm",
]


def parse_dropped_file(path):
    """Parse dropped_{split}.txt and return list of dicts."""
    dropped = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            entry = {
                "filename": parts[0],
                "method": parts[1],
                "n_visible": int(parts[2]),
                "n_kpts_used": int(parts[3]),
                "min_conf": float(parts[4]),
            }
            if len(parts) > 5 and parts[5] != "N/A":
                entry["confidence"] = [float(x) for x in parts[5].split(",")]
            else:
                entry["confidence"] = None
            dropped.append(entry)
    return dropped


def confidence_color(conf):
    """Map confidence [0, 1] to a color: red (low) -> yellow (mid) -> green (high)."""
    if conf < 0.5:
        r = 255
        g = int(255 * (conf / 0.5))
    else:
        r = int(255 * (1.0 - (conf - 0.5) / 0.5))
        g = 255
    return (r, g, 0)


def crop_to_full_keypoints(kp_crop, crop_box):
    """Convert crop-relative [0,1] keypoints to full-image pixel coords."""
    x1, y1, x2, y2 = crop_box
    kp_px = np.zeros_like(kp_crop)
    kp_px[:, 0] = kp_crop[:, 0] * (x2 - x1) + x1
    kp_px[:, 1] = kp_crop[:, 1] * (y2 - y1) + y1
    return kp_px


def infer_fullimage(model, img_path, transform, num_keypoints, device, mode, vis_tensor):
    """Run model inference on the full image without any crop.

    Uses KeypointTransform to resize + normalize — same preprocessing as the
    dataset but without cropping.  Keypoints are returned in [0, 1] relative
    to the full (resized) image.
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    if img.mode == "L":
        img = img.convert("RGB")

    # Reuse the existing transform: resize to image_size and normalize.
    # Pass dummy keypoints (they're unchanged by the transform).
    dummy_kp = np.zeros((num_keypoints, 2), dtype=np.float32)
    dummy_vis = np.zeros(num_keypoints, dtype=np.int64)
    img_tensor, _, _ = transform(img, dummy_kp, dummy_vis)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    fwd_kwargs = {"pixel_values": img_tensor}
    if mode == "keypoint_pnp":
        # Provide a full-image crop box so the internal PnP uses the original K.
        fwd_kwargs["crop_box"] = torch.tensor(
            [[0.0, 0.0, float(orig_w), float(orig_h)]], device=device
        )
        fwd_kwargs["img_size"] = torch.tensor(
            [[float(orig_w), float(orig_h)]], device=device
        )
        fwd_kwargs["visibility"] = vis_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(**fwd_kwargs)

    pred_kp = out["keypoints"].cpu().squeeze(0).numpy()  # (K, 2) in [0, 1]
    conf = None
    if "heatmaps" in out:
        hm = out["heatmaps"].cpu().squeeze(0).numpy()  # (K, H, W)
        conf = hm.max(axis=(1, 2))

    return pred_kp, conf


def draw_crop_panel(
    full_display, pred_kp_crop, gt_kp_crop, visibility, crop_box,
    confidence_per_kpt, drop_info, scale, radius=6,
):
    """Draw the LEFT panel: crop inference that failed.

    Predicted keypoints are color-coded by confidence (red=low, green=high).
    GT keypoints are shown in cyan. Occluded keypoints are shown as hollow circles.
    """
    img = full_display.copy()
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = crop_box * scale

    # Draw crop bounding box
    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)

    # Convert keypoints to full-image pixel coords (scaled)
    pred_px = crop_to_full_keypoints(pred_kp_crop, crop_box) * scale
    gt_px = crop_to_full_keypoints(gt_kp_crop, crop_box) * scale

    n_kpts = len(pred_kp_crop)

    # Draw GT keypoints (cyan, smaller)
    for i in range(n_kpts):
        gx, gy = gt_px[i, 0], gt_px[i, 1]
        r = radius - 2
        if visibility[i] > 0:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r],
                         fill="cyan", outline="white")
        else:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r],
                         outline="cyan", width=1)

    # Draw predicted keypoints (color-coded by confidence)
    for i in range(n_kpts):
        px, py = pred_px[i, 0], pred_px[i, 1]
        conf = confidence_per_kpt[i] if confidence_per_kpt is not None else 0.5
        color = confidence_color(conf)

        if visibility[i] > 0:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill=color, outline="white")
            gx, gy = gt_px[i, 0], gt_px[i, 1]
            draw.line([(px, py), (gx, gy)], fill="yellow", width=1)
        else:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         outline=color, width=2)

        conf_str = f"{conf:.2f}" if confidence_per_kpt is not None else "?"
        label = f"{i}:{conf_str}"
        draw.text((px + radius + 2, py - 6), label, fill="white")

    # Info overlay at top
    disp_w = img.width
    overlay_h = 60
    draw.rectangle([0, 0, disp_w, overlay_h], fill=(80, 0, 0))

    filename = drop_info["filename"]
    method = drop_info["method"]
    n_vis = drop_info["n_visible"]
    n_used = drop_info["n_kpts_used"]
    min_conf = drop_info["min_conf"]

    draw.text((4, 4), "CROP INFERENCE (FAILED)", fill="red")
    line2 = f"{filename}  |  reason: {method}  |  visible: {n_vis}/11  |  used: {n_used}  |  min_conf: {min_conf:.4f}"
    draw.text((4, 20), line2, fill="white")

    # Confidence bar for each keypoint
    if confidence_per_kpt is not None:
        bar_y = 40
        bar_h = 10
        bar_total_w = min(disp_w - 20, n_kpts * 50)
        bar_w = bar_total_w / n_kpts
        for i in range(n_kpts):
            c = confidence_per_kpt[i]
            color = confidence_color(c)
            bx = 10 + i * bar_w
            fill_h = int(bar_h * min(c / 5.0, 1.0))
            draw.rectangle([bx, bar_y + bar_h - fill_h, bx + bar_w - 2, bar_y + bar_h],
                           fill=color)
            draw.rectangle([bx, bar_y, bx + bar_w - 2, bar_y + bar_h],
                           outline="gray")

    # Legend at bottom
    disp_h = img.height
    draw.rectangle([0, disp_h - 18, disp_w, disp_h], fill=(0, 0, 0))
    draw.text((4, disp_h - 16),
              "pred: red=low conf, green=high | cyan=GT | yellow=error | hollow=occluded",
              fill="gray")

    return img


def draw_fullimg_panel(
    full_display, pred_kp_full, gt_kp_crop, visibility, crop_box,
    confidence_per_kpt, scale, radius=6,
):
    """Draw the RIGHT panel: full-image inference (no crop).

    Predicted keypoints are in [0, 1] relative to the full (resized) image.
    The original crop region is shown in orange for reference.
    GT keypoints are re-derived from crop-relative coords + crop_box.
    """
    img = full_display.copy()
    draw = ImageDraw.Draw(img)
    disp_w, disp_h = img.size

    # Show original crop region in orange for reference
    x1, y1, x2, y2 = crop_box * scale
    draw.rectangle([x1, y1, x2, y2], outline="orange", width=2)

    # Map predicted keypoints to display space (full image relative)
    pred_px = np.stack([
        pred_kp_full[:, 0] * disp_w,
        pred_kp_full[:, 1] * disp_h,
    ], axis=1)

    # GT keypoints in display space (derived from crop_box)
    gt_px = crop_to_full_keypoints(gt_kp_crop, crop_box) * scale

    n_kpts = len(pred_kp_full)

    # Draw GT keypoints (cyan)
    for i in range(n_kpts):
        gx, gy = gt_px[i, 0], gt_px[i, 1]
        r = radius - 2
        if visibility[i] > 0:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r],
                         fill="cyan", outline="white")
        else:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r],
                         outline="cyan", width=1)

    # Draw predicted keypoints (color-coded by confidence)
    for i in range(n_kpts):
        px, py = pred_px[i, 0], pred_px[i, 1]
        conf = confidence_per_kpt[i] if confidence_per_kpt is not None else 0.5
        color = confidence_color(conf)

        if visibility[i] > 0:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill=color, outline="white")
            gx, gy = gt_px[i, 0], gt_px[i, 1]
            draw.line([(px, py), (gx, gy)], fill="yellow", width=1)
        else:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         outline=color, width=2)

        conf_str = f"{conf:.2f}" if confidence_per_kpt is not None else "?"
        draw.text((px + radius + 2, py - 6), f"{i}:{conf_str}", fill="white")

    # Header
    draw.rectangle([0, 0, disp_w, 22], fill=(0, 80, 0))
    draw.text((4, 4), "FULL IMAGE INFERENCE", fill="lime")

    # Legend at bottom
    draw.rectangle([0, disp_h - 18, disp_w, disp_h], fill=(0, 0, 0))
    draw.text((4, disp_h - 16),
              "orange = original crop region | cyan = GT | yellow = pred→GT error",
              fill="gray")

    return img


def stitch_panels(left, right):
    """Stitch two same-height panels side by side with a thin separator."""
    assert left.height == right.height
    sep = 4
    combined = Image.new("RGB", (left.width + sep + right.width, left.height), (40, 40, 40))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + sep, 0))
    return combined


def main():
    parser = argparse.ArgumentParser(description="Visualize dropped images from evaluate_robust")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <ckpt_dir>/viz_dropped)")
    parser.add_argument("--display_width", type=int, default=960)
    parser.add_argument("--max_images", type=int, default=0, help="Max images to visualize per split (0=all)")
    parser.add_argument("--no_crop", action="store_true", help="Match evaluate_robust --no_crop flag")
    parser.add_argument("--gt_crop", action="store_true", help="Match evaluate_robust --gt_crop flag")
    parser.add_argument("--resize_first", action="store_true", help="Match evaluate_robust --resize_first flag")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "viz_dropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and config
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    epoch = ckpt.get("epoch", "?")
    mode = config["model"]["mode"]
    root = Path(config["data"]["root"])
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})
    image_size = config["data"]["image_size"]
    num_keypoints = config["data"]["num_keypoints"]
    imagenet_normalize = config["data"].get("imagenet_normalize", True)

    # Load 3D model data
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

    # Build model
    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=True,
        head_hidden_dims=config["model"]["head_hidden"],
        num_keypoints=num_keypoints,
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
        print(f"  Ignoring {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()

    transform = KeypointTransform(
        image_size=image_size,
        is_train=False,
        imagenet_normalize=imagenet_normalize,
    )

    for split in args.splits:
        # Check for dropped file
        dropped_path = ckpt_dir / f"dropped_{split}.txt"
        if not dropped_path.exists():
            print(f"  No dropped file for {split} ({dropped_path}), skipping")
            continue

        dropped = parse_dropped_file(dropped_path)
        if not dropped:
            print(f"  No dropped images for {split}")
            continue

        if args.max_images > 0:
            dropped = dropped[:args.max_images]

        dropped_filenames = {d["filename"] for d in dropped}
        drop_info_map = {d["filename"]: d for d in dropped}

        print(f"\n{'':=<80}")
        print(f"  {split}: {len(dropped)} dropped images to visualize")
        print(f"{'':=<80}")

        # Load dataset (for crop inference side)
        if split not in config["data"]["splits"]:
            print(f"  Split {split} not in config, skipping")
            continue

        split_cfg = config["data"]["splits"][split]
        pose_json = config["data"].get("pose_labels", {}).get(split)

        dataset = SpeedPlusKeypointDataset(
            image_dir=str(root / split_cfg["images"]),
            label_dir=str(root / split_cfg["labels"]),
            num_keypoints=num_keypoints,
            bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
            transform=transform,
            pose_json=pose_json,
            include_list=dropped_filenames,
            no_crop=args.no_crop,
            gt_crop=args.gt_crop,
            resize_first=image_size if args.resize_first else 0,
        )

        print(f"  Loaded {len(dataset)} images from dataset")

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        tiles = []
        tile_h = None

        for idx in range(len(dataset)):
            sample = dataset[idx]
            filename = sample["filename"]
            if filename not in drop_info_map:
                continue

            info = drop_info_map[filename]

            # ---- Crop inference (left panel) ----
            image_tensor = sample["image"].unsqueeze(0).to(device)
            gt_kp = sample["keypoints"].numpy()
            vis = sample["visibility"].numpy()
            crop_box = sample["crop_box"].numpy()

            fwd_kwargs = {"pixel_values": image_tensor}
            if mode == "keypoint_pnp":
                fwd_kwargs["crop_box"] = sample["crop_box"].unsqueeze(0).to(device)
                fwd_kwargs["img_size"] = sample["img_size"].unsqueeze(0).to(device)
                fwd_kwargs["visibility"] = sample["visibility"].unsqueeze(0).to(device)

            with torch.no_grad():
                model_out = model(**fwd_kwargs)

            pred_kp_crop = model_out["keypoints"].cpu().squeeze(0).numpy()

            conf_crop = None
            if "heatmaps" in model_out:
                hm = model_out["heatmaps"].cpu().squeeze(0).numpy()  # (K, H, W)
                conf_crop = hm.max(axis=(1, 2))

            if conf_crop is None and info["confidence"] is not None:
                conf_crop = np.array(info["confidence"])

            # Load original full image for display
            img_path = dataset.samples[idx][0]
            full_image = Image.open(img_path)
            if full_image.mode == "L":
                full_image = full_image.convert("RGB")

            orig_w, orig_h = full_image.size
            scale = args.display_width / orig_w
            display_h = int(orig_h * scale)
            full_display = full_image.resize((args.display_width, display_h), Image.BILINEAR)

            left_panel = draw_crop_panel(
                full_display, pred_kp_crop, gt_kp, vis, crop_box,
                conf_crop, info, scale, radius=6,
            )

            # ---- Full-image inference (right panel) ----
            pred_kp_full, conf_full = infer_fullimage(
                model, img_path, transform, num_keypoints, device,
                mode, sample["visibility"],
            )

            right_panel = draw_fullimg_panel(
                full_display, pred_kp_full, gt_kp, vis, crop_box,
                conf_full, scale, radius=6,
            )

            # ---- Stitch and save ----
            combined = stitch_panels(left_panel, right_panel)
            stem = Path(filename).stem
            combined.save(split_dir / f"{stem}.png")

            # Save heatmap overlay for the crop inference side (if available)
            if "heatmaps" in model_out:
                sx1, sy1, sx2, sy2 = (crop_box * scale).astype(int)
                sx1, sy1 = max(0, sx1), max(0, sy1)
                sx2 = min(args.display_width, sx2)
                sy2 = min(display_h, sy2)
                crop_region = full_display.crop((sx1, sy1, sx2, sy2))
                hm_viz = visualize_heatmaps(
                    crop_region,
                    model_out["heatmaps"].cpu().squeeze(0).numpy(),
                )
                hm_viz.save(split_dir / f"{stem}_heatmap.png")

            tiles.append(combined)
            tile_h = display_h

        if not tiles:
            print(f"  No images visualized for {split}")
            continue

        # Build summary grid (1 column, each tile is the full side-by-side comparison)
        tile_w = args.display_width * 2 + 4  # left + sep + right
        n_rows = len(tiles)
        header_h = 28
        grid_w = tile_w
        grid_h = n_rows * tile_h + header_h

        grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid)
        draw.text((8, 6),
                  f"DROPPED — {split.upper()} — {len(tiles)} images — epoch {epoch}  |  LEFT: crop failed  |  RIGHT: full-image fallback",
                  fill="red")

        for i, tile in enumerate(tiles):
            grid.paste(tile, (0, i * tile_h + header_h))

        grid_path = out_dir / f"grid_dropped_{split}.png"
        grid.save(grid_path)
        print(f"  Saved grid to {grid_path}")
        print(f"  Individual images in {split_dir}/")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
