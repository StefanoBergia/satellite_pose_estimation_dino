"""Visualize images that were dropped (PnP failed) by evaluate_robust.py.

Side-by-side annotated visualizations:
  LEFT:  crop inference (the run that failed)
  RIGHT: full-image inference (no bbox crop)

All coordinate transforms are done in the main loop before calling draw functions.
Draw functions receive pre-computed display-space pixel coords.

Usage:
    python visualize_dropped.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits lightbox sunlamp
    python visualize_dropped.py --checkpoint outputs_keypoints_heatmap/best_model.pth --splits val --max_images 20
"""

import argparse
from pathlib import Path

import cv2
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
    """Map confidence [0,1] to color: red (low) -> yellow (mid) -> green (high)."""
    if conf < 0.5:
        r = 255
        g = int(255 * (conf / 0.5))
    else:
        r = int(255 * (1.0 - (conf - 0.5) / 0.5))
        g = 255
    return (r, g, 0)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def crop_to_working(kp_crop, crop_box):
    """Convert crop-relative [0,1] keypoints to working-image pixel coords.

    crop_box: (x1, y1, x2, y2) in working-image pixels
    """
    x1, y1, x2, y2 = crop_box
    kp_px = np.zeros_like(kp_crop)
    kp_px[:, 0] = kp_crop[:, 0] * (x2 - x1) + x1
    kp_px[:, 1] = kp_crop[:, 1] * (y2 - y1) + y1
    return kp_px


def working_to_display(pts, scale_x, scale_y):
    """Scale (N,2) points from working-image space to display space."""
    out = np.zeros_like(pts)
    out[:, 0] = pts[:, 0] * scale_x
    out[:, 1] = pts[:, 1] * scale_y
    return out


def box_to_display(crop_box, scale_x, scale_y):
    """Scale (x1,y1,x2,y2) crop_box from working-image space to display space."""
    x1, y1, x2, y2 = crop_box
    return np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    dtype=np.float32)


def quat_to_dcm(q_wxyz):
    """Convert (w,x,y,z) quaternion to direction cosine matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])


def project_gt_keypoints(q_wxyz, t, points_3d, K):
    """Project 3D model keypoints to 2D using the GT pose.

    q_wxyz: (w,x,y,z) — q_vbs2tango (camera→body), SPEED+ convention
    t: (3,) translation in camera frame
    points_3d: (K,3) 3D model keypoints in body frame
    K: (3,3) camera intrinsics (original image space)

    Returns: (K,2) pixel coords in the ORIGINAL image coordinate space.
    """
    R_body2cam = quat_to_dcm(q_wxyz)  # standard quat→dcm = R_body2cam in SPEED+ convention
    rvec, _ = cv2.Rodrigues(R_body2cam.astype(np.float64))
    kp_2d, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3).astype(np.float64),
        rvec,
        t.reshape(3, 1).astype(np.float64),
        K.astype(np.float64),
        None,
    )
    return kp_2d.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Draw functions — all coords already in display pixels
# ---------------------------------------------------------------------------

def draw_dropped_image(
    full_image, pred_px, gt_px, visibility, box_disp,
    confidence_per_kpt, drop_info, radius=6,
):
    """Draw LEFT panel: crop inference that failed.

    pred_px, gt_px, box_disp: all in display pixel coords.
    """
    img = full_image.copy()
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box_disp
    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)

    n_kpts = len(pred_px)

    # GT keypoints (cyan)
    for i in range(n_kpts):
        gx, gy = gt_px[i, 0], gt_px[i, 1]
        r = radius - 2
        if visibility[i] > 0:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill="cyan", outline="white")
        else:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline="cyan", width=1)

    # Predicted keypoints (color-coded by confidence)
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

    # Info overlay
    disp_w = img.width
    overlay_h = 60
    draw.rectangle([0, 0, disp_w, overlay_h], fill=(0, 0, 0, 180))
    filename = drop_info["filename"]
    method = drop_info["method"]
    n_vis = drop_info["n_visible"]
    n_used = drop_info["n_kpts_used"]
    min_conf = drop_info["min_conf"]
    line1 = f"{filename}  |  reason: {method}  |  visible: {n_vis}/11  |  used: {n_used}  |  min_conf: {min_conf:.4f}"
    draw.text((4, 4), line1, fill="white")

    if confidence_per_kpt is not None:
        bar_y, bar_h = 24, 12
        bar_total_w = min(disp_w - 20, n_kpts * 50)
        bar_w = bar_total_w / n_kpts
        for i in range(n_kpts):
            c = confidence_per_kpt[i]
            color = confidence_color(c)
            bx = 10 + i * bar_w
            fill_h = int(bar_h * min(c / 5.0, 1.0))
            draw.rectangle([bx, bar_y + bar_h - fill_h, bx + bar_w - 2, bar_y + bar_h], fill=color)
            draw.rectangle([bx, bar_y, bx + bar_w - 2, bar_y + bar_h], outline="gray")
            name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
            draw.text((bx, bar_y + bar_h + 2), name, fill="gray")

    disp_h = img.height
    draw.rectangle([0, disp_h - 18, disp_w, disp_h], fill=(0, 0, 0))
    draw.text((4, disp_h - 16),
              "pred: red=low conf, green=high conf | cyan=GT | yellow=error | hollow=occluded",
              fill="gray")
    return img


def draw_fullimg_panel(
    full_display, pred_px, gt_px, visibility, box_disp,
    confidence_per_kpt, pnp_pass, pnp_inliers, radius=6,
):
    """Draw RIGHT panel: full-image inference.

    pred_px, gt_px, box_disp: all in display pixel coords.
    pnp_pass: bool — whether PnP succeeded on full-image keypoints.
    pnp_inliers: int — number of RANSAC inliers.
    """
    img = full_display.copy()
    draw = ImageDraw.Draw(img)
    disp_w, disp_h = img.size

    # Original crop region in orange for reference
    x1, y1, x2, y2 = box_disp
    draw.rectangle([x1, y1, x2, y2], outline="orange", width=2)

    n_kpts = len(pred_px)

    # GT keypoints (cyan)
    for i in range(n_kpts):
        gx, gy = gt_px[i, 0], gt_px[i, 1]
        r = radius - 2
        if visibility[i] > 0:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill="cyan", outline="white")
        else:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline="cyan", width=1)

    # Predicted keypoints (color-coded by confidence)
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

    # Header with PnP pass/fail
    if pnp_pass:
        header_bg = (0, 100, 0)
        pnp_str = f"PASS  ({pnp_inliers} inliers)"
        pnp_color = "lime"
    else:
        header_bg = (120, 0, 0)
        pnp_str = f"FAIL  ({pnp_inliers} inliers)"
        pnp_color = "red"
    draw.rectangle([0, 0, disp_w, 22], fill=header_bg)
    draw.text((4, 4), f"FULL IMAGE  |  PnP: {pnp_str}  |  orange = original crop", fill=pnp_color)

    draw.rectangle([0, disp_h - 18, disp_w, disp_h], fill=(0, 0, 0))
    draw.text((4, disp_h - 16),
              "pred: red=low conf, green=high conf | cyan=GT | yellow=error | hollow=occluded",
              fill="gray")
    return img


def stitch_panels(left, right):
    """Stitch two same-height panels side by side with a thin separator."""
    sep = 4
    combined = Image.new("RGB", (left.width + sep + right.width, left.height), (40, 40, 40))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + sep, 0))
    return combined


def try_pnp(kp_px, visibility, points_3d, camera_matrix, dist_coeffs,
             reproj_error=8.0, iterations=100, ransac_confidence=0.999):
    """Run EPnP RANSAC on predicted keypoints. Returns (success, n_inliers)."""
    vis_mask = visibility > 0
    if vis_mask.sum() < 4:
        return False, 0
    pts_2d = kp_px[vis_mask].reshape(-1, 1, 2).astype(np.float64)
    pts_3d = points_3d[vis_mask].reshape(-1, 1, 3).astype(np.float64)
    cv2.setRNGSeed(42)
    ok, _, _, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, camera_matrix.astype(np.float64),
        dist_coeffs.astype(np.float64) if dist_coeffs is not None else None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=reproj_error,
        iterationsCount=iterations,
        confidence=ransac_confidence,
    )
    if ok and inliers is not None and len(inliers) >= 4:
        return True, len(inliers)
    return False, 0


def infer_fullimage(model, img_path, transform, num_keypoints, device, mode, vis_tensor):
    """Run model inference on the full image without any crop.

    Returns pred_kp in [0,1] relative to the full squashed image.
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    if img.mode == "L":
        img = img.convert("RGB")
    dummy_kp = np.zeros((num_keypoints, 2), dtype=np.float32)
    dummy_vis = np.zeros(num_keypoints, dtype=np.int64)
    img_tensor, _, _ = transform(img, dummy_kp, dummy_vis)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    fwd_kwargs = {"pixel_values": img_tensor}
    if mode == "keypoint_pnp":
        fwd_kwargs["crop_box"] = torch.tensor(
            [[0.0, 0.0, float(orig_w), float(orig_h)]], device=device)
        fwd_kwargs["img_size"] = torch.tensor(
            [[float(orig_w), float(orig_h)]], device=device)
        fwd_kwargs["visibility"] = vis_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(**fwd_kwargs)

    pred_kp = out["keypoints"].cpu().squeeze(0).numpy()
    conf = None
    if "heatmaps" in out:
        hm = out["heatmaps"].cpu().squeeze(0).numpy()
        conf = hm.max(axis=(1, 2))
    return pred_kp, conf


def main():
    parser = argparse.ArgumentParser(description="Visualize dropped images from evaluate_robust")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--display_width", type=int, default=960)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--gt_crop", action="store_true")
    parser.add_argument("--resize_first", action="store_true")
    parser.add_argument("--reproj_error", type=float, default=8.0,
                        help="RANSAC reprojection error for the 'would pass' PnP check (default: 8.0)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "viz_dropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

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
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )

    for split in args.splits:
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
            gt_kp = sample["keypoints"].numpy()        # crop-relative [0,1]
            vis = sample["visibility"].numpy()
            crop_box = sample["crop_box"].numpy()      # working-image pixel coords

            fwd_kwargs = {"pixel_values": image_tensor}
            if mode == "keypoint_pnp":
                fwd_kwargs["crop_box"] = sample["crop_box"].unsqueeze(0).to(device)
                fwd_kwargs["img_size"] = sample["img_size"].unsqueeze(0).to(device)
                fwd_kwargs["visibility"] = sample["visibility"].unsqueeze(0).to(device)

            with torch.no_grad():
                model_out = model(**fwd_kwargs)

            pred_kp = model_out["keypoints"].cpu().squeeze(0).numpy()  # crop-relative [0,1]

            conf_per_kpt = None
            if "heatmaps" in model_out:
                hm = model_out["heatmaps"].cpu().squeeze(0).numpy()
                conf_per_kpt = hm.max(axis=(1, 2))
            if conf_per_kpt is None and info["confidence"] is not None:
                conf_per_kpt = np.array(info["confidence"])

            # ---- Load original image for display background ----
            img_path = dataset.samples[idx][0]
            full_image = Image.open(img_path)
            if full_image.mode == "L":
                full_image = full_image.convert("RGB")

            orig_w, orig_h = full_image.size
            # Scale that maps original image → display (preserves aspect ratio)
            scale = args.display_width / orig_w
            display_h = int(orig_h * scale)
            full_display = full_image.resize((args.display_width, display_h), Image.BILINEAR)

            # ---- Coordinate transforms ----
            #
            # crop_box is in "working image space":
            #   - resize_first=True  → working = image_size × image_size (e.g. 512×512)
            #   - resize_first=False → working = orig_w × orig_h
            #
            # Display image is always orig_w*scale × orig_h*scale.
            # Working→display scale factors:
            if args.resize_first:
                sw_x = args.display_width / image_size     # e.g. 960/512 = 1.875
                sw_y = display_h / image_size              # e.g. 600/512 = 1.172
            else:
                sw_x = scale    # display_w / orig_w
                sw_y = scale    # display_h / orig_h  (same since aspect ratio preserved)

            # Crop box in display coords
            box_disp = box_to_display(crop_box, sw_x, sw_y)

            # Left panel predicted keypoints:
            # crop-relative [0,1] → working-image px → display px
            pred_kp_working = crop_to_working(pred_kp, crop_box)
            pred_px_crop = working_to_display(pred_kp_working, sw_x, sw_y)

            # GT keypoints: project 3D model points with GT pose → original image px → display px
            # This gives the true satellite keypoint positions regardless of crop box placement.
            if sample["has_pose"].item() and pnp_data is not None:
                q = sample["quaternion"].numpy()
                t = sample["translation"].numpy()
                kp_2d_orig = project_gt_keypoints(
                    q, t, pnp_data["points_3d"], pnp_data["camera_matrix"]
                )
                # project_gt_keypoints output is in original image pixel space
                gt_px_disp = kp_2d_orig * scale
            else:
                # Fallback: crop-relative → working → display
                gt_kp_working = crop_to_working(gt_kp, crop_box)
                gt_px_disp = working_to_display(gt_kp_working, sw_x, sw_y)

            # ---- Full-image inference (right panel) ----
            pred_kp_full, conf_full = infer_fullimage(
                model, img_path, transform, num_keypoints, device,
                mode, sample["visibility"],
            )

            # Full-image pred: [0,1] relative to 512×512 squashed image.
            # Mapping to display: [0,1] → orig pixel (via *orig_w, *orig_h) → display (via *scale)
            # = pred * [display_w, display_h]
            pred_px_full = np.stack([
                pred_kp_full[:, 0] * args.display_width,
                pred_kp_full[:, 1] * display_h,
            ], axis=1)

            # ---- Would full-image inference pass PnP? ----
            pnp_pass, pnp_inliers = False, 0
            if pnp_data is not None:
                # Map full-image [0,1] keypoints to original pixel space for PnP
                kp_px_full_orig = np.stack([
                    pred_kp_full[:, 0] * orig_w,
                    pred_kp_full[:, 1] * orig_h,
                ], axis=1)
                pnp_pass, pnp_inliers = try_pnp(
                    kp_px_full_orig, vis,
                    pnp_data["points_3d"], pnp_data["camera_matrix"],
                    pnp_data["dist_coeffs"],
                    reproj_error=args.reproj_error,
                )

            # ---- Draw panels ----
            left_panel = draw_dropped_image(
                full_display, pred_px_crop, gt_px_disp, vis, box_disp,
                conf_per_kpt, info, radius=6,
            )
            right_panel = draw_fullimg_panel(
                full_display, pred_px_full, gt_px_disp, vis, box_disp,
                conf_full, pnp_pass, pnp_inliers, radius=6,
            )

            combined = stitch_panels(left_panel, right_panel)
            stem = Path(filename).stem
            combined.save(split_dir / f"{stem}.png")

            # Heatmap overlay for crop inference side
            if "heatmaps" in model_out:
                bx1, by1, bx2, by2 = box_disp.astype(int)
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2 = min(args.display_width, bx2)
                by2 = min(display_h, by2)
                crop_region = full_display.crop((bx1, by1, bx2, by2))
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

        tile_w = args.display_width * 2 + 4
        header_h = 28
        grid = Image.new("RGB", (tile_w, len(tiles) * tile_h + header_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid)
        draw.text(
            (8, 6),
            f"DROPPED — {split.upper()} — {len(tiles)} images — epoch {epoch}"
            "  |  LEFT: crop inference (failed)  |  RIGHT: full-image inference",
            fill="red",
        )
        for i, tile in enumerate(tiles):
            grid.paste(tile, (0, i * tile_h + header_h))

        grid_path = out_dir / f"grid_dropped_{split}.png"
        grid.save(grid_path)
        print(f"  Saved grid to {grid_path}")
        print(f"  Individual images in {split_dir}/")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
