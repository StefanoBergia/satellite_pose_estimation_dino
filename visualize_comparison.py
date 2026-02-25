"""Side-by-side inference comparison of two models on the same images.

Loads two models (e.g. DINO vs HRNet-W32), runs inference on the same
sampled images, and produces side-by-side visualizations with per-model
metrics. Generates individual comparison images and summary grids per split.

Usage:
    python visualize_comparison.py \
        --config_a config.yaml \
        --checkpoint_a outputs_keypoints_heatmap_FDA/best_model.pth \
        --config_b config_hrnet_w32.yaml \
        --checkpoint_b outputs_hrnet_w32/converted_model.pth \
        --splits val lightbox sunlamp \
        --num_samples 10 \
        --out_dir outputs/viz_comparison
"""

import argparse
import random
from pathlib import Path

import cv2
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
)
from evaluate_robust import solve_pnp_robust
from visualize_inference import (
    crop_to_full_keypoints,
    draw_on_full_image,
    draw_wireframe_comparison,
    compute_sample_metrics,
)


def build_model(config, checkpoint_path, device):
    """Build and load a SatellitePoseModel from config + checkpoint."""
    mode = config["model"]["mode"]
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
        backbone_type=config["model"].get("backbone_type", "dinov3"),
        hrnet_pretrained=config["model"].get("hrnet_pretrained", None),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    return model, epoch


def build_transform(config):
    """Build inference transform from a config."""
    return KeypointTransform(
        image_size=config["data"]["image_size"],
        is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )


def build_dataset(config, split, include_list=None, transform=None):
    """Build a SpeedPlusKeypointDataset for a given split."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][split]
    pose_labels = config["data"].get("pose_labels", {})
    pose_json = pose_labels.get(split)

    return SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=include_list,
    )


def get_include_list(config, split):
    """Get the test-only include list for lightbox/sunlamp splits."""
    if split not in ("lightbox", "sunlamp"):
        return None
    fda_cfg = config.get("fda", {})
    splits_dir = Path(fda_cfg.get("splits_dir", "data/splits"))
    test_list_path = splits_dir / f"{split}_test.txt"
    if test_list_path.exists():
        with open(test_list_path) as f:
            return set(line.strip() for line in f if line.strip())
    return None


def model_label(config):
    """Derive a short label from the backbone_type."""
    bt = config["model"].get("backbone_type", "dinov3")
    labels = {
        "dinov3": "DINO",
        "hrnet": "HRNet-W48",
        "hrnet_w32": "HRNet-W32",
    }
    return labels.get(bt, bt)


def run_inference(model, sample, mode, device):
    """Run a single forward pass and return model output dict."""
    image_tensor = sample["image"].unsqueeze(0).to(device)
    fwd_kwargs = {"pixel_values": image_tensor}
    if mode == "keypoint_pnp":
        fwd_kwargs["crop_box"] = sample["crop_box"].unsqueeze(0).to(device)
        fwd_kwargs["img_size"] = sample["img_size"].unsqueeze(0).to(device)
        fwd_kwargs["visibility"] = sample["visibility"].unsqueeze(0).to(device)

    with torch.no_grad():
        return model(**fwd_kwargs)


def solve_pose(model_out, sample, points_3d, camera_matrix, dist_coeffs):
    """Solve EPnP from predicted keypoints. Returns (pred_R, pred_t) or (None, None)."""
    if points_3d is None:
        return None, None
    kp_tensor = model_out["keypoints"].cpu()
    cb_tensor = sample["crop_box"].unsqueeze(0)
    vis_tensor = sample["visibility"].unsqueeze(0)
    confidence = None
    if "heatmaps" in model_out:
        hm = model_out["heatmaps"].cpu().squeeze(0).numpy()
        confidence = hm.max(axis=(1, 2)).reshape(1, -1)
    rotations, translations, success, _ = solve_pnp_batch(
        kp_tensor, cb_tensor, vis_tensor,
        points_3d, camera_matrix, dist_coeffs,
        confidence=confidence,
    )
    if success[0]:
        return rotations[0], translations[0]
    return None, None


def _kp_to_full_px(pred_kp_norm, crop_box, img_size):
    """Convert (K,2) normalised crop-coords to full-image pixel coords."""
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = x2 - x1, y2 - y1
    px = np.zeros_like(pred_kp_norm)
    px[:, 0] = pred_kp_norm[:, 0] * crop_w + x1
    px[:, 1] = pred_kp_norm[:, 1] * crop_h + y1
    return px


def solve_pose_robust(model_out, sample, points_3d, camera_matrix, dist_coeffs):
    """Full refinement pipeline (EPnP RANSAC + cascading inliers + LM retrim).

    Mirrors evaluate_robust.py defaults. Returns (pred_R, pred_t) or (None, None).
    """
    if points_3d is None:
        return None, None

    pred_kp = model_out["keypoints"].cpu().squeeze(0).numpy()  # (K,2) normalised
    crop_box = sample["crop_box"].numpy()
    vis = sample["visibility"].numpy()

    kp_px = _kp_to_full_px(pred_kp, crop_box, sample["img_size"].numpy())

    confidence = None
    if "heatmaps" in model_out:
        hm = model_out["heatmaps"].cpu().squeeze(0).numpy()
        confidence = hm.max(axis=(1, 2))  # (K,)

    res = solve_pnp_robust(
        kp_px=kp_px,
        visibility=vis,
        confidence=confidence,
        points_3d=points_3d,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        min_landmarks=6,
        reproj_error=8.0,
        confidence_threshold=0.3,
        min_inliers_schedule=[11, 9, 8, 6, 4],
        refine_lm=True,
        refine_retrim=True,
        refine_keep_frac=0.8,
        refine_min_keep=6,
    )
    if res["success"]:
        return res["R"], res["t"]
    return None, None


def compute_sample_metrics_robust(model_out, sample, points_3d, camera_matrix, dist_coeffs):
    """Like compute_sample_metrics but uses solve_pnp_robust for pose metrics."""
    metrics = {}

    pred_kp = model_out["keypoints"].cpu()
    gt_kp = sample["keypoints"].unsqueeze(0)
    vis = sample["visibility"].unsqueeze(0)
    crop_box = sample["crop_box"].unsqueeze(0)

    px_err = compute_pixel_error(pred_kp, gt_kp, vis, crop_box)
    metrics["px_err"] = f"{px_err.item():.1f}px"

    if points_3d is not None and sample["has_pose"]:
        pred_R, pred_t = solve_pose_robust(
            model_out, sample, points_3d, camera_matrix, dist_coeffs)
        if pred_R is not None:
            pred_R_t = torch.from_numpy(pred_R).unsqueeze(0).float()
            pred_t_t = torch.from_numpy(pred_t).unsqueeze(0).float()
            gt_q = sample["quaternion"].unsqueeze(0)
            gt_t = sample["translation"].unsqueeze(0)
            has_pose = torch.tensor([True])

            rot_err = compute_rotation_error(pred_R_t, gt_q, has_pose)
            trans_err = compute_translation_error(pred_t_t, gt_t, has_pose)
            metrics["rot(LM)"] = f"{rot_err.item():.1f}\u00b0"
            metrics["t(LM)"] = f"{trans_err.item():.3f}m"

    return metrics


def draw_panel(full_display, pred_kp, gt_kp, vis, crop_box, scale,
               pred_R, pred_t, gt_R, gt_t, camera_matrix,
               label, metrics, display_h, panel_w):
    """Draw one model's annotated panel."""
    viz = draw_on_full_image(
        full_display, pred_kp, gt_kp, vis, crop_box,
        pred_R=pred_R, pred_t=pred_t,
        gt_R=gt_R, gt_t=gt_t,
        camera_matrix=camera_matrix, scale=scale, radius=5,
    )
    draw = ImageDraw.Draw(viz)

    # Top label bar
    draw.rectangle([0, 0, panel_w, 22], fill=(0, 0, 0))
    draw.text((4, 4), label, fill="yellow")

    # Bottom metric bar
    metric_text = "  ".join(f"{k}={v}" for k, v in metrics.items())
    draw.rectangle([0, display_h - 20, panel_w, display_h], fill=(0, 0, 0))
    draw.text((4, display_h - 18), metric_text, fill="white")

    return viz


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side inference comparison of two models")
    parser.add_argument("--config_a", type=str, required=True)
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--config_b", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="outputs/viz_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display_width", type=int, default=960,
                        help="Width of each panel (total image is 2x this)")
    args = parser.parse_args()

    # Load configs
    with open(args.config_a) as f:
        config_a = yaml.safe_load(f)
    with open(args.config_b) as f:
        config_b = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_a = model_label(config_a)
    label_b = model_label(config_b)
    print(f"Model A: {label_a}  ({args.checkpoint_a})")
    print(f"Model B: {label_b}  ({args.checkpoint_b})")

    # Load models
    print("Loading models...")
    model_a, epoch_a = build_model(config_a, args.checkpoint_a, device)
    model_b, epoch_b = build_model(config_b, args.checkpoint_b, device)
    print(f"  A: epoch {epoch_a}  |  B: epoch {epoch_b}")

    # Load PnP data (shared geometry — use config_a)
    camera_matrix = points_3d = dist_coeffs = None
    geo_cfg = config_a.get("geometry", {})
    if geo_cfg.get("camera") and geo_cfg.get("points_3d"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
        camera_matrix = pnp_data["camera_matrix"].astype(np.float32)
        points_3d = pnp_data["points_3d"]
        dist_coeffs = pnp_data["dist_coeffs"]

    # Transforms
    transform_a = build_transform(config_a)
    transform_b = build_transform(config_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_a = config_a["model"]["mode"]
    mode_b = config_b["model"]["mode"]

    for split in args.splits:
        if split not in config_a["data"]["splits"]:
            print(f"  Skipping {split} (not in config_a)")
            continue
        if split not in config_b["data"]["splits"]:
            print(f"  Skipping {split} (not in config_b)")
            continue

        # Use the same include list (test split) for both
        include_list = get_include_list(config_a, split)
        if include_list is not None:
            print(f"  [{split}] Using test split: {len(include_list)} images")

        # Build datasets with different transforms but same images
        ds_a = build_dataset(config_a, split, include_list, transform_a)
        ds_b = build_dataset(config_b, split, include_list, transform_b)
        assert len(ds_a) == len(ds_b), (
            f"Dataset sizes differ: {len(ds_a)} vs {len(ds_b)}")

        random.seed(args.seed)
        n_samples = min(args.num_samples, len(ds_a))
        indices = random.sample(range(len(ds_a)), n_samples)

        print(f"\n[{split}] Comparing {n_samples} samples...")

        tiles = []
        for idx in indices:
            sample_a = ds_a[idx]
            sample_b = ds_b[idx]

            # Run both models
            out_a = run_inference(model_a, sample_a, mode_a, device)
            out_b = run_inference(model_b, sample_b, mode_b, device)

            pred_kp_a = out_a["keypoints"].cpu().squeeze(0).numpy()
            pred_kp_b = out_b["keypoints"].cpu().squeeze(0).numpy()

            # Metrics (A: basic EPnP, B: full LM refinement)
            metrics_a = compute_sample_metrics(
                out_a, sample_a, points_3d, camera_matrix, dist_coeffs)
            metrics_b = compute_sample_metrics_robust(
                out_b, sample_b, points_3d, camera_matrix, dist_coeffs)

            # GT data (same for both — use sample_a)
            gt_kp = sample_a["keypoints"].numpy()
            vis = sample_a["visibility"].numpy()
            crop_box_a = sample_a["crop_box"].numpy()
            crop_box_b = sample_b["crop_box"].numpy()

            # Load original full image
            img_path = ds_a.samples[idx][0]
            full_image = Image.open(img_path)
            if full_image.mode == "L":
                full_image = full_image.convert("RGB")

            orig_w, orig_h = full_image.size
            scale = args.display_width / orig_w
            display_h = int(orig_h * scale)
            full_display = full_image.resize(
                (args.display_width, display_h), Image.BILINEAR)

            # Solve pose: A=basic EPnP, B=full LM refinement
            pred_R_a, pred_t_a = solve_pose(
                out_a, sample_a, points_3d, camera_matrix, dist_coeffs)
            pred_R_b, pred_t_b = solve_pose_robust(
                out_b, sample_b, points_3d, camera_matrix, dist_coeffs)

            # GT pose
            gt_R = gt_t = None
            if sample_a["has_pose"]:
                gt_q = sample_a["quaternion"].numpy()
                gt_R = quaternion_to_matrix(
                    torch.tensor(gt_q).unsqueeze(0)
                ).squeeze(0).numpy()
                gt_t = sample_a["translation"].numpy()

            # Draw panels
            panel_a = draw_panel(
                full_display, pred_kp_a, gt_kp, vis, crop_box_a, scale,
                pred_R_a, pred_t_a, gt_R, gt_t, camera_matrix,
                f"{label_a} (ep {epoch_a})", metrics_a, display_h,
                args.display_width)
            panel_b = draw_panel(
                full_display, pred_kp_b, gt_kp, vis, crop_box_b, scale,
                pred_R_b, pred_t_b, gt_R, gt_t, camera_matrix,
                f"{label_b} (ep {epoch_b})", metrics_b, display_h,
                args.display_width)

            # Paste side-by-side
            compare = Image.new("RGB",
                                (args.display_width * 2, display_h),
                                color=(30, 30, 30))
            compare.paste(panel_a, (0, 0))
            compare.paste(panel_b, (args.display_width, 0))

            compare.save(out_dir / f"{split}_{idx:05d}_compare.png")
            tiles.append(compare)

            # Wireframe comparison (one per model)
            if points_3d is not None:
                for tag, pR, pt, lbl in [
                    ("a", pred_R_a, pred_t_a, label_a),
                    ("b", pred_R_b, pred_t_b, label_b),
                ]:
                    if pR is not None or gt_R is not None:
                        wf = draw_wireframe_comparison(
                            full_display, points_3d, camera_matrix, scale,
                            pred_R=pR, pred_t=pt,
                            gt_R=gt_R, gt_t=gt_t)
                        wf_draw = ImageDraw.Draw(wf)
                        wf_draw.rectangle(
                            [0, 0, args.display_width, 22], fill=(0, 0, 0))
                        wf_draw.text((4, 4), f"{lbl} wireframe", fill="yellow")
                        wf.save(out_dir /
                                f"{split}_{idx:05d}_wireframe_{tag}.png")

        # Build summary grid: stacked rows of side-by-side pairs
        if not tiles:
            continue
        tile_w = args.display_width * 2
        tile_h = display_h
        header_h = 28
        n_rows = n_samples
        grid_w = tile_w
        grid_h = n_rows * tile_h + header_h

        grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid)
        draw.text((8, 6),
                  f"{split.upper()}  |  Left: {label_a} (ep {epoch_a})  |  "
                  f"Right: {label_b} (ep {epoch_b})",
                  fill="white")

        for i, tile in enumerate(tiles):
            grid.paste(tile, (0, i * tile_h + header_h))

        grid_path = out_dir / f"grid_{split}.png"
        grid.save(grid_path)
        print(f"  Saved grid to {grid_path}")

    print(f"\nDone. All outputs in {out_dir}/")
    print(f"Legend: red=predicted, green=GT keypoints, yellow=error line, "
          f"cyan=crop bbox")
    if camera_matrix is not None:
        print("        'Pred' thick axes=predicted pose, "
              "'GT' thin pastel axes=ground truth pose")


if __name__ == "__main__":
    main()
