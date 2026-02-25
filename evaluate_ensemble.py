"""Ensemble evaluation: confidence-weighted keypoint fusion from HRNet + DINO.

For each image:
  1. Run both models -> per-keypoint predictions in [0,1] + heatmap confidence
  2. Map both to full-image pixel space (common coordinate frame)
  3. Fuse: weighted average by confidence per keypoint
  4. Run PnP once on fused keypoints with original camera matrix

Coordinate alignment:
  HRNet (resize_first + gt_crop): crop_box is in 512-space, so we map
    [0,1] -> 512-space pixels -> original pixels via (orig_size / 512).
  DINO (YOLO bbox crop): crop_box is in original-space, so we map
    [0,1] -> original pixels directly.

Usage:
    python evaluate_ensemble.py \
        --hrnet_checkpoint outputs_hrnet_w32/converted_model.pth \
        --dino_checkpoint outputs_self_training_dg/final_model.pth
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.utils import load_pnp_data
from evaluate_robust import (
    solve_pnp_robust,
    compute_pose_errors,
    print_selection_report,
    print_results_table,
    save_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model_and_config(ckpt_path, device):
    """Load checkpoint -> model, config, epoch."""
    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    epoch = ckpt.get("epoch", "?")
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
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if unexpected:
        print(f"    Ignoring {len(unexpected)} unexpected keys "
              "(e.g. DSU/MixStyle modules)")
    model.to(device)
    model.eval()

    backbone = config["model"].get("backbone_type", "dinov3")
    print(f"    backbone={backbone}  mode={mode}  epoch={epoch}  "
          f"img_size={config['data']['image_size']}")

    return model, config, epoch


def make_dataset(config, split, transform, gt_crop=False, resize_first=0,
                 test_split=False, splits_dir="data/splits"):
    """Create a dataset for a given split with the specified crop pipeline."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][split]
    pose_json = config["data"].get("pose_labels", {}).get(split)

    include_list = None
    if split in ("lightbox", "sunlamp") and test_split:
        test_list_path = Path(splits_dir) / f"{split}_test.txt"
        if test_list_path.exists():
            with open(test_list_path) as f:
                include_list = set(line.strip() for line in f if line.strip())
            print(f"    {split}: test-only subset ({len(include_list)} images)")

    return SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=include_list,
        gt_crop=gt_crop,
        resize_first=resize_first,
    )


@torch.no_grad()
def run_inference(model, batch, device, use_argmax=False):
    """Run model forward pass -> keypoints [0,1] + confidence.

    Args:
        model: SatellitePoseModel in eval mode.
        batch: dict from DataLoader.
        device: torch device.
        use_argmax: if True, extract keypoints via hard argmax on heatmaps.

    Returns:
        kp: (B, K, 2) numpy in [0,1] crop-relative.
        conf: (B, K) numpy heatmap peak confidence, or None if no heatmaps.
    """
    images = batch["image"].to(device)
    model_out = model(pixel_values=images)

    # Override with argmax extraction if requested (matches HRNet eval pipeline)
    if use_argmax and "heatmaps" in model_out:
        hm = model_out["heatmaps"]  # (B, K, H, W)
        B_hm, K_hm, H_hm, W_hm = hm.shape
        flat = hm.view(B_hm, K_hm, -1)
        idx = torch.argmax(flat, dim=-1)  # (B, K)
        y_hm = (idx // W_hm).to(torch.float32)
        x_hm = (idx % W_hm).to(torch.float32)
        coords_argmax = torch.stack([
            x_hm / (W_hm - 1),
            y_hm / (H_hm - 1),
        ], dim=-1)  # (B, K, 2)
        model_out["keypoints"] = coords_argmax

    kp = model_out["keypoints"].detach().cpu().numpy()  # (B, K, 2)

    conf = None
    if "heatmaps" in model_out:
        hm = model_out["heatmaps"].detach().cpu().numpy()
        conf = hm.max(axis=(2, 3))  # (B, K)

    return kp, conf


def kp_to_full_image_pixels(kp, crop_box, img_size, resize_first=0):
    """Map [0,1] crop-relative keypoints to original full-image pixel space.

    Args:
        kp: (B, K, 2) keypoints in [0,1] crop-relative coords.
        crop_box: (B, 4) [x1, y1, x2, y2] —
            in resize_first-space if resize_first > 0, else original-space.
        img_size: (B, 2) original image (w, h).
        resize_first: if > 0, crop_box lives in this resized space
            and needs scaling to original.

    Returns:
        (B, K, 2) keypoints in original full-image pixel coordinates.
    """
    x1 = crop_box[:, 0:1]  # (B, 1)
    y1 = crop_box[:, 1:2]
    x2 = crop_box[:, 2:3]
    y2 = crop_box[:, 3:4]
    crop_w = x2 - x1
    crop_h = y2 - y1

    # [0,1] -> pixels in the crop_box coordinate space
    px = np.zeros_like(kp)
    px[:, :, 0] = kp[:, :, 0] * crop_w + x1
    px[:, :, 1] = kp[:, :, 1] * crop_h + y1

    # If crop_box was in resized space, scale to original
    if resize_first > 0:
        orig_w = img_size[:, 0:1]  # (B, 1)
        orig_h = img_size[:, 1:2]
        px[:, :, 0] *= orig_w / resize_first
        px[:, :, 1] *= orig_h / resize_first

    return px


def fuse_keypoints(kp_a, conf_a, kp_b, conf_b):
    """Confidence-weighted keypoint fusion.

    fused[k] = (conf_a[k]*kp_a[k] + conf_b[k]*kp_b[k]) / (conf_a[k]+conf_b[k])
    fused_conf[k] = max(conf_a[k], conf_b[k])

    Args:
        kp_a, kp_b: (B, K, 2) in full-image pixels.
        conf_a, conf_b: (B, K) per-keypoint confidence.

    Returns:
        fused_kp: (B, K, 2)
        fused_conf: (B, K)
    """
    ca = conf_a[:, :, None]  # (B, K, 1)
    cb = conf_b[:, :, None]
    denom = ca + cb + 1e-8
    fused_kp = (ca * kp_a + cb * kp_b) / denom
    fused_conf = np.maximum(conf_a, conf_b)
    return fused_kp, fused_conf


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_ensemble_split(
    hrnet_model, dino_model,
    hrnet_loader, dino_loader,
    device, pnp_data,
    hrnet_resize_first,
    min_landmarks, reproj_error, confidence_threshold,
    min_inliers_schedule, refine_lm, refine_retrim,
    refine_keep_frac, refine_min_keep,
    ransac_iterations, ransac_confidence,
    sanity_check=0,
):
    """Run ensemble evaluation on one split.

    Returns:
        all_pnp_results, all_pose_errors, method_counts
    """
    all_pnp_results = []
    all_pose_errors = []
    method_counts = {"threshold": 0, "top-N": 0, "all_visible": 0, "failed": 0}
    sanity_count = 0

    for hrnet_batch, dino_batch in tqdm(
        zip(hrnet_loader, dino_loader),
        total=len(hrnet_loader),
        desc="Evaluating",
        leave=False,
    ):
        B = hrnet_batch["image"].size(0)

        # --- Run both models ---
        hrnet_kp, hrnet_conf = run_inference(
            hrnet_model, hrnet_batch, device, use_argmax=True
        )
        dino_kp, dino_conf = run_inference(
            dino_model, dino_batch, device, use_argmax=False
        )

        # --- Map to full-image pixel space ---
        hrnet_crop = hrnet_batch["crop_box"].numpy()
        dino_crop = dino_batch["crop_box"].numpy()
        img_size = hrnet_batch["img_size"].numpy()

        hrnet_px = kp_to_full_image_pixels(
            hrnet_kp, hrnet_crop, img_size, resize_first=hrnet_resize_first
        )
        dino_px = kp_to_full_image_pixels(
            dino_kp, dino_crop, img_size, resize_first=0
        )

        # Default confidence = 1.0 if no heatmaps
        if hrnet_conf is None:
            hrnet_conf = np.ones((B, hrnet_kp.shape[1]), dtype=np.float32)
        if dino_conf is None:
            dino_conf = np.ones((B, dino_kp.shape[1]), dtype=np.float32)

        # --- Fuse ---
        fused_px, fused_conf = fuse_keypoints(
            hrnet_px, hrnet_conf, dino_px, dino_conf
        )

        # --- Sanity check prints ---
        if sanity_check > 0 and sanity_count < sanity_check:
            for b in range(min(B, sanity_check - sanity_count)):
                print(f"\n  --- Sample {sanity_count} ---")
                print(f"  img_size (orig): {img_size[b]}")
                print(f"  HRNet crop_box ({hrnet_resize_first}-space): "
                      f"{hrnet_crop[b]}")
                print(f"  DINO  crop_box (orig-space): {dino_crop[b]}")
                for k in range(min(3, fused_px.shape[1])):
                    print(f"    KP[{k}]: HRNet={hrnet_px[b, k]}  "
                          f"DINO={dino_px[b, k]}  "
                          f"Fused={fused_px[b, k]}  "
                          f"conf=({hrnet_conf[b, k]:.4f}, "
                          f"{dino_conf[b, k]:.4f})")
                sanity_count += 1

        # --- PnP per sample ---
        vis_np = hrnet_batch["visibility"].numpy()
        has_pose_np = hrnet_batch["has_pose"].numpy()
        gt_q_np = hrnet_batch["quaternion"].numpy()
        gt_t_np = hrnet_batch["translation"].numpy()

        for b in range(B):
            if not has_pose_np[b]:
                continue

            pnp_res = solve_pnp_robust(
                fused_px[b],
                vis_np[b],
                fused_conf[b],
                pnp_data["points_3d"],
                pnp_data["camera_matrix"],
                pnp_data["dist_coeffs"],
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

            all_pnp_results.append(pnp_res)
            method = pnp_res["selection_method"]
            method_counts[method] = method_counts.get(method, 0) + 1

            if pnp_res["success"]:
                errors = compute_pose_errors(
                    pnp_res["R"], pnp_res["t"], gt_q_np[b], gt_t_np[b]
                )
                all_pose_errors.append(errors)
            else:
                all_pose_errors.append(None)

    return all_pnp_results, all_pose_errors, method_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation: HRNet + DINO confidence-weighted "
                    "keypoint fusion"
    )
    parser.add_argument("--hrnet_checkpoint", type=str, required=True,
                        help="HRNet-W32 checkpoint (e.g. "
                             "outputs_hrnet_w32/converted_model.pth)")
    parser.add_argument("--dino_checkpoint", type=str, required=True,
                        help="DINO checkpoint (e.g. "
                             "outputs_self_training_dg/final_model.pth)")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="results_ensemble.txt",
                        help="Output results file")

    # PnP parameters
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence threshold for fused keypoints")
    parser.add_argument("--min_landmarks", type=int, default=8)
    parser.add_argument("--reproj_error", type=float, default=15.0)
    parser.add_argument("--min_inliers_schedule", type=str, default="",
                        help="Cascading inlier schedule (e.g. '11,9,8,6,4')")
    parser.add_argument("--refine_lm", type=int, default=1,
                        help="Enable LM refinement (default: 1)")
    parser.add_argument("--refine_retrim", type=int, default=1,
                        help="Enable LM retrimming (default: 1)")
    parser.add_argument("--refine_keep_frac", type=float, default=0.8)
    parser.add_argument("--refine_min_keep", type=int, default=6)
    parser.add_argument("--ransac_iterations", type=int, default=200)
    parser.add_argument("--ransac_confidence", type=float, default=0.99)

    # Split control
    parser.add_argument("--test_split", action="store_true",
                        help="Use test-only subset for lightbox/sunlamp")
    parser.add_argument("--splits_dir", type=str, default="data/splits")

    # Debug
    parser.add_argument("--sanity_check", type=int, default=0,
                        help="Print first N samples for coord alignment check")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit to first N samples per split (0=all)")

    args = parser.parse_args()

    # Parse cascading schedule
    schedule_str = (args.min_inliers_schedule or "").strip()
    if schedule_str:
        min_inliers_schedule = sorted(
            [int(x.strip()) for x in schedule_str.split(",") if x.strip()],
            reverse=True,
        )
    else:
        min_inliers_schedule = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load both models ----
    print("=" * 80)
    print("  ENSEMBLE EVALUATION: HRNet + DINO keypoint fusion")
    print("=" * 80)

    print("\nHRNet model:")
    hrnet_model, hrnet_cfg, hrnet_epoch = load_model_and_config(
        args.hrnet_checkpoint, device
    )
    print("\nDINO model:")
    dino_model, dino_cfg, dino_epoch = load_model_and_config(
        args.dino_checkpoint, device
    )

    # ---- PnP data (shared) ----
    geo_cfg = hrnet_cfg.get("geometry", {})
    pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
    print(f"\n3D points: {pnp_data['points_3d'].shape}")

    # ---- Transforms ----
    hrnet_transform = KeypointTransform(
        image_size=hrnet_cfg["data"]["image_size"],
        is_train=False,
        imagenet_normalize=hrnet_cfg["data"].get("imagenet_normalize", True),
    )
    dino_transform = KeypointTransform(
        image_size=dino_cfg["data"]["image_size"],
        is_train=False,
        imagenet_normalize=dino_cfg["data"].get("imagenet_normalize", True),
    )

    # HRNet uses resize_first = image_size (512)
    hrnet_resize_first = hrnet_cfg["data"]["image_size"]

    print(f"\nPnP settings:")
    print(f"  Confidence threshold:    {args.confidence}")
    print(f"  Min landmarks:           {args.min_landmarks}")
    print(f"  Reproj error:            {args.reproj_error} px")
    print(f"  Cascading schedule:      {min_inliers_schedule}")
    print(f"  LM refinement:           {'ON' if args.refine_lm else 'OFF'}")
    print(f"  LM retrim:               {'ON' if args.refine_retrim else 'OFF'}")
    print(f"  RANSAC iterations:       {args.ransac_iterations}")
    print(f"  RANSAC confidence:       {args.ransac_confidence}")

    settings = {
        "confidence": args.confidence,
        "min_landmarks": args.min_landmarks,
        "reproj_error": args.reproj_error,
    }

    # ---- Evaluate each split ----
    all_results = {}
    for split in args.splits:
        if split not in hrnet_cfg["data"]["splits"]:
            print(f"\nSkipping {split} (not in HRNet config)")
            continue
        if split not in dino_cfg["data"]["splits"]:
            print(f"\nSkipping {split} (not in DINO config)")
            continue

        # Create separate datasets for each model's preprocessing
        hrnet_ds = make_dataset(
            hrnet_cfg, split, hrnet_transform,
            gt_crop=True, resize_first=hrnet_resize_first,
            test_split=args.test_split, splits_dir=args.splits_dir,
        )
        dino_ds = make_dataset(
            dino_cfg, split, dino_transform,
            gt_crop=False, resize_first=0,
            test_split=args.test_split, splits_dir=args.splits_dir,
        )

        assert len(hrnet_ds) == len(dino_ds), (
            f"Dataset size mismatch for {split}: "
            f"HRNet={len(hrnet_ds)}, DINO={len(dino_ds)}"
        )

        # Optionally limit to first N samples
        if args.max_samples > 0:
            n = min(args.max_samples, len(hrnet_ds))
            hrnet_ds = Subset(hrnet_ds, list(range(n)))
            dino_ds = Subset(dino_ds, list(range(n)))

        hrnet_loader = DataLoader(
            hrnet_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        dino_loader = DataLoader(
            dino_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        print(f"\n{'':=<80}")
        print(f"  Evaluating ensemble: {split} ({len(hrnet_ds)} samples)")
        print(f"  HRNet: gt_crop + resize_first={hrnet_resize_first} + argmax")
        print(f"  DINO:  YOLO bbox crop + soft-argmax")
        print(f"{'':=<80}")

        pnp_results, pose_errors, method_counts = evaluate_ensemble_split(
            hrnet_model, dino_model,
            hrnet_loader, dino_loader,
            device, pnp_data,
            hrnet_resize_first=hrnet_resize_first,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            min_inliers_schedule=min_inliers_schedule,
            refine_lm=bool(args.refine_lm),
            refine_retrim=bool(args.refine_retrim),
            refine_keep_frac=args.refine_keep_frac,
            refine_min_keep=args.refine_min_keep,
            ransac_iterations=args.ransac_iterations,
            ransac_confidence=args.ransac_confidence,
            sanity_check=args.sanity_check,
        )

        n_total = len(pnp_results)
        n_solved = sum(1 for r in pnp_results if r["success"])
        n_dropped = n_total - n_solved

        # Selection breakdown
        print_selection_report(
            method_counts, pnp_results, args.confidence,
            args.min_landmarks, n_total,
        )

        # Pose metrics (solved samples only)
        solved_errors = [e for e in pose_errors if e is not None]
        if solved_errors:
            mean_slab = np.mean([e["slab"] for e in solved_errors])
            mean_ori = np.mean([e["orient_score"] for e in solved_errors])
            mean_pos = np.mean([e["pos_score"] for e in solved_errors])
            mean_rot = np.mean([e["rot_deg"] for e in solved_errors])
            mean_t = np.mean([e["pos_abs"] for e in solved_errors])
        else:
            mean_slab = mean_ori = mean_pos = mean_rot = mean_t = 0.0

        # Inlier stats
        inliers = [r["n_inliers"] for r in pnp_results if r["success"]]
        if inliers:
            print(f"  Inlier stats: min={min(inliers)}, "
                  f"median={np.median(inliers):.0f}, "
                  f"mean={np.mean(inliers):.1f}, max={max(inliers)}")

        all_results[split] = {
            "loss": 0.0,  # N/A for ensemble (no single-model loss)
            "px_err": 0.0,
            "px_rmse": 0.0,
            "pck": 0.0,
            "epnp_slab": mean_slab,
            "epnp_ori": mean_ori,
            "epnp_pos": mean_pos,
            "epnp_rot": mean_rot,
            "epnp_t": mean_t,
            "solved%": n_solved / max(n_total, 1),
            "dropped": n_dropped,
        }

    # ---- Print and save ----
    mode_str = "ensemble (HRNet+DINO)"
    epoch_str = f"H:{hrnet_epoch}/D:{dino_epoch}"
    print_results_table(all_results, mode_str, epoch_str, settings)
    save_results(all_results, args.output, mode_str, epoch_str, settings)


if __name__ == "__main__":
    main()
