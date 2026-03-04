"""Self-Training with Pseudo-Labels for domain adaptation.

Iterative approach:
  1. Load pretrained model
  2. Generate pseudo-labels on style-split real images (unlabeled)
  3. Fine-tune on synthetic (GT) + real (pseudo-labels) mixed data
  4. Repeat from step 2 with updated model

Usage:
    python -m domain_adaptation.option2_self_training.self_train \
        --config domain_adaptation/option2_self_training/config_self_train.yaml

    python -m domain_adaptation.option2_self_training.self_train \
        --config domain_adaptation/option2_self_training/config_self_train.yaml \
        --pretrained outputs_keypoints_heatmap_FDA/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.trainer import Trainer
from src.utils import load_pnp_data
from train import build_dataset, maybe_subset
from evaluate_robust import (
    evaluate_split,
    print_selection_report,
)

from domain_adaptation.option2_self_training.pseudo_label_dataset import PseudoLabelDataset


class AddPseudoKeys(Dataset):
    """Wraps a dataset to add pseudo_weight and confidence keys for collate compatibility."""
    def __init__(self, dataset, num_keypoints):
        self.dataset = dataset
        self.num_keypoints = num_keypoints
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["pseudo_weight"] = torch.tensor(1.0, dtype=torch.float32)
        sample["confidence"] = torch.ones(self.num_keypoints, dtype=torch.float32)
        return sample


def load_split_list(splits_dir, domain, split_type):
    """Load filename list from split file."""
    path = Path(splits_dir) / f"{domain}_{split_type}.txt"
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def build_model(config):
    """Build SatellitePoseModel from config."""
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})
    mode = config["model"]["mode"]
    return SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=config["model"]["freeze_backbone"],
        unfreeze_last_n_blocks=config["model"].get("unfreeze_last_n_blocks", 0),
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


@torch.no_grad()
def generate_pseudo_labels(model, dataset, device, batch_size=32, num_workers=2):
    """Generate pseudo-labels by running inference on unlabeled images.

    Returns:
        dict {filename: {"keypoints": (K,2), "confidence": (K,)}}
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    pseudo_labels = {}
    idx = 0
    for batch in tqdm(loader, desc="Generating pseudo-labels"):
        images = batch["image"].to(device)
        model_out = model(pixel_values=images)

        kp = model_out["keypoints"].cpu().numpy()
        if "heatmaps" in model_out:
            hm = model_out["heatmaps"].cpu().numpy()
            confidence = hm.max(axis=(2, 3))
        else:
            confidence = np.ones((kp.shape[0], kp.shape[1]))

        B = images.shape[0]
        for i in range(B):
            # Get filename from dataset
            if hasattr(dataset, 'samples'):
                img_path = dataset.samples[idx][0]
                fname = Path(img_path).name
            else:
                fname = f"sample_{idx}.jpg"

            pseudo_labels[fname] = {
                "keypoints": kp[i],          # (K, 2) crop-relative [0,1]
                "confidence": confidence[i],  # (K,)
            }
            idx += 1

    return pseudo_labels


@torch.no_grad()
def evaluate_on_test(model, config, domain, test_list, device, pnp_data,
                     batch_size=32, min_landmarks=8, reproj_error=15.0,
                     confidence_threshold=0.95, no_crop=False, gt_crop=False,
                     resize_first=False, crop_pnp=False,
                     ransac_iterations=200, ransac_confidence=0.99,
                     min_kpt_area=0.0, t_ratio_max=0.0,
                     kpt_extractor="softargmax", rmse_inliers_thr=0.0,
                     no_conf_filter=False, min_inliers_schedule=None,
                     refine_lm=True, refine_retrim=True,
                     refine_keep_frac=0.8, refine_min_keep=6):
    """Robust evaluation on test split using adaptive confidence + top-N fallback."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][domain]
    transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )
    mode = config["model"]["mode"]
    pose_cfg = config.get("pose", {})

    pose_json = config["data"].get("pose_labels", {}).get(domain)

    dataset = SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=test_list,
        no_crop=no_crop,
        gt_crop=gt_crop,
        resize_first=config["data"]["image_size"] if resize_first else 0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    kp_metrics, pnp_results, pose_errors, method_counts, _, _ = evaluate_split(
        model, loader, mode, device, pnp_data,
        min_landmarks=min_landmarks,
        reproj_error=reproj_error,
        confidence_threshold=confidence_threshold,
        crop_pnp=crop_pnp,
        image_size=config["data"]["image_size"],
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence,
        min_kpt_area=min_kpt_area,
        t_ratio_max=t_ratio_max,
        resize_first=resize_first,
        heatmap_size=pose_cfg.get("heatmap_size", 128),
        kpt_extractor=kpt_extractor,
        rmse_inliers_thr=rmse_inliers_thr,
        no_conf_filter=no_conf_filter,
        min_inliers_schedule=min_inliers_schedule,
        refine_lm=refine_lm,
        refine_retrim=refine_retrim,
        refine_keep_frac=refine_keep_frac,
        refine_min_keep=refine_min_keep,
    )

    n_total = len(pnp_results)
    n_solved = sum(1 for r in pnp_results if r["success"])

    solved_errors = [e for e in pose_errors if e is not None]
    result = {**kp_metrics}
    if solved_errors:
        result["epnp_slab"] = np.mean([e["slab"] for e in solved_errors])
        result["epnp_rot"] = np.mean([e["rot_deg"] for e in solved_errors])
        result["epnp_t"] = np.mean([e["pos_abs"] for e in solved_errors])
        result["solved%"] = n_solved / max(n_total, 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Self-Training with Pseudo-Labels")
    parser.add_argument("--config", type=str,
                        default="domain_adaptation/option2_self_training/config_self_train.yaml")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Override pretrained checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--subset_size", type=int, default=None)

    # Evaluation parameters (matching evaluate_robust.py)
    parser.add_argument("--no_crop", action="store_true",
                        help="Feed full images (no YOLO bbox crop)")
    parser.add_argument("--gt_crop", action="store_true",
                        help="Crop around GT keypoints instead of YOLO bbox")
    parser.add_argument("--crop_pnp", action="store_true",
                        help="Run PnP in crop-resized space with adjusted K")
    parser.add_argument("--resize_first", action="store_true",
                        help="Resize full image to image_size before cropping")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax",
                        choices=["softargmax", "argmax"],
                        help="Keypoint extraction method")
    parser.add_argument("--reproj_error", type=float, default=15.0,
                        help="RANSAC reprojection error in pixels")
    parser.add_argument("--ransac_confidence", type=float, default=0.99,
                        help="RANSAC confidence parameter")
    parser.add_argument("--ransac_iterations", type=int, default=200,
                        help="RANSAC max iterations")
    parser.add_argument("--t_ratio_max", type=float, default=0.0,
                        help="Max ||t_est||/||t_gt|| ratio to accept (0=disabled)")
    parser.add_argument("--min_kpt_area", type=float, default=0.0,
                        help="Min bbox area of visible keypoints to accept PnP (0=disabled)")
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0,
                        help="Reject PnP solutions with inlier RMSE > threshold (0=disabled)")
    parser.add_argument("--no_conf_filter", action="store_true",
                        help="Skip confidence pre-filtering; feed all visible to RANSAC")
    parser.add_argument("--min_inliers_schedule", type=str, default="",
                        help="Comma-separated descending min inlier thresholds")
    parser.add_argument("--refine_lm", type=int, default=0,
                        help="Enable LM refinement after EPnP (default: 0)")
    parser.add_argument("--refine_retrim", type=int, default=0,
                        help="Enable 2-pass LM with outlier retrimming (default: 0)")
    args = parser.parse_args()

    # Parse cascading schedule
    schedule_str = (args.min_inliers_schedule or "").strip()
    if schedule_str:
        args.min_inliers_schedule_list = sorted(
            [int(x.strip()) for x in schedule_str.split(",") if x.strip()],
            reverse=True,
        )
    else:
        args.min_inliers_schedule_list = None

    # Load configs
    with open(args.config) as f:
        st_config = yaml.safe_load(f)

    base_config_path = st_config.get("base_config", "config.yaml")
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Override training params
    if "train" in st_config:
        config["train"].update(st_config["train"])

    st_cfg = st_config["self_training"]
    splits_dir = st_config.get("splits_dir", "data/splits")
    checkpoint_path = args.pretrained or st_cfg["pretrained_checkpoint"]
    confidence_threshold = st_cfg.get("confidence_threshold", 0.3)
    pseudo_weight = st_cfg.get("pseudo_label_weight", 0.5)
    num_iterations = st_cfg.get("num_iterations", 3)
    epochs_per_iter = st_cfg.get("epochs_per_iteration", 10)
    target_domains = st_cfg.get("target_domains", ["sunlamp", "lightbox"])
    synth_subset_size = st_cfg.get("synth_subset_size", None)

    # Incremental self-training config
    inc_cfg = st_cfg.get("incremental", {})
    incremental_mode = inc_cfg.get("enabled", False)
    inc_variant = inc_cfg.get("mode", "sequential")   # "sequential" or "cumulative"
    n_chunks = inc_cfg.get("n_chunks", 5)
    chunks_dir = inc_cfg.get("chunks_dir", "data/splits_incremental")
    holdout_chunk = inc_cfg.get("holdout_chunk", None)  # index to hold out from adaptation
    if incremental_mode:
        train_chunks = [i for i in range(n_chunks) if i != holdout_chunk]
        num_iterations = len(train_chunks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Self-training iterations: {num_iterations}")
    print(f"Epochs per iteration: {epochs_per_iter}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Pseudo-label weight: {pseudo_weight}")
    print(f"Target domains: {target_domains}")

    # Load PnP data for evaluation
    geo_cfg = config.get("geometry", {})
    pnp_data = None
    if geo_cfg.get("points_3d") and geo_cfg.get("camera"):
        pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])

    # Load style/test splits
    style_lists = {}
    test_lists = {}
    chunk_lists = {}
    for domain in target_domains:
        test_lists[domain] = load_split_list(splits_dir, domain, "test")
        if incremental_mode:
            chunk_lists[domain] = [
                load_split_list(chunks_dir, domain, f"chunk{i}")
                for i in range(n_chunks)
            ]
            holdout_info = f", chunk {holdout_chunk} held out" if holdout_chunk is not None else ""
            print(f"  {domain}: {len(train_chunks)}/{n_chunks} chunks for adaptation "
                  f"(~{len(chunk_lists[domain][0])} each{holdout_info}), "
                  f"{len(test_lists[domain])} test  [{inc_variant}]")
        else:
            style_lists[domain] = load_split_list(splits_dir, domain, "style")
            print(f"  {domain}: {len(style_lists[domain])} style, {len(test_lists[domain])} test")

    # Build model and load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    # Build synthetic training dataset
    train_dataset = build_dataset(config, "train", is_train=True, mode=mode)
    print(f"Synthetic training set: {len(train_dataset)} samples")

    # Evaluation transform (no augmentation)
    eval_transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )
    # Training transform (with augmentation) for pseudo-labeled data
    train_transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=True,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
        color_jitter=config.get("augmentation", {}).get("color_jitter", 0.3),
    )

    root = Path(config["data"]["root"])
    output_dir = Path(args.output_dir or config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Self-Training Loop ===
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Self-Training Iteration {iteration}/{num_iterations}")
        print(f"{'='*60}")

        # Step 1: Generate pseudo-labels on style-split real images
        all_pseudo_datasets = []
        for domain in target_domains:
            print(f"\n  Generating pseudo-labels for {domain} style split...")

            # Determine which images to use for this iteration
            if incremental_mode:
                if inc_variant == "cumulative":
                    chunks_to_use = train_chunks[:iteration]
                else:
                    chunks_to_use = [train_chunks[iteration - 1]]
                current_style: set = set()
                for ci in chunks_to_use:
                    current_style.update(chunk_lists[domain][ci])
                print(f"    [{inc_variant} chunks {chunks_to_use}] {len(current_style)} images")
            else:
                current_style = style_lists[domain]

            # Build dataset for inference (style split only, no augmentation)
            split_cfg = config["data"]["splits"][domain]
            style_dataset = SpeedPlusKeypointDataset(
                image_dir=str(root / split_cfg["images"]),
                label_dir=str(root / split_cfg["labels"]),
                num_keypoints=config["data"]["num_keypoints"],
                bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
                transform=eval_transform,
                include_list=current_style,
                no_crop=args.no_crop,
                gt_crop=args.gt_crop,
                resize_first=config["data"]["image_size"] if args.resize_first else 0,
            )

            pseudo_labels = generate_pseudo_labels(
                model, style_dataset, device,
                batch_size=config["train"]["batch_size"],
                num_workers=config["train"]["num_workers"],
            )

            # Count accepted
            accepted = sum(
                1 for pl in pseudo_labels.values()
                if np.mean(pl["confidence"]) >= confidence_threshold
            )
            print(f"    Generated {len(pseudo_labels)} pseudo-labels, "
                  f"{accepted} above threshold ({confidence_threshold})")

            # Build pseudo-label dataset (with training augmentation)
            pseudo_ds = PseudoLabelDataset(
                image_dir=str(root / split_cfg["images"]),
                label_dir=str(root / split_cfg["labels"]),
                pseudo_labels=pseudo_labels,
                confidence_threshold=confidence_threshold,
                num_keypoints=config["data"]["num_keypoints"],
                bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
                transform=train_transform,
            )
            all_pseudo_datasets.append(pseudo_ds)

        # Step 2: Build mixed training set
        if all_pseudo_datasets:
            pseudo_combined = ConcatDataset(all_pseudo_datasets)
            # Optionally downsample synthetic set — resample each iteration for coverage
            synth_ds = train_dataset
            if synth_subset_size and synth_subset_size < len(train_dataset):
                indices = torch.randperm(len(train_dataset))[:synth_subset_size].tolist()
                synth_ds = torch.utils.data.Subset(train_dataset, indices)
            wrapped_train = AddPseudoKeys(synth_ds, config["data"]["num_keypoints"])
            mixed_dataset = ConcatDataset([wrapped_train, pseudo_combined])
            print(f"\n  Mixed dataset: {len(synth_ds)} synthetic + "
                  f"{len(pseudo_combined)} pseudo = {len(mixed_dataset)} total")
        else:
            mixed_dataset = train_dataset
            print(f"\n  No pseudo-labels accepted, training on synthetic only")

        # Step 3: Fine-tune
        config["train"]["epochs"] = epochs_per_iter
        config["train"]["output_dir"] = str(output_dir / f"iter{iteration}")

        batch_size = config["train"]["batch_size"]
        train_loader = DataLoader(
            mixed_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config["train"]["num_workers"],
            pin_memory=True, drop_last=True,
        )

        # Build eval loaders
        eval_loaders = {}
        for domain in target_domains:
            test_ds = SpeedPlusKeypointDataset(
                image_dir=str(root / config["data"]["splits"][domain]["images"]),
                label_dir=str(root / config["data"]["splits"][domain]["labels"]),
                num_keypoints=config["data"]["num_keypoints"],
                bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
                transform=eval_transform,
                pose_json=config["data"].get("pose_labels", {}).get(domain),
                include_list=test_lists[domain],
                no_crop=args.no_crop,
                gt_crop=args.gt_crop,
                resize_first=config["data"]["image_size"] if args.resize_first else 0,
            )
            eval_loaders[domain] = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=config["train"]["num_workers"], pin_memory=True,
            )

        # Add val split
        val_ds = build_dataset(config, "val", is_train=False, mode=mode)
        eval_loaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=config["train"]["num_workers"], pin_memory=True,
        )

        config["train"]["save_best_model"] = False
        config.setdefault("pose", {})["lambda_msssim"] = 0.0  # no MS-SSIM in self-training
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            eval_loaders=eval_loaders,
            config=config,
            device=device,
        )
        trainer.train()

        # Step 4: Evaluate on test splits
        print(f"\n  Evaluation after iteration {iteration}:")
        for domain in target_domains:
            metrics = evaluate_on_test(
                model, config, domain, test_lists[domain], device, pnp_data,
                no_crop=args.no_crop, gt_crop=args.gt_crop,
                resize_first=args.resize_first, crop_pnp=args.crop_pnp,
                ransac_iterations=args.ransac_iterations,
                ransac_confidence=args.ransac_confidence,
                min_kpt_area=args.min_kpt_area,
                t_ratio_max=args.t_ratio_max,
                kpt_extractor=args.kpt_extractor,
                rmse_inliers_thr=args.rmse_inliers_thr,
                no_conf_filter=args.no_conf_filter,
                min_inliers_schedule=args.min_inliers_schedule_list,
                refine_lm=bool(args.refine_lm),
                refine_retrim=bool(args.refine_retrim),
                reproj_error=args.reproj_error,
            )
            print(f"    {domain}: px_err={metrics.get('px_err', 0):.2f}  "
                  f"pck={metrics.get('pck', 0):.4f}  "
                  f"SLAB={metrics.get('epnp_slab', 0):.4f}")

    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "self_training_config": st_config,
    }, final_path)
    print(f"\nFinal model saved to {final_path}")
    print("Done!")


if __name__ == "__main__":
    main()
