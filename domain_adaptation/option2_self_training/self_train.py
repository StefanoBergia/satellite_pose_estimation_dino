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
from torch.utils.data import DataLoader, ConcatDataset
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

    for batch in tqdm(loader, desc="Generating pseudo-labels"):
        images = batch["image"].to(device)
        model_out = model(pixel_values=images)

        kp = model_out["keypoints"].cpu().numpy()  # (B, K, 2)

        # Extract confidence from heatmaps
        if "heatmaps" in model_out:
            hm = model_out["heatmaps"].cpu().numpy()  # (B, K, H, W)
            confidence = hm.max(axis=(2, 3))  # (B, K) peak heatmap value
        else:
            # For MLP head, use distance from center as proxy
            confidence = np.ones((kp.shape[0], kp.shape[1]))

        # Map back to filenames via dataset
        # The dataset stores (img_path, label_path) tuples
        batch_size_actual = images.shape[0]
        for i in range(batch_size_actual):
            # Get the dataset index for this batch item
            # Since we iterate sequentially, track the global index
            pass

    # Re-iterate with index tracking
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
                     confidence_threshold=0.95):
    """Robust evaluation on test split using adaptive confidence + top-N fallback."""
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][domain]
    transform = KeypointTransform(image_size=config["data"]["image_size"], is_train=False)
    mode = config["model"]["mode"]

    pose_json = config["data"].get("pose_labels", {}).get(domain)

    dataset = SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=test_list,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    kp_metrics, pnp_results, pose_errors, method_counts = evaluate_split(
        model, loader, mode, device, pnp_data,
        min_landmarks=min_landmarks,
        reproj_error=reproj_error,
        confidence_threshold=confidence_threshold,
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
    args = parser.parse_args()

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
    for domain in target_domains:
        style_lists[domain] = load_split_list(splits_dir, domain, "style")
        test_lists[domain] = load_split_list(splits_dir, domain, "test")
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
        image_size=config["data"]["image_size"], is_train=False
    )
    # Training transform (with augmentation) for pseudo-labeled data
    train_transform = KeypointTransform(
        image_size=config["data"]["image_size"], is_train=True,
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

            # Build dataset for inference (style split only, no augmentation)
            split_cfg = config["data"]["splits"][domain]
            style_dataset = SpeedPlusKeypointDataset(
                image_dir=str(root / split_cfg["images"]),
                label_dir=str(root / split_cfg["labels"]),
                num_keypoints=config["data"]["num_keypoints"],
                bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
                transform=eval_transform,
                include_list=style_lists[domain],
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
            mixed_dataset = ConcatDataset([train_dataset, pseudo_combined])
            print(f"\n  Mixed dataset: {len(train_dataset)} synthetic + "
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
