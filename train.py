import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import SpeedPlusKeypointDataset
from src.fda import FDAStylePool
from src.transforms import KeypointTransform
from src.model import SatellitePoseModel
from src.trainer import Trainer


def build_dataset(
    config: dict, split: str, is_train: bool, mode: str
) -> SpeedPlusKeypointDataset:
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][split]
    image_dir = root / split_cfg["images"]
    label_dir = root / split_cfg["labels"]

    aug_cfg = config.get("augmentation", {})
    transform = KeypointTransform(
        image_size=config["data"]["image_size"],
        is_train=is_train,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
        color_jitter=aug_cfg.get("color_jitter", 0.3) if is_train else 0.0,
        gaussian_blur_prob=aug_cfg.get("gaussian_blur_prob", 0.0) if is_train else 0.0,
        gaussian_blur_kernel=tuple(aug_cfg.get("gaussian_blur_kernel", [3, 7])),
        gaussian_noise_prob=aug_cfg.get("gaussian_noise_prob", 0.0) if is_train else 0.0,
        gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.02),
        random_erasing_prob=aug_cfg.get("random_erasing_prob", 0.0) if is_train else 0.0,
        sun_flare_prob=aug_cfg.get("sun_flare_prob", 0.0) if is_train else 0.0,
        sun_flare_intensity=tuple(aug_cfg.get("sun_flare_intensity", [0.3, 1.0])),
        brightness_gradient_prob=aug_cfg.get("brightness_gradient_prob", 0.0) if is_train else 0.0,
        brightness_gradient_strength=tuple(aug_cfg.get("brightness_gradient_strength", [0.2, 0.6])),
        random_gamma_prob=aug_cfg.get("random_gamma_prob", 0.0) if is_train else 0.0,
        random_gamma_range=tuple(aug_cfg.get("random_gamma_range", [0.5, 2.0])),
        motion_blur_prob=aug_cfg.get("motion_blur_prob", 0.0) if is_train else 0.0,
        motion_blur_kernel=tuple(aug_cfg.get("motion_blur_kernel", [5, 15])),
        clahe_prob=aug_cfg.get("clahe_prob", 0.0) if is_train else 0.0,
        clahe_clip_limit=tuple(aug_cfg.get("clahe_clip_limit", [1.0, 4.0])),
        channel_dropout_prob=aug_cfg.get("channel_dropout_prob", 0.0) if is_train else 0.0,
    )

    # Pose JSON (only load for pose modes)
    pose_json = None
    if mode != "keypoint_only":
        pose_labels = config["data"].get("pose_labels", {})
        if split in pose_labels:
            pose_json = pose_labels[split]

    # FDA augmentation (training split only)
    fda_pool = None
    fda_prob = 0.0
    bg_labels = {}
    if is_train and split == "train":
        fda_prob = aug_cfg.get("fda_prob", 0.0)
        if fda_prob > 0:
            fda_cfg = config.get("fda", {})
            splits_dir = Path(fda_cfg.get("splits_dir", "data/splits"))

            # Load background labels for training images
            train_bg_path = splits_dir / "train_bg_labels.json"
            with open(train_bg_path) as f:
                bg_labels = json.load(f)

            # Build style pool from target domains
            target_domains = aug_cfg.get("fda_target_domains", ["lightbox", "sunlamp"])
            fda_image_dirs = {}
            fda_style_lists = {}
            fda_bg_labels = {}
            for domain in target_domains:
                domain_split_cfg = config["data"]["splits"][domain]
                fda_image_dirs[domain] = str(root / domain_split_cfg["images"])
                fda_style_lists[domain] = str(splits_dir / f"{domain}_style.txt")
                domain_bg_path = splits_dir / f"{domain}_bg_labels.json"
                with open(domain_bg_path) as f:
                    fda_bg_labels[domain] = json.load(f)

            fda_pool = FDAStylePool(
                image_dirs=fda_image_dirs,
                style_lists=fda_style_lists,
                bg_labels=fda_bg_labels,
                beta=aug_cfg.get("fda_beta", 0.05),
                lambd=aug_cfg.get("fda_lambda", 0.5),
            )

    # Test-split filtering for eval splits (when FDA splits exist)
    include_list = None
    if not is_train and split in ("lightbox", "sunlamp"):
        fda_cfg = config.get("fda", {})
        splits_dir = Path(fda_cfg.get("splits_dir", "data/splits"))
        test_list_path = splits_dir / f"{split}_test.txt"
        if test_list_path.exists():
            with open(test_list_path) as f:
                include_list = set(line.strip() for line in f if line.strip())

    return SpeedPlusKeypointDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=transform,
        pose_json=pose_json,
        include_list=include_list,
        fda_pool=fda_pool,
        fda_prob=fda_prob,
        bg_labels=bg_labels,
    )


def maybe_subset(dataset, subset_size: int | None):
    """Wrap dataset in a Subset if subset_size is specified."""
    if subset_size is not None and subset_size < len(dataset):
        return Subset(dataset, list(range(subset_size)))
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train keypoint/pose estimation on SPEED+")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--subset_size", type=int, default=None,
        help="Use only the first N samples per split (for quick testing)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to a pretrained checkpoint to warm-start from (loaded with strict=False)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["model"]["mode"]
    subset_size = args.subset_size or config["data"].get("subset_size")

    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    if subset_size:
        print(f"Subset mode: using first {subset_size} samples per split")

    # Build datasets
    print("Building datasets...")
    train_dataset = maybe_subset(
        build_dataset(config, "train", is_train=True, mode=mode), subset_size
    )
    print(f"  train: {len(train_dataset)} samples")

    eval_datasets = {}
    for split in ["val", "lightbox", "sunlamp"]:
        if split in config["data"]["splits"]:
            ds = maybe_subset(
                build_dataset(config, split, is_train=False, mode=mode), subset_size
            )
            eval_datasets[split] = ds
            print(f"  {split}: {len(ds)} samples")

    # Build dataloaders
    batch_size = config["train"]["batch_size"]
    if subset_size and subset_size < batch_size:
        batch_size = subset_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    eval_loaders = {}
    for split, ds in eval_datasets.items():
        eval_loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=True,
        )

    # Build model
    print("Loading model...")
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})
    model = SatellitePoseModel(
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

    # Load pretrained checkpoint (warm-start)
    if args.pretrained:
        print(f"Loading pretrained checkpoint: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")
        if not missing and not unexpected:
            print("  All keys matched.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        config=config,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
