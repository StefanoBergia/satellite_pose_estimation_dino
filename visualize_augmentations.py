"""Visualize augmentations applied to dataset samples.

Usage:
    python visualize_augmentations.py --config config.yaml --num_samples 5 --num_augmentations 6 --out_dir outputs/aug_viz
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform


def denormalize_tensor(tensor):
    """Undo ImageNet normalization and convert to PIL Image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def draw_keypoints(image, keypoints, visibility, radius=3):
    """Draw keypoints on an image. Returns a copy."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(len(keypoints)):
        if visibility[i] == 0:
            continue
        x, y = keypoints[i, 0] * w, keypoints[i, 1] * h
        color = "lime" if visibility[i] == 2 else "orange"
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline="white")
        draw.text((x + radius + 2, y - radius), str(i), fill="white")
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize augmentations")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of different images to show")
    parser.add_argument("--num_augmentations", type=int, default=6,
                        help="Number of augmented versions per image")
    parser.add_argument("--out_dir", type=str, default="outputs/aug_viz")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = config.get("augmentation", {})

    # Dataset without transforms (raw crops)
    root = Path(config["data"]["root"])
    split_cfg = config["data"]["splits"][args.split]
    raw_dataset = SpeedPlusKeypointDataset(
        image_dir=str(root / split_cfg["images"]),
        label_dir=str(root / split_cfg["labels"]),
        num_keypoints=config["data"]["num_keypoints"],
        bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
        transform=None,
    )

    # Training transform (photometric only)
    train_transform = KeypointTransform(
        image_size=config["data"]["image_size"],
        is_train=True,
        color_jitter=aug_cfg.get("color_jitter", 0.3),
        gaussian_blur_prob=aug_cfg.get("gaussian_blur_prob", 0.0),
        gaussian_blur_kernel=tuple(aug_cfg.get("gaussian_blur_kernel", [3, 7])),
        gaussian_noise_prob=aug_cfg.get("gaussian_noise_prob", 0.0),
        gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.02),
        random_erasing_prob=aug_cfg.get("random_erasing_prob", 0.0),
        sun_flare_prob=aug_cfg.get("sun_flare_prob", 0.0),
        sun_flare_intensity=tuple(aug_cfg.get("sun_flare_intensity", [0.3, 1.0])),
        brightness_gradient_prob=aug_cfg.get("brightness_gradient_prob", 0.0),
        brightness_gradient_strength=tuple(aug_cfg.get("brightness_gradient_strength", [0.2, 0.6])),
        random_gamma_prob=aug_cfg.get("random_gamma_prob", 0.0),
        random_gamma_range=tuple(aug_cfg.get("random_gamma_range", [0.5, 2.0])),
        motion_blur_prob=aug_cfg.get("motion_blur_prob", 0.0),
        motion_blur_kernel=tuple(aug_cfg.get("motion_blur_kernel", [5, 15])),
        clahe_prob=aug_cfg.get("clahe_prob", 0.0),
        clahe_clip_limit=tuple(aug_cfg.get("clahe_clip_limit", [1.0, 4.0])),
        channel_dropout_prob=aug_cfg.get("channel_dropout_prob", 0.0),
    )

    # Pick random sample indices
    indices = np.random.choice(len(raw_dataset), size=min(args.num_samples, len(raw_dataset)), replace=False)

    for sample_idx in indices:
        sample = raw_dataset[sample_idx]
        raw_crop = sample["image"]  # PIL Image (no transform applied)
        kp = sample["keypoints"].numpy()
        vis = sample["visibility"].numpy()

        # Original (resized to match augmented size for comparison)
        orig_resized = raw_crop.resize(
            (config["data"]["image_size"], config["data"]["image_size"]),
            Image.BILINEAR,
        )
        orig_with_kp = draw_keypoints(orig_resized, kp, vis)

        # Generate augmented versions
        aug_images = [orig_with_kp]
        for _ in range(args.num_augmentations):
            aug_tensor, aug_kp, aug_vis = train_transform(raw_crop, kp, vis)
            aug_pil = denormalize_tensor(aug_tensor)
            aug_with_kp = draw_keypoints(aug_pil, aug_kp, aug_vis)
            aug_images.append(aug_with_kp)

        # Stitch into a single row
        img_size = config["data"]["image_size"]
        n_cols = len(aug_images)
        canvas = Image.new("RGB", (img_size * n_cols, img_size), color=(30, 30, 30))
        for i, img in enumerate(aug_images):
            canvas.paste(img, (i * img_size, 0))

        # Add labels
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), "Original", fill="white")
        for i in range(1, n_cols):
            draw.text((i * img_size + 4, 2), f"Aug {i}", fill="white")

        out_path = out_dir / f"sample_{sample_idx:05d}.png"
        canvas.save(out_path)
        print(f"Saved {out_path}")

    print(f"\nDone. {len(indices)} images saved to {out_dir}/")
    print(f"Augmentations (photometric only): color_jitter={aug_cfg.get('color_jitter', 0.3)}, "
          f"blur_p={aug_cfg.get('gaussian_blur_prob', 0)}, "
          f"noise_p={aug_cfg.get('gaussian_noise_prob', 0)}, "
          f"erase_p={aug_cfg.get('random_erasing_prob', 0)}")


if __name__ == "__main__":
    main()
