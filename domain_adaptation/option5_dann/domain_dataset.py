"""Balanced 3-class domain dataset for DANN adversarial training.

Provides equal-sized samples from synthetic, lightbox, and sunlamp domains.
All three classes are undersampled to match the smallest class (sunlamp, ~558 images).
Labels are domain indices (not keypoint labels).

Classes:
    0: synthetic (from train split)
    1: lightbox  (from lightbox_style.txt)
    2: sunlamp   (from sunlamp_style.txt)
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.transforms import KeypointTransform


class DomainDataset(Dataset):
    """Balanced domain dataset for DANN domain classification branch.

    Images are pre-processed (resize + normalize) identically to the task
    branch so features are directly comparable.

    Args:
        config: base config dict (data.root, image_size, imagenet_normalize, etc.)
        style_lists_dir: path to directory containing lightbox_style.txt and
                         sunlamp_style.txt (default: "data/splits")
        seed: random seed for reproducible undersampling
    """

    def __init__(
        self,
        config: dict,
        style_lists_dir: str = "data/splits",
        seed: int = 42,
    ):
        root = Path(config["data"]["root"])
        splits = config["data"]["splits"]
        image_size = config["data"]["image_size"]
        imagenet_normalize = config["data"].get("imagenet_normalize", True)

        # Transform: resize + normalize only (no augmentation, no keypoints)
        self.transform = KeypointTransform(
            image_size=image_size,
            is_train=False,
            imagenet_normalize=imagenet_normalize,
        )

        style_dir = Path(style_lists_dir)

        # --- Lightbox (class 1) ---
        lightbox_image_dir = root / splits["lightbox"]["images"]
        with open(style_dir / "lightbox_style.txt") as f:
            lightbox_files = [
                lightbox_image_dir / line.strip()
                for line in f if line.strip()
            ]

        # --- Sunlamp (class 2) — reference (smallest) class ---
        sunlamp_image_dir = root / splits["sunlamp"]["images"]
        with open(style_dir / "sunlamp_style.txt") as f:
            sunlamp_files = [
                sunlamp_image_dir / line.strip()
                for line in f if line.strip()
            ]

        n_per_class = len(sunlamp_files)

        # --- Synthetic (class 0) — random subsample from train images ---
        synthetic_image_dir = root / splits["train"]["images"]
        all_synthetic = sorted([
            p for p in synthetic_image_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        rng = random.Random(seed)
        synthetic_sampled = rng.sample(all_synthetic, min(n_per_class, len(all_synthetic)))
        lightbox_sampled = rng.sample(lightbox_files, min(n_per_class, len(lightbox_files)))
        sunlamp_sampled = list(sunlamp_files)  # already reference size

        # Build flat sample list: (path, domain_label)
        self.samples = (
            [(p, 0) for p in synthetic_sampled] +
            [(p, 1) for p in lightbox_sampled] +
            [(p, 2) for p in sunlamp_sampled]
        )
        rng.shuffle(self.samples)

        print(
            f"DomainDataset: {len(synthetic_sampled)} synthetic + "
            f"{len(lightbox_sampled)} lightbox + {len(sunlamp_sampled)} sunlamp "
            f"= {len(self.samples)} total (n_per_class={n_per_class})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, domain_label = self.samples[idx]

        image = Image.open(img_path)
        if image.mode == "L":
            image = image.convert("RGB")

        # Pass dummy keypoints/visibility — only the image tensor is used
        dummy_kp = np.zeros((11, 2), dtype=np.float32)
        dummy_vis = np.zeros(11, dtype=np.int64)
        image_tensor, _, _ = self.transform(image, dummy_kp, dummy_vis)

        return {
            "image": image_tensor,
            "domain_label": torch.tensor(domain_label, dtype=torch.long),
        }
