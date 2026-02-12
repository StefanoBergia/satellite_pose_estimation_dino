"""Prepare FDA style/test splits and background classification labels.

One-time preprocessing script that:
1. Splits lightbox and sunlamp images into style pool (for FDA augmentation)
   and test pool (for evaluation) — deterministic and reproducible.
2. Classifies all synthetic training images and style-pool images by
   background type (earth vs black).
3. Writes split lists and classification labels to data/splits/.

The shared dataset folder is never modified — only filename lists are saved
in the project's own data/splits/ directory.

Usage:
    python scripts/prepare_fda_splits.py
    python scripts/prepare_fda_splits.py --style_fraction 0.3 --seed 123
    python scripts/prepare_fda_splits.py --config config.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path so we can import src.fda
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fda import classify_background


def list_images(directory: Path) -> list[str]:
    """List all image filenames in a directory, sorted."""
    return sorted(
        p.name for p in directory.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def split_filenames(filenames: list[str], style_fraction: float,
                    seed: int) -> tuple[list[str], list[str]]:
    """Deterministic split into style and test sets."""
    rng = random.Random(seed)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)
    n_style = int(len(shuffled) * style_fraction)
    return sorted(shuffled[:n_style]), sorted(shuffled[n_style:])


def classify_images(image_dir: Path, filenames: list[str],
                    var_threshold: float, label: str = "") -> dict[str, str]:
    """Classify a list of images by background type."""
    bg_labels = {}
    n = len(filenames)
    for i, fname in enumerate(filenames):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"    {label} classifying: {i + 1}/{n}...")
        img = Image.open(image_dir / fname).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        bg_labels[fname] = classify_background(arr, var_threshold=var_threshold)
    return bg_labels


def write_filelist(path: Path, filenames: list[str]):
    """Write a list of filenames, one per line."""
    with open(path, "w") as f:
        for fname in filenames:
            f.write(fname + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare FDA splits and bg labels")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file to read dataset paths from")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--style_fraction", type=float, default=0.2,
                        help="Fraction of real images for style pool (rest for test)")
    parser.add_argument("--var_threshold", type=float, default=0.003,
                        help="Background classification variance threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--skip_train_bg", action="store_true",
                        help="Skip classifying training images (if already done)")
    args = parser.parse_args()

    # Load config for dataset paths
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    root = Path(config["data"]["root"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {root}")
    print(f"Output dir:   {output_dir}")
    print(f"Style fraction: {args.style_fraction}")
    print(f"Seed: {args.seed}")
    print(f"Var threshold: {args.var_threshold}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Split lightbox and sunlamp
    # -----------------------------------------------------------------------
    for domain in ["lightbox", "sunlamp"]:
        split_cfg = config["data"]["splits"].get(domain)
        if split_cfg is None:
            print(f"  {domain}: not in config, skipping")
            continue

        image_dir = root / split_cfg["images"]
        filenames = list_images(image_dir)
        print(f"{domain}: {len(filenames)} total images")

        style_list, test_list = split_filenames(
            filenames, args.style_fraction, args.seed
        )
        print(f"  style pool: {len(style_list)}")
        print(f"  test pool:  {len(test_list)}")

        write_filelist(output_dir / f"{domain}_style.txt", style_list)
        write_filelist(output_dir / f"{domain}_test.txt", test_list)

        # Classify style pool images
        print(f"  Classifying style pool backgrounds...")
        bg_labels = classify_images(
            image_dir, style_list, args.var_threshold, label=domain
        )
        n_earth = sum(1 for v in bg_labels.values() if v == "earth")
        n_black = sum(1 for v in bg_labels.values() if v == "black")
        print(f"  bg classification: {n_black} black, {n_earth} earth")

        with open(output_dir / f"{domain}_bg_labels.json", "w") as f:
            json.dump(bg_labels, f)

        print()

    # -----------------------------------------------------------------------
    # Step 2: Classify all synthetic training images
    # -----------------------------------------------------------------------
    if not args.skip_train_bg:
        train_cfg = config["data"]["splits"]["train"]
        train_image_dir = root / train_cfg["images"]
        train_filenames = list_images(train_image_dir)
        print(f"train: {len(train_filenames)} total images")
        print("  Classifying backgrounds (this may take a few minutes)...")

        train_bg_labels = classify_images(
            train_image_dir, train_filenames, args.var_threshold, label="train"
        )
        n_earth = sum(1 for v in train_bg_labels.values() if v == "earth")
        n_black = sum(1 for v in train_bg_labels.values() if v == "black")
        print(f"  bg classification: {n_black} black, {n_earth} earth")

        with open(output_dir / "train_bg_labels.json", "w") as f:
            json.dump(train_bg_labels, f)
    else:
        print("train: skipping bg classification (--skip_train_bg)")

    print()
    print("Done! Files written to:", output_dir)
    print()
    print("Generated files:")
    for p in sorted(output_dir.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:30s}  {size:>10,} bytes")


if __name__ == "__main__":
    main()
