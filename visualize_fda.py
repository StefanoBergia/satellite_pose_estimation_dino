"""Visualize Fourier Domain Adaptation (FDA) on SPEED+ images.

Automatically classifies images as earth-background vs black-background and
only pairs images of the same type, preventing the spectral mismatch that
causes artifacts when mixing background types.

Compares:
  1. FDA-blend: interpolated amplitude swap (stable, one beta/lambda to set)
  2. FDA-blend with background-aware pairing (the recommended approach)

Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
Segmentation" (CVPR 2020).

Usage:
    python visualize_fda.py                          # default comparison
    python visualize_fda.py --n_samples 6 --seed 99
    python visualize_fda.py --target_domain sunlamp
    python visualize_fda.py --var_threshold 0.005     # tune bg classifier
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Background classification
# ---------------------------------------------------------------------------

def classify_background(img: np.ndarray, var_threshold: float = 0.003) -> str:
    """Classify an image as 'earth' or 'black' background.

    Strategy: split the border region into patches and check if ANY patch has
    high spatial variance. Earth textures (clouds, continents, horizon
    gradients) produce high local variance even if they only cover a corner
    or one edge of the image. Uniform backgrounds — whether pitch-black
    (synthetic/sunlamp) or ambient-bright (lightbox) — have low variance
    everywhere.

    This works across all domains regardless of absolute brightness.

    Args:
        img: (H, W) float image in [0, 1]
        var_threshold: per-patch variance threshold on [0, 1] scale.
                       Earth patches typically > 0.005, uniform < 0.002.
                       Default 0.003 is a conservative middle ground.

    Returns:
        'earth' or 'black'
    """
    h, w = img.shape
    margin_h = max(1, int(h * 0.10))
    margin_w = max(1, int(w * 0.10))

    # Extract 4 border strips
    strips = [
        img[:margin_h, :],                    # top
        img[-margin_h:, :],                   # bottom
        img[margin_h:-margin_h, :margin_w],   # left
        img[margin_h:-margin_h, -margin_w:],  # right
    ]

    # Split each strip into patches and check variance of each
    n_patches_per_strip = 4
    for strip in strips:
        sh, sw = strip.shape
        if sh == 0 or sw == 0:
            continue
        # Split along the longer axis
        if sw >= sh:
            patch_w = max(1, sw // n_patches_per_strip)
            for i in range(n_patches_per_strip):
                patch = strip[:, i * patch_w : (i + 1) * patch_w]
                if patch.size > 0 and patch.var() > var_threshold:
                    return "earth"
        else:
            patch_h = max(1, sh // n_patches_per_strip)
            for i in range(n_patches_per_strip):
                patch = strip[i * patch_h : (i + 1) * patch_h, :]
                if patch.size > 0 and patch.var() > var_threshold:
                    return "earth"

    return "black"


def load_and_classify_images(directory: Path, n: int, seed: int = 42,
                             var_threshold: float = 0.003) -> dict:
    """Load n random images, classify by background type.

    Returns:
        dict with keys 'earth' and 'black', each a list of (name, array)
    """
    all_images = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rng = random.Random(seed)
    selected = rng.sample(all_images, min(n, len(all_images)))

    classified = {"earth": [], "black": []}
    for p in selected:
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        bg = classify_background(arr, var_threshold=var_threshold)
        classified[bg].append((p.name, arr))

    return classified


# ---------------------------------------------------------------------------
# FDA core
# ---------------------------------------------------------------------------

def _fft_components(img_2d: np.ndarray):
    """Compute shifted FFT, returning amplitude and phase."""
    fft = np.fft.fftshift(np.fft.fft2(img_2d))
    return np.abs(fft), np.angle(fft)


def _lowfreq_mask(h: int, w: int, beta: float) -> np.ndarray:
    """Binary mask selecting the central (2*bh x 2*bw) rectangle."""
    mask = np.zeros((h, w), dtype=bool)
    bh = max(1, int(h * beta))
    bw = max(1, int(w * beta))
    ch, cw = h // 2, w // 2
    mask[ch - bh : ch + bh, cw - bw : cw + bw] = True
    return mask


def _reconstruct(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse FFT from amplitude + phase."""
    fft = amplitude * np.exp(1j * phase)
    return np.fft.ifft2(np.fft.ifftshift(fft)).real


def fda_blend(source: np.ndarray, target: np.ndarray, beta: float = 0.05,
              lambd: float = 0.5) -> np.ndarray:
    """FDA with blending: interpolate low-freq amplitude.

    mixed_amp = (1 - lambd) * src_amp + lambd * tgt_amp  (in low-freq region)
    """
    h, w = source.shape
    src_amp, src_phase = _fft_components(source)
    tgt_amp, _ = _fft_components(target)

    mask = _lowfreq_mask(h, w, beta)
    mixed_amp = src_amp.copy()
    mixed_amp[mask] = (1 - lambd) * src_amp[mask] + lambd * tgt_amp[mask]

    return np.clip(_reconstruct(mixed_amp, src_phase), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_to_match(target: np.ndarray, source_shape: tuple) -> np.ndarray:
    """Resize target image to match source dimensions."""
    if target.shape == source_shape:
        return target
    h, w = source_shape[:2]
    return np.array(
        Image.fromarray((target * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize FDA with background-aware pairing")
    parser.add_argument(
        "--data_root", type=str,
        default="/nfs/home/caracciolo/dataset/speedplus_yolo/images",
    )
    parser.add_argument(
        "--target_domain", type=str, choices=["sunlamp", "lightbox", "both"],
        default="both",
    )
    parser.add_argument("--n_sources", type=int, default=30,
                        help="Number of synthetic images to load (before filtering)")
    parser.add_argument("--n_targets", type=int, default=30,
                        help="Number of target images to load (before filtering)")
    parser.add_argument("--n_show", type=int, default=4,
                        help="Number of rows per background type to show")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lambd", type=float, default=0.5)
    parser.add_argument("--var_threshold", type=float, default=0.003,
                        help="Border patch variance threshold for earth vs black (0-1 scale)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Load and classify
    print("Loading and classifying synthetic images...")
    src_by_bg = load_and_classify_images(
        data_root / "train", args.n_sources, seed=args.seed,
        var_threshold=args.var_threshold,
    )
    print(f"  Synthetic — black: {len(src_by_bg['black'])}, earth: {len(src_by_bg['earth'])}")

    domains = []
    if args.target_domain in ("sunlamp", "both"):
        domains.append("sunlamp")
    if args.target_domain in ("lightbox", "both"):
        domains.append("lightbox")

    tgt_by_domain = {}
    for domain in domains:
        print(f"Loading and classifying {domain} images...")
        tgt_by_bg = load_and_classify_images(
            data_root / domain, args.n_targets, seed=args.seed + 1,
            var_threshold=args.var_threshold,
        )
        tgt_by_domain[domain] = tgt_by_bg
        print(f"  {domain} — black: {len(tgt_by_bg['black'])}, earth: {len(tgt_by_bg['earth'])}")

    rng = random.Random(args.seed)

    # -----------------------------------------------------------------------
    # Figure 1 (per domain): Background-aware FDA
    #   Two sections: black-bg pairs, then earth-bg pairs
    #   Columns: source | target | naive (random pair) | matched (same bg type)
    # -----------------------------------------------------------------------
    for domain in domains:
        tgt_by_bg = tgt_by_domain[domain]

        # Collect all targets in a flat list for "naive" random pairing
        all_targets = tgt_by_bg["black"] + tgt_by_bg["earth"]

        rows = []  # (src_name, src_img, src_bg, tgt_matched_name, tgt_matched_img,
                    #  tgt_naive_name, tgt_naive_img)

        for bg_type in ["black", "earth"]:
            src_list = src_by_bg[bg_type]
            tgt_list = tgt_by_bg[bg_type]
            if not src_list or not tgt_list:
                continue

            for i in range(min(args.n_show, len(src_list))):
                src_name, src_img = src_list[i]
                # Matched target (same bg type)
                tgt_matched_name, tgt_matched_img = rng.choice(tgt_list)
                # Naive target (opposite bg type if available, to show the problem)
                opposite = "earth" if bg_type == "black" else "black"
                if tgt_by_bg[opposite]:
                    tgt_naive_name, tgt_naive_img = rng.choice(tgt_by_bg[opposite])
                else:
                    tgt_naive_name, tgt_naive_img = rng.choice(all_targets)

                rows.append((src_name, src_img, bg_type,
                             tgt_matched_name, tgt_matched_img,
                             tgt_naive_name, tgt_naive_img))

        if not rows:
            print(f"  No valid pairs for {domain}, skipping.")
            continue

        n_rows = len(rows)
        # Columns: source | matched target | FDA matched | mismatched target | FDA mismatched
        n_cols = 5
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows), squeeze=False,
        )
        fig.suptitle(
            f"Background-aware FDA: synthetic → {domain}\n"
            f"β={args.beta}, λ={args.lambd}  |  var_threshold={args.var_threshold}",
            fontsize=13, fontweight="bold",
        )

        col_headers = [
            "Source (synthetic)",
            "Matched target\n(same bg type)",
            "FDA (matched)",
            "Mismatched target\n(opposite bg type)",
            "FDA (mismatched)",
        ]

        for row_idx, (src_name, src_img, bg_type,
                      tgt_m_name, tgt_m_img,
                      tgt_n_name, tgt_n_img) in enumerate(rows):

            tgt_m_resized = resize_to_match(tgt_m_img, src_img.shape)
            tgt_n_resized = resize_to_match(tgt_n_img, src_img.shape)

            fda_matched = fda_blend(src_img, tgt_m_resized, args.beta, args.lambd)
            fda_mismatched = fda_blend(src_img, tgt_n_resized, args.beta, args.lambd)

            images = [src_img, tgt_m_img, fda_matched, tgt_n_img, fda_mismatched]
            labels = [
                f"{src_name}\n[{bg_type}]",
                f"{tgt_m_name}\n[{classify_background(tgt_m_img, args.var_threshold)}]",
                "OK",
                f"{tgt_n_name}\n[{classify_background(tgt_n_img, args.var_threshold)}]",
                "MISMATCH",
            ]

            for col_idx in range(n_cols):
                axes[row_idx, col_idx].imshow(images[col_idx], cmap="gray", vmin=0, vmax=1)
                header = col_headers[col_idx] + "\n" if row_idx == 0 else ""
                axes[row_idx, col_idx].set_title(header + labels[col_idx], fontsize=7)
                axes[row_idx, col_idx].axis("off")

                # Green border for matched, red for mismatched
                if col_idx == 2:
                    for spine in axes[row_idx, col_idx].spines.values():
                        spine.set_edgecolor("green")
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                elif col_idx == 4:
                    for spine in axes[row_idx, col_idx].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)
                        spine.set_visible(True)

        plt.tight_layout()
        out = f"fda_bgaware_{domain}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2 (per domain): Beta/lambda sweep on MATCHED pairs only
    #   Shows that with correct pairing, a wider range of params works well
    # -----------------------------------------------------------------------
    sweep_configs = [
        ("β=0.01 λ=0.5", 0.01, 0.5),
        ("β=0.03 λ=0.5", 0.03, 0.5),
        ("β=0.05 λ=0.3", 0.05, 0.3),
        ("β=0.05 λ=0.5", 0.05, 0.5),
        ("β=0.05 λ=0.7", 0.05, 0.7),
        ("β=0.08 λ=0.5", 0.08, 0.5),
    ]

    for domain in domains:
        tgt_by_bg = tgt_by_domain[domain]

        rows = []
        for bg_type in ["black", "earth"]:
            src_list = src_by_bg[bg_type]
            tgt_list = tgt_by_bg[bg_type]
            if not src_list or not tgt_list:
                continue
            for i in range(min(args.n_show, len(src_list))):
                src_name, src_img = src_list[i]
                tgt_name, tgt_img = rng.choice(tgt_list)
                rows.append((src_name, src_img, bg_type, tgt_name, tgt_img))

        if not rows:
            continue

        n_rows = len(rows)
        n_cols = 2 + len(sweep_configs)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows), squeeze=False,
        )
        fig.suptitle(
            f"FDA parameter sweep (matched bg): synthetic → {domain}",
            fontsize=13, fontweight="bold",
        )

        for row_idx, (src_name, src_img, bg_type, tgt_name, tgt_img) in enumerate(rows):
            tgt_resized = resize_to_match(tgt_img, src_img.shape)

            axes[row_idx, 0].imshow(src_img, cmap="gray", vmin=0, vmax=1)
            header = "Source\n" if row_idx == 0 else ""
            axes[row_idx, 0].set_title(f"{header}{src_name}\n[{bg_type}]", fontsize=7)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
            header = "Target\n" if row_idx == 0 else ""
            axes[row_idx, 1].set_title(f"{header}{tgt_name}", fontsize=7)
            axes[row_idx, 1].axis("off")

            for col_idx, (label, beta, lambd) in enumerate(sweep_configs, start=2):
                result = fda_blend(src_img, tgt_resized, beta, lambd)
                axes[row_idx, col_idx].imshow(result, cmap="gray", vmin=0, vmax=1)
                axes[row_idx, col_idx].set_title(label if row_idx == 0 else "", fontsize=7)
                axes[row_idx, col_idx].axis("off")

        plt.tight_layout()
        out = f"fda_sweep_{domain}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 3: Background classification summary
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), squeeze=False)
    fig.suptitle(
        f"Background classification examples  (var_threshold={args.var_threshold})",
        fontsize=13, fontweight="bold",
    )

    for col, bg_type in enumerate(["black", "earth"]):
        samples = src_by_bg[bg_type][:3]
        for row, (name, img) in enumerate(samples):
            if row >= 2:
                break
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"synthetic [{bg_type}]\n{name}", fontsize=7)
            ax.axis("off")

    col_offset = 2
    for domain in domains:
        tgt_by_bg = tgt_by_domain[domain]
        for bg_idx, bg_type in enumerate(["black", "earth"]):
            samples = tgt_by_bg[bg_type][:2]
            for row, (name, img) in enumerate(samples):
                if row >= 2:
                    break
                ci = col_offset + bg_idx
                ax = axes[row, ci]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{domain} [{bg_type}]\n{name}", fontsize=7)
                ax.axis("off")
        col_offset += 2

    # Hide unused axes
    for ax in axes.ravel():
        if not ax.images:
            ax.axis("off")

    plt.tight_layout()
    out = "fda_bg_classification.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
