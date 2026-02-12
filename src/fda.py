"""Fourier Domain Adaptation (FDA) for sim-to-real style transfer.

Transfers the low-frequency amplitude spectrum from real target-domain images
onto synthetic source images, bridging the visual domain gap while preserving
spatial structure (edges, keypoints).

Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
Segmentation" (CVPR 2020).
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Background classification
# ---------------------------------------------------------------------------

def classify_background(img: np.ndarray, var_threshold: float = 0.003) -> str:
    """Classify an image as 'earth' or 'black' background.

    Uses spatial variance in border patches. Earth textures (clouds,
    continents, horizon gradients) produce high local variance even if they
    only cover a corner. Uniform backgrounds — pitch-black or ambient-bright
    (lightbox) — have low variance everywhere.

    Args:
        img: (H, W) float image in [0, 1]
        var_threshold: per-patch variance threshold on [0, 1] scale.

    Returns:
        'earth' or 'black'
    """
    h, w = img.shape
    margin_h = max(1, int(h * 0.10))
    margin_w = max(1, int(w * 0.10))

    strips = [
        img[:margin_h, :],
        img[-margin_h:, :],
        img[margin_h:-margin_h, :margin_w],
        img[margin_h:-margin_h, -margin_w:],
    ]

    n_patches_per_strip = 4
    for strip in strips:
        sh, sw = strip.shape
        if sh == 0 or sw == 0:
            continue
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


# ---------------------------------------------------------------------------
# FDA core
# ---------------------------------------------------------------------------

def fda_blend(source: np.ndarray, target: np.ndarray, beta: float = 0.05,
              lambd: float = 0.5) -> np.ndarray:
    """FDA with blending: interpolate low-freq amplitude.

    mixed_amp = (1 - lambd) * src_amp + lambd * tgt_amp  (in low-freq region)

    Args:
        source: (H, W) float image in [0, 1]
        target: (H, W) float image in [0, 1], same shape as source
        beta: fraction of spectrum to blend (0.01-0.10 typical)
        lambd: blending strength (0=no change, 1=full target style)

    Returns:
        (H, W) float image in [0, 1]
    """
    h, w = source.shape

    src_fft = np.fft.fftshift(np.fft.fft2(source))
    tgt_fft = np.fft.fftshift(np.fft.fft2(target))

    src_amp = np.abs(src_fft)
    src_phase = np.angle(src_fft)
    tgt_amp = np.abs(tgt_fft)

    # Low-frequency mask: central rectangle
    bh = max(1, int(h * beta))
    bw = max(1, int(w * beta))
    ch, cw = h // 2, w // 2

    mixed_amp = src_amp.copy()
    mixed_amp[ch - bh : ch + bh, cw - bw : cw + bw] = (
        (1 - lambd) * src_amp[ch - bh : ch + bh, cw - bw : cw + bw]
        + lambd * tgt_amp[ch - bh : ch + bh, cw - bw : cw + bw]
    )

    result = np.fft.ifft2(np.fft.ifftshift(mixed_amp * np.exp(1j * src_phase))).real
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Style pool
# ---------------------------------------------------------------------------

class FDAStylePool:
    """Pool of real-domain images for FDA style transfer during training.

    Organizes style images by background type (earth/black) for matched
    pairing. Loads target images on-the-fly to avoid excessive memory usage.

    Args:
        image_dirs: mapping of domain name to image directory path,
                    e.g. {"lightbox": "/path/to/images/lightbox"}
        style_lists: mapping of domain name to style-list file path,
                     e.g. {"lightbox": "data/splits/lightbox_style.txt"}
        bg_labels: mapping of domain name to dict of {filename: bg_type},
                   e.g. {"lightbox": {"img001.jpg": "black", ...}}
        beta: FDA low-frequency bandwidth
        lambd: FDA blending strength
    """

    def __init__(
        self,
        image_dirs: dict[str, str],
        style_lists: dict[str, str],
        bg_labels: dict[str, dict[str, str]],
        beta: float = 0.05,
        lambd: float = 0.5,
    ):
        self.beta = beta
        self.lambd = lambd

        # Group style image paths by background type
        self.pools: dict[str, list[Path]] = {"earth": [], "black": []}

        for domain, list_path in style_lists.items():
            img_dir = Path(image_dirs[domain])
            with open(list_path, "r") as f:
                filenames = [line.strip() for line in f if line.strip()]

            domain_bg = bg_labels.get(domain, {})
            for fname in filenames:
                bg_type = domain_bg.get(fname, "black")
                self.pools[bg_type].append(img_dir / fname)

        n_earth = len(self.pools["earth"])
        n_black = len(self.pools["black"])
        print(f"  FDAStylePool: {n_black} black + {n_earth} earth = {n_black + n_earth} total")

        if n_black == 0 and n_earth == 0:
            raise ValueError("FDA style pool is empty — check split files and paths")

    def apply(self, source_crop: Image.Image, bg_type: str) -> Image.Image:
        """Apply FDA to a source crop using a random target from the matching pool.

        Args:
            source_crop: PIL Image (RGB, the bbox crop before resize)
            bg_type: 'earth' or 'black'

        Returns:
            PIL Image (RGB, same size as source_crop)
        """
        pool = self.pools.get(bg_type, [])
        if not pool:
            # Fall back to the other pool if this one is empty
            pool = self.pools["black"] or self.pools["earth"]
        if not pool:
            return source_crop

        # Pick a random target
        target_path = random.choice(pool)
        target_img = Image.open(target_path).convert("L")
        target_arr = np.array(target_img, dtype=np.float32) / 255.0

        # Convert source crop to grayscale float
        source_gray = source_crop.convert("L")
        source_arr = np.array(source_gray, dtype=np.float32) / 255.0

        # Resize target to match source crop dimensions
        sh, sw = source_arr.shape
        if target_arr.shape != (sh, sw):
            target_resized = Image.fromarray(
                (target_arr * 255).astype(np.uint8)
            ).resize((sw, sh), Image.BILINEAR)
            target_arr = np.array(target_resized, dtype=np.float32) / 255.0

        # Apply FDA blend
        result_arr = fda_blend(source_arr, target_arr, self.beta, self.lambd)

        # Convert back to RGB PIL (grayscale repeated 3x)
        result_uint8 = (result_arr * 255).astype(np.uint8)
        result_pil = Image.fromarray(result_uint8, mode="L").convert("RGB")

        return result_pil
