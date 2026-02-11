import random
import math

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


def _apply_sun_flare(image: torch.Tensor, intensity_range=(0.3, 1.0), radius_range=(0.2, 0.6)):
    """Simulate a sun flare: bright elliptical glow from a random edge.

    The flare source is placed at the image edge (simulating the sun just
    outside the frame), with an elliptical falloff blending into the image.

    Args:
        image: (C, H, W) tensor in [0, 1]
        intensity_range: min/max peak brightness added
        radius_range: min/max flare radius as fraction of image size
    """
    _, H, W = image.shape
    intensity = random.uniform(*intensity_range)
    radius_frac = random.uniform(*radius_range)

    # Place flare source at a random edge position
    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        cy, cx = 0.0, random.uniform(0.2, 0.8)
    elif edge == "bottom":
        cy, cx = 1.0, random.uniform(0.2, 0.8)
    elif edge == "left":
        cy, cx = random.uniform(0.2, 0.8), 0.0
    else:
        cy, cx = random.uniform(0.2, 0.8), 1.0

    # Build elliptical distance field
    y_coords = torch.linspace(0, 1, H, device=image.device)
    x_coords = torch.linspace(0, 1, W, device=image.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Elongate flare along the edge (wider parallel to edge, narrower into image)
    if edge in ("top", "bottom"):
        rx, ry = radius_frac * 1.5, radius_frac
    else:
        rx, ry = radius_frac, radius_frac * 1.5

    dist_sq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    flare = intensity * torch.exp(-dist_sq * 2.0)  # Gaussian falloff

    # Add slight bloom: secondary wider, dimmer glow
    bloom = (intensity * 0.3) * torch.exp(-dist_sq * 0.5)
    flare = flare + bloom

    return (image + flare.unsqueeze(0)).clamp(0.0, 1.0)


def _apply_brightness_gradient(image: torch.Tensor, strength_range=(0.2, 0.6)):
    """Apply a linear brightness gradient across the image.

    Simulates directional lighting: one side of the spacecraft is lit,
    the other is in shadow — common in space imagery.

    Args:
        image: (C, H, W) tensor in [0, 1]
        strength_range: min/max gradient strength (0 = no effect, 1 = full black-to-white)
    """
    _, H, W = image.shape
    strength = random.uniform(*strength_range)
    angle = random.uniform(0, 2 * math.pi)

    y_coords = torch.linspace(-1, 1, H, device=image.device)
    x_coords = torch.linspace(-1, 1, W, device=image.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Directional ramp: project coordinates onto random angle direction
    gradient = math.cos(angle) * xx + math.sin(angle) * yy  # range [-sqrt2, sqrt2]
    gradient = gradient / (2 ** 0.5)  # normalize to [-1, 1]
    gradient = gradient * strength  # scale

    return (image + gradient.unsqueeze(0)).clamp(0.0, 1.0)


def _apply_random_gamma(image: torch.Tensor, gamma_range=(0.5, 2.0)):
    """Apply random gamma correction.

    Simulates camera response curve differences between synthetic renders
    and real sensors. gamma < 1 brightens, gamma > 1 darkens.

    Args:
        image: (C, H, W) tensor in [0, 1]
        gamma_range: (min_gamma, max_gamma)
    """
    # Sample gamma in log space for symmetric distribution around 1.0
    log_gamma = random.uniform(math.log(gamma_range[0]), math.log(gamma_range[1]))
    gamma = math.exp(log_gamma)
    return image.pow(gamma).clamp(0.0, 1.0)


def _apply_motion_blur(image: torch.Tensor, kernel_range=(5, 15)):
    """Apply directional motion blur with a random angle.

    Simulates slight spacecraft or camera motion during exposure.
    Unlike Gaussian blur (isotropic), this is a directional streak.

    Args:
        image: (C, H, W) tensor in [0, 1]
        kernel_range: (min, max) kernel size in pixels (odd values)
    """
    ksize = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)  # odd
    angle = random.uniform(0, 360)

    # Build a motion blur kernel: a line of 1s rotated by angle
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= kernel.sum() + 1e-8

    # Apply per-channel via cv2.filter2D
    img_np = image.permute(1, 2, 0).numpy()  # (H, W, C)
    blurred = cv2.filter2D(img_np, -1, kernel)
    return torch.from_numpy(blurred).permute(2, 0, 1).clamp(0.0, 1.0)


def _apply_clahe(image: torch.Tensor, clip_limit_range=(1.0, 4.0), grid_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Locally adjusts contrast, simulating different camera auto-exposure
    behaviors. Particularly useful for the lightbox/sunlamp domain gap
    where contrast characteristics differ from synthetic renders.

    Args:
        image: (C, H, W) tensor in [0, 1]
        clip_limit_range: (min, max) CLAHE clip limit
        grid_size: tile grid size for local histogram equalization
    """
    clip_limit = random.uniform(*clip_limit_range)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H, W, C)

    # Apply CLAHE per channel
    for c in range(img_np.shape[2]):
        img_np[:, :, c] = clahe.apply(img_np[:, :, c])

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)


def _apply_channel_dropout(image: torch.Tensor, drop_range=(1, 2)):
    """Randomly zero out 1 or 2 RGB channels.

    Since the original images are grayscale repeated 3x, this simulates
    sensor channel noise or imbalance. The remaining channel(s) still
    carry the full intensity information.

    Args:
        image: (C, H, W) tensor in [0, 1]
        drop_range: (min, max) number of channels to drop
    """
    C = image.shape[0]
    n_drop = random.randint(*drop_range)
    n_drop = min(n_drop, C - 1)  # keep at least 1 channel
    drop_idx = random.sample(range(C), n_drop)
    out = image.clone()
    for idx in drop_idx:
        out[idx] = 0.0
    return out


class KeypointTransform:
    """Applies photometric augmentations to a cropped image.

    Keypoints are in [0,1] crop-relative coordinates and are NOT modified
    (all augmentations are purely photometric — no spatial transforms).
    """

    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True,
        color_jitter: float = 0.3,
        gaussian_blur_prob: float = 0.0,
        gaussian_blur_kernel: tuple = (3, 7),
        gaussian_noise_prob: float = 0.0,
        gaussian_noise_std: float = 0.02,
        random_erasing_prob: float = 0.0,
        sun_flare_prob: float = 0.0,
        sun_flare_intensity: tuple = (0.3, 1.0),
        brightness_gradient_prob: float = 0.0,
        brightness_gradient_strength: tuple = (0.2, 0.6),
        random_gamma_prob: float = 0.0,
        random_gamma_range: tuple = (0.5, 2.0),
        motion_blur_prob: float = 0.0,
        motion_blur_kernel: tuple = (5, 15),
        clahe_prob: float = 0.0,
        clahe_clip_limit: tuple = (1.0, 4.0),
        channel_dropout_prob: float = 0.0,
        imagenet_normalize: bool = True,
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.sun_flare_prob = sun_flare_prob
        self.sun_flare_intensity = sun_flare_intensity
        self.brightness_gradient_prob = brightness_gradient_prob
        self.brightness_gradient_strength = brightness_gradient_strength
        self.random_gamma_prob = random_gamma_prob
        self.random_gamma_range = random_gamma_range
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel = motion_blur_kernel
        self.clahe_prob = clahe_prob
        self.clahe_clip_limit = clahe_clip_limit
        self.channel_dropout_prob = channel_dropout_prob

        # Photometric augmentations (don't affect keypoints)
        if is_train:
            self.color_aug = transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter * 0.5,
            )
            if gaussian_blur_prob > 0:
                self.gaussian_blur = transforms.RandomApply(
                    [transforms.GaussianBlur(
                        kernel_size=list(gaussian_blur_kernel),
                        sigma=(0.1, 2.0),
                    )],
                    p=gaussian_blur_prob,
                )
            else:
                self.gaussian_blur = None
            if random_erasing_prob > 0:
                self.random_erasing = transforms.RandomErasing(
                    p=random_erasing_prob,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            else:
                self.random_erasing = None
        else:
            self.color_aug = None
            self.gaussian_blur = None
            self.random_erasing = None

        self.to_tensor = transforms.ToTensor()

        if imagenet_normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.normalize = None

    def __call__(self, image: Image.Image, keypoints: np.ndarray, visibility: np.ndarray):
        """
        Args:
            image: PIL Image (crop)
            keypoints: (N, 2) in [0,1] crop-relative coords
            visibility: (N,) int array

        Returns:
            image_tensor: (3, H, W)
            keypoints: (N, 2) unchanged
            visibility: (N,) unchanged
        """
        kp = keypoints.copy()
        vis = visibility.copy()

        if self.is_train:
            # Photometric augmentations only (no spatial transforms)
            if self.color_aug is not None:
                image = self.color_aug(image)

            # Gaussian blur (on PIL image)
            if self.gaussian_blur is not None:
                image = self.gaussian_blur(image)

        # Resize to target size
        image = F.resize(image, [self.image_size, self.image_size])

        # To tensor and normalize
        image = self.to_tensor(image)

        # Gaussian noise (on tensor, before normalization)
        if self.is_train and self.gaussian_noise_prob > 0 and random.random() < self.gaussian_noise_prob:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = (image + noise).clamp(0.0, 1.0)

        # Space lighting & domain gap augmentations (on tensor, before normalization)
        if self.is_train:
            if self.sun_flare_prob > 0 and random.random() < self.sun_flare_prob:
                image = _apply_sun_flare(image, self.sun_flare_intensity)
            if self.brightness_gradient_prob > 0 and random.random() < self.brightness_gradient_prob:
                image = _apply_brightness_gradient(image, self.brightness_gradient_strength)
            if self.random_gamma_prob > 0 and random.random() < self.random_gamma_prob:
                image = _apply_random_gamma(image, self.random_gamma_range)
            if self.motion_blur_prob > 0 and random.random() < self.motion_blur_prob:
                image = _apply_motion_blur(image, self.motion_blur_kernel)
            if self.clahe_prob > 0 and random.random() < self.clahe_prob:
                image = _apply_clahe(image, self.clahe_clip_limit)
            if self.channel_dropout_prob > 0 and random.random() < self.channel_dropout_prob:
                image = _apply_channel_dropout(image)

        if self.normalize is not None:
            image = self.normalize(image)

        # Random erasing (on normalized tensor)
        if self.is_train and self.random_erasing is not None:
            image = self.random_erasing(image)

        return image, kp, vis
