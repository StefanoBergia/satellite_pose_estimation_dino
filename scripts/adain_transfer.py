"""
AdaIN (Adaptive Instance Normalization) style transfer for SPEED+ domain adaptation.

Transfers lighting/contrast/reflections from sunlamp (or lightbox) images onto
synthetic training images by matching feature statistics (mean, variance) in
VGG feature space. This preserves geometry perfectly — only appearance changes.

Supports pose-aware matching: pairs each content image with the style reference
that has the most similar pose.

The alpha parameter controls transfer strength:
  - alpha=1.0: full style transfer
  - alpha=0.5: blend halfway between content and style features
  - alpha=0.0: no change (original content)

Usage:
    python scripts/adain_transfer.py \
        --content_dir /path/to/speedplus_yolo/images/train \
        --style_dir /path/to/speedplus_yolo/images/sunlamp \
        --output_dir ./outputs/adain_sunlamp \
        --content_poses /path/to/speedplusv2/synthetic/train.json \
        --style_poses /path/to/speedplusv2/sunlamp/test.json \
        --alpha 1.0 \
        --max_images 20

NOTE: output_dir should always be a LOCAL folder (e.g. ./outputs/...),
never inside the original dataset directory.

Requirements:
    pip install torch torchvision pillow numpy
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights


# ---------------------------------------------------------------------------
# Pose matching (reused from telestyle_transfer.py)
# ---------------------------------------------------------------------------

def load_pose_dict(json_path):
    with open(json_path) as f:
        data = json.load(f)
    poses = {}
    for entry in data:
        q = np.array(entry["q_vbs2tango_true"], dtype=np.float64)
        t = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float64)
        poses[entry["filename"]] = {"q": q, "t": t}
    return poses


def build_style_pose_index(style_poses):
    filenames = list(style_poses.keys())
    quats = np.array([style_poses[f]["q"] for f in filenames])
    trans = np.array([style_poses[f]["t"] for f in filenames])
    return filenames, quats, trans


def find_nearest_style(content_pose, style_filenames, style_quats, style_trans,
                       rotation_weight=1.0, translation_weight=0.5, top_k=3, rng=None):
    q_c = content_pose["q"]
    t_c = content_pose["t"]

    dots = np.abs(style_quats @ q_c)
    np.clip(dots, 0.0, 1.0, out=dots)
    q_dists = 2.0 * np.arccos(dots)

    t_norm = np.linalg.norm(t_c)
    if t_norm > 1e-6:
        t_dists = np.linalg.norm(style_trans - t_c, axis=1) / t_norm
    else:
        t_dists = np.linalg.norm(style_trans - t_c, axis=1)

    combined = rotation_weight * q_dists + translation_weight * t_dists
    top_indices = np.argpartition(combined, min(top_k, len(combined) - 1))[:top_k]
    if rng is not None:
        chosen = rng.choice(top_indices)
    else:
        chosen = top_indices[np.argmin(combined[top_indices])]

    return style_filenames[chosen], float(combined[chosen]), float(np.degrees(q_dists[chosen]))


# ---------------------------------------------------------------------------
# VGG Encoder (up to relu4_1)
# ---------------------------------------------------------------------------

class VGGEncoder(nn.Module):
    """VGG19 feature extractor up to relu4_1 for AdaIN."""

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # relu1_1=1, relu2_1=6, relu3_1=11, relu4_1=20
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])   # relu1_1
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])   # relu2_1
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)])  # relu3_1
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(12, 21)]) # relu4_1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


# ---------------------------------------------------------------------------
# Simple Decoder (mirrors VGG encoder, relu4_1 -> image)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Lightweight decoder that inverts VGG features back to an image.
    Architecture mirrors VGG encoder from relu4_1 backwards, using
    nearest-neighbor upsampling instead of unpooling.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # From relu4_1: 512 channels
            nn.Conv2d(512, 256, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # relu3_1 level: 256 channels
            nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # relu2_1 level: 128 channels
            nn.Conv2d(128, 128, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # relu1_1 level: 64 channels
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# AdaIN core
# ---------------------------------------------------------------------------

def calc_mean_std(feat, eps=1e-5):
    """Calculate channel-wise mean and std of a feature map."""
    B, C = feat.shape[:2]
    mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    std = feat.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + eps
    return mean, std


def adain(content_feat, style_feat):
    """Adaptive Instance Normalization: match content features to style statistics."""
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


# ---------------------------------------------------------------------------
# Style transfer engine
# ---------------------------------------------------------------------------

class AdaINTransfer:

    def __init__(self, decoder_weights=None, device="cuda"):
        self.device = device
        self.encoder = VGGEncoder().to(device).eval()

        self.decoder = Decoder().to(device)
        if decoder_weights and os.path.exists(decoder_weights):
            print(f"Loading decoder weights from {decoder_weights}")
            self.decoder.load_state_dict(
                torch.load(decoder_weights, map_location=device, weights_only=True))
        else:
            print("WARNING: No decoder weights provided. Using random decoder.")
            print("For good results, download pre-trained AdaIN decoder weights.")
            print("See: https://github.com/naoto0804/pytorch-AdaIN")
            print("     Download decoder.pth from the repo releases.")
        self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.denorm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.denorm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def transfer(self, content_img, style_img, alpha=1.0):
        """Run AdaIN style transfer.

        Args:
            content_img: PIL Image
            style_img: PIL Image
            alpha: blending strength (0=content, 1=full style)

        Returns:
            PIL Image
        """
        c_tensor = self.transform(content_img).unsqueeze(0).to(self.device)
        s_tensor = self.transform(style_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            c_feats = self.encoder(c_tensor)
            s_feats = self.encoder(s_tensor)

            # AdaIN on relu4_1 features
            t = adain(c_feats[-1], s_feats[-1])

            # Alpha blending in feature space
            t = alpha * t + (1 - alpha) * c_feats[-1]

            # Decode back to image
            output = self.decoder(t)

        # Denormalize
        output = output.squeeze(0) * self.denorm_std + self.denorm_mean
        output = output.clamp(0, 1)

        # To PIL
        output_np = (output.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(output_np)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AdaIN style transfer for SPEED+ domain adaptation")
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/adain_transfer")
    parser.add_argument("--decoder_weights", type=str, default=None,
                        help="Path to pre-trained AdaIN decoder weights (decoder.pth). "
                             "Download from https://github.com/naoto0804/pytorch-AdaIN")
    parser.add_argument("--content_poses", type=str, default=None)
    parser.add_argument("--style_poses", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Style strength: 0.0=no change, 1.0=full transfer")
    parser.add_argument("--num_style_refs", type=int, default=0,
                        help="Subsample style pool (0=all)")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rotation_weight", type=float, default=1.0)
    parser.add_argument("--translation_weight", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize images to this size for processing (preserves aspect ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Content images
    content_images = sorted([
        f for f in os.listdir(args.content_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if args.max_images > 0:
        content_images = content_images[:args.max_images]

    # Style images
    style_image_files = sorted([
        f for f in os.listdir(args.style_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    # Pose matching setup
    use_pose_matching = args.content_poses and args.style_poses
    if use_pose_matching:
        print("Loading poses for pose-aware matching...")
        content_poses = load_pose_dict(args.content_poses)
        style_poses = load_pose_dict(args.style_poses)

        if args.num_style_refs > 0 and args.num_style_refs < len(style_image_files):
            random.seed(args.seed)
            style_image_files = random.sample(style_image_files, args.num_style_refs)

        style_image_files = [f for f in style_image_files if f in style_poses]
        style_fnames, style_quats, style_trans = build_style_pose_index(
            {f: style_poses[f] for f in style_image_files}
        )
        print(f"Pose matching: {len(style_fnames)} style refs")
    else:
        random.seed(args.seed)
        n_refs = args.num_style_refs if args.num_style_refs > 0 else 10
        style_pool = random.sample(style_image_files, min(n_refs, len(style_image_files)))
        print(f"Random style pool: {len(style_pool)} refs (no pose matching)")

    print(f"Content images: {len(content_images)}, alpha={args.alpha}")

    # Load model
    engine = AdaINTransfer(decoder_weights=args.decoder_weights, device="cuda")
    match_log = []

    for i, fname in enumerate(content_images):
        out_path = os.path.join(args.output_dir, fname)
        if args.resume and os.path.exists(out_path):
            print(f"[{i+1}/{len(content_images)}] Skipping {fname}")
            continue

        content_path = os.path.join(args.content_dir, fname)
        content_img = Image.open(content_path).convert("RGB")
        orig_size = content_img.size

        # Select style image
        if use_pose_matching and fname in content_poses:
            style_fname, dist, rot_deg = find_nearest_style(
                content_poses[fname], style_fnames, style_quats, style_trans,
                rotation_weight=args.rotation_weight,
                translation_weight=args.translation_weight,
                top_k=args.top_k, rng=rng,
            )
            style_path = os.path.join(args.style_dir, style_fname)
            print(f"[{i+1}/{len(content_images)}] {fname} <- {style_fname} "
                  f"(rot={rot_deg:.1f}deg, dist={dist:.3f})")
            match_log.append({
                "content": fname, "style": style_fname,
                "pose_distance": round(dist, 4),
                "rotation_distance_deg": round(rot_deg, 2),
            })
        else:
            style_fname = style_pool[i % len(style_pool)]
            style_path = os.path.join(args.style_dir, style_fname)
            print(f"[{i+1}/{len(content_images)}] {fname} <- {style_fname} (random)")
            match_log.append({
                "content": fname, "style": style_fname,
                "pose_distance": None, "rotation_distance_deg": None,
            })

        style_img = Image.open(style_path).convert("RGB")

        # Resize for processing (preserve aspect ratio)
        def resize_keep_ar(img, max_size):
            w, h = img.size
            ratio = max_size / max(w, h)
            if ratio < 1:
                return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            return img

        content_resized = resize_keep_ar(content_img, args.image_size)
        style_resized = resize_keep_ar(style_img, args.image_size)

        result = engine.transfer(content_resized, style_resized, alpha=args.alpha)

        # Resize back to original size
        if result.size != orig_size:
            result = result.resize(orig_size, Image.LANCZOS)

        result.save(out_path)

    # Save match log
    log_path = os.path.join(args.output_dir, "match_log.json")
    with open(log_path, "w") as f:
        json.dump(match_log, f, indent=2)

    print(f"\nDone! Output in {args.output_dir}")
    print(f"Match log: {log_path}")


if __name__ == "__main__":
    main()
