"""
TeleStyle-based style transfer for SPEED+ domain adaptation.

Transfers sunlamp (or lightbox) visual style onto synthetic training images
while preserving content/geometry — labels remain valid.

Pose-aware matching: for each content image, selects the style reference
with the most similar pose (quaternion + translation distance), so that
viewpoint/lighting is consistent and the diffusion model has an easier job.

Usage:
    python scripts/telestyle_transfer.py \
        --content_dir /path/to/speedplus_yolo/images/train \
        --style_dir /path/to/speedplus_yolo/images/sunlamp \
        --output_dir ./outputs/telestyle_sunlamp \
        --content_poses /path/to/speedplusv2/synthetic/train.json \
        --style_poses /path/to/speedplusv2/sunlamp/test.json \
        --max_images 20 \
        --seed 42

NOTE: output_dir should always be a LOCAL folder (e.g. ./outputs/...),
never inside the original dataset directory.

The output directory will contain stylized images with the SAME filenames
as the originals, so you can reuse existing label files directly.

A matching log (match_log.json) is saved alongside the output images,
recording which style image was paired with each content image and the
pose distance between them.

Requirements:
    pip install git+https://github.com/modelscope/DiffSynth-Studio.git@11315d7
    pip install transformers==4.57.3 diffusers==0.36.0 accelerate pillow safetensors sentencepiece
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Pose matching utilities
# ---------------------------------------------------------------------------

def load_pose_dict(json_path):
    """Load SPEED+ pose JSON into {filename: {"q": [w,x,y,z], "t": [x,y,z]}} dict."""
    with open(json_path) as f:
        data = json.load(f)
    poses = {}
    for entry in data:
        q = np.array(entry["q_vbs2tango_true"], dtype=np.float64)
        t = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float64)
        poses[entry["filename"]] = {"q": q, "t": t}
    return poses


def quaternion_distance(q1, q2):
    """Geodesic distance between two quaternions in radians.
    d = 2 * arccos(|q1 . q2|)
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def pose_distance(pose1, pose2, rotation_weight=1.0, translation_weight=0.5):
    """Combined pose distance: weighted sum of rotation (rad) and normalized translation error."""
    q_dist = quaternion_distance(pose1["q"], pose2["q"])
    t1, t2 = pose1["t"], pose2["t"]
    t_norm = np.linalg.norm(t1)
    if t_norm > 1e-6:
        t_dist = np.linalg.norm(t1 - t2) / t_norm
    else:
        t_dist = np.linalg.norm(t1 - t2)
    return rotation_weight * q_dist + translation_weight * t_dist


def build_style_pose_index(style_poses):
    """Pre-compute arrays for fast nearest-neighbor lookup."""
    filenames = list(style_poses.keys())
    quats = np.array([style_poses[f]["q"] for f in filenames])
    trans = np.array([style_poses[f]["t"] for f in filenames])
    return filenames, quats, trans


def find_nearest_style(content_pose, style_filenames, style_quats, style_trans,
                       rotation_weight=1.0, translation_weight=0.5, top_k=3, rng=None):
    """Find the best matching style image by pose distance.

    Returns one of the top_k closest matches (randomly chosen) to add variety
    while still keeping pose similarity high.
    """
    q_c = content_pose["q"]
    t_c = content_pose["t"]

    # Quaternion distances (vectorized)
    dots = np.abs(style_quats @ q_c)
    np.clip(dots, 0.0, 1.0, out=dots)
    q_dists = 2.0 * np.arccos(dots)

    # Translation distances (vectorized)
    t_norm = np.linalg.norm(t_c)
    if t_norm > 1e-6:
        t_dists = np.linalg.norm(style_trans - t_c, axis=1) / t_norm
    else:
        t_dists = np.linalg.norm(style_trans - t_c, axis=1)

    combined = rotation_weight * q_dists + translation_weight * t_dists

    # Pick randomly from top_k nearest
    top_indices = np.argpartition(combined, min(top_k, len(combined) - 1))[:top_k]
    if rng is not None:
        chosen = rng.choice(top_indices)
    else:
        chosen = top_indices[np.argmin(combined[top_indices])]

    return style_filenames[chosen], float(combined[chosen]), float(np.degrees(q_dists[chosen]))


# ---------------------------------------------------------------------------
# TeleStyle inference
# ---------------------------------------------------------------------------

class TeleStyleTransfer:

    def __init__(self, device="cuda"):
        self.device = device
        self._load_models()

    def _load_models(self):
        print("Loading Qwen-Image-Edit base model...")
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="text_encoder/model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                ),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                download_source="huggingface",
                origin_file_pattern="processor/",
            ),
        )

        print("Loading TeleStyle LoRA weights...")
        telestyle_path = hf_hub_download(
            repo_id="Tele-AI/TeleStyle",
            filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
        )
        speedup_path = hf_hub_download(
            repo_id="Tele-AI/TeleStyle",
            filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        )
        self.pipe.load_lora(self.pipe.dit, telestyle_path)
        self.pipe.load_lora(self.pipe.dit, speedup_path)
        print("Models loaded successfully.")

    def transfer(self, content_path, style_path, seed=123, min_edge=1024):
        """Transfer style from style_path onto content_path, preserving content geometry."""
        w, h = Image.open(content_path).convert("RGB").size
        min_edge = min_edge - min_edge % 16

        if w > h:
            r = w / h
            h = min_edge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = min_edge
            h = int(w * r) - int(w * r) % 16

        images = [
            Image.open(content_path).convert("RGB").resize((w, h)),
            Image.open(style_path).convert("RGB").resize((min_edge, min_edge)),
        ]

        prompt = (
            "Style Transfer the style of Figure 2 to Figure 1, "
            "and keep the content and characteristics of Figure 1."
        )

        image = self.pipe(
            prompt,
            edit_image=images,
            seed=seed,
            num_inference_steps=4,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0,
        )
        return image


def main():
    parser = argparse.ArgumentParser(description="TeleStyle domain adaptation for SPEED+")
    parser.add_argument("--content_dir", type=str, required=True,
                        help="Directory with source images (e.g. synthetic train)")
    parser.add_argument("--style_dir", type=str, required=True,
                        help="Directory with style reference images (e.g. sunlamp)")
    parser.add_argument("--output_dir", type=str, default="./outputs/telestyle",
                        help="Output directory for stylized images (always use a local folder)")
    parser.add_argument("--content_poses", type=str, default=None,
                        help="Pose JSON for content images (e.g. synthetic/train.json). "
                             "If provided, enables pose-aware style matching.")
    parser.add_argument("--style_poses", type=str, default=None,
                        help="Pose JSON for style images (e.g. sunlamp/test.json)")
    parser.add_argument("--num_style_refs", type=int, default=0,
                        help="Number of style refs to subsample (0 = use all for pose matching)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Pick randomly among top_k nearest poses for variety")
    parser.add_argument("--rotation_weight", type=float, default=1.0,
                        help="Weight for rotation distance in pose matching")
    parser.add_argument("--translation_weight", type=float, default=0.5,
                        help="Weight for translation distance in pose matching")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max content images to process (0 = all)")
    parser.add_argument("--min_edge", type=int, default=1024,
                        help="Minimum edge size for processing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already exist in output_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Collect content images
    content_images = sorted([
        f for f in os.listdir(args.content_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if args.max_images > 0:
        content_images = content_images[:args.max_images]

    # Collect style images
    style_image_files = sorted([
        f for f in os.listdir(args.style_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    # ------- Pose-aware matching -------
    use_pose_matching = args.content_poses and args.style_poses
    if use_pose_matching:
        print("Loading pose labels for pose-aware style matching...")
        content_poses = load_pose_dict(args.content_poses)
        style_poses = load_pose_dict(args.style_poses)

        # Optionally subsample style pool
        if args.num_style_refs > 0 and args.num_style_refs < len(style_image_files):
            random.seed(args.seed)
            style_image_files = random.sample(style_image_files, args.num_style_refs)

        # Filter to images that have pose labels
        style_image_files = [f for f in style_image_files if f in style_poses]
        style_fnames, style_quats, style_trans = build_style_pose_index(
            {f: style_poses[f] for f in style_image_files}
        )
        print(f"Pose matching: {len(style_fnames)} style refs available")
    else:
        # Fallback: random rotation through a pool (old behavior)
        random.seed(args.seed)
        n_refs = args.num_style_refs if args.num_style_refs > 0 else 10
        style_pool = random.sample(style_image_files, min(n_refs, len(style_image_files)))
        style_pool = [os.path.join(args.style_dir, f) for f in style_pool]
        print(f"Random style pool: {len(style_pool)} reference images (no pose matching)")

    print(f"Content images to process: {len(content_images)}")

    # Load model
    engine = TeleStyleTransfer(device="cuda")

    # Match log: records which style was used for each content image
    match_log = []

    # Process
    for i, fname in enumerate(content_images):
        out_path = os.path.join(args.output_dir, fname)

        if args.resume and os.path.exists(out_path):
            print(f"[{i+1}/{len(content_images)}] Skipping {fname} (already exists)")
            continue

        content_path = os.path.join(args.content_dir, fname)

        # Select style image
        if use_pose_matching and fname in content_poses:
            style_fname, dist, rot_deg = find_nearest_style(
                content_poses[fname], style_fnames, style_quats, style_trans,
                rotation_weight=args.rotation_weight,
                translation_weight=args.translation_weight,
                top_k=args.top_k,
                rng=rng,
            )
            style_path = os.path.join(args.style_dir, style_fname)
            print(f"[{i+1}/{len(content_images)}] {fname} <- {style_fname} "
                  f"(pose_dist={dist:.3f}, rot={rot_deg:.1f}°)")
            match_log.append({
                "content": fname,
                "style": style_fname,
                "pose_distance": round(dist, 4),
                "rotation_distance_deg": round(rot_deg, 2),
            })
        else:
            # Fallback: cycle through pool
            style_path = style_pool[i % len(style_pool)]
            style_fname = os.path.basename(style_path)
            print(f"[{i+1}/{len(content_images)}] {fname} <- {style_fname} (random)")
            match_log.append({
                "content": fname,
                "style": style_fname,
                "pose_distance": None,
                "rotation_distance_deg": None,
            })

        with torch.no_grad():
            result = engine.transfer(
                content_path, style_path,
                seed=args.seed + i,
                min_edge=args.min_edge,
            )

        # Save with original filename so labels stay valid
        result.save(out_path)

    # Save match log
    log_path = os.path.join(args.output_dir, "match_log.json")
    with open(log_path, "w") as f:
        json.dump(match_log, f, indent=2)

    print(f"\nDone! Stylized images saved to {args.output_dir}")
    print(f"Match log saved to {log_path}")
    print("Labels from the original content_dir can be reused directly.")


if __name__ == "__main__":
    main()
