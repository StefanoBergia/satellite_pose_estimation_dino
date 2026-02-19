"""
Compare TeleStyle-augmented images against originals with keypoint + pose overlay.

Generates a 4-panel comparison for each image:
  1. Original        — with its keypoints + orientation axes (green)
  2. Style reference  — with its own keypoints + orientation axes (orange)
  3. Augmented        — with original keypoints + orientation (green) — check consistency
  4. Augmented        — with BOTH original (green) and style ref (orange) keypoints + axes

Requires:
  - match_log.json (auto-detected from stylized_dir/) to know which style ref was used
  - Pose JSONs for content and style images (SPEED+ format)
  - Camera intrinsics + 3D points for drawing orientation axes

Usage:
    python scripts/compare_style_transfer.py \
        --original_dir /path/to/speedplus_yolo/images/train \
        --stylized_dir ./outputs/telestyle_sunlamp \
        --label_dir /path/to/speedplus_yolo/labels/train \
        --style_dir /path/to/speedplus_yolo/images/sunlamp \
        --style_label_dir /path/to/speedplus_yolo/labels/sunlamp \
        --content_poses /path/to/speedplusv2/synthetic/train.json \
        --style_poses /path/to/speedplusv2/sunlamp/test.json \
        --camera camera.json \
        --points_3d tango3Dpoints.json \
        --output_dir ./outputs/style_transfer_comparison \
        --num_images 20
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KP_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8800", "#88FF00", "#0088FF", "#FF0088",
    "#FFFFFF",
]

WIREFRAME_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (1, 8), (5, 8),
    (2, 9), (6, 9),
    (3, 10), (7, 10),
]


# ---------------------------------------------------------------------------
# Pose / projection utilities
# ---------------------------------------------------------------------------

def load_pose_dict(json_path):
    """Load SPEED+ pose JSON into {filename: {"q": array, "t": array}}."""
    with open(json_path) as f:
        data = json.load(f)
    poses = {}
    for entry in data:
        q = np.array(entry["q_vbs2tango_true"], dtype=np.float64)
        t = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float64)
        poses[entry["filename"]] = {"q": q, "t": t}
    return poses


def quat_to_dcm(q):
    """Quaternion (w, x, y, z) scalar-first to 3x3 DCM (body-to-inertial).
    q_vbs2tango: camera(VBS) -> body(Tango).
    For projection we need body->camera: R = dcm.T
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def project_3d(points_3d, q, t, camera_matrix, img_w, img_h, display_w, display_h):
    """Project 3D points to display pixel coords using pose and camera.

    Args:
        points_3d: (K, 3)
        q: quaternion (w, x, y, z)
        t: translation (3,)
        camera_matrix: (3, 3)
        img_w, img_h: original image size
        display_w, display_h: display panel size

    Returns:
        pts_2d: (K, 2) display pixel coords, NaN if behind camera
    """
    # body->camera rotation
    R = quat_to_dcm(q).T
    pts_cam = (R @ points_3d.T).T + t  # (K, 3)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    sx = display_w / img_w
    sy = display_h / img_h

    pts_2d = np.full((len(points_3d), 2), np.nan)
    valid = pts_cam[:, 2] > 0
    pts_2d[valid, 0] = (fx * pts_cam[valid, 0] / pts_cam[valid, 2] + cx) * sx
    pts_2d[valid, 1] = (fy * pts_cam[valid, 1] / pts_cam[valid, 2] + cy) * sy
    return pts_2d


def draw_axes(draw, q, t, camera_matrix, img_w, img_h, display_w, display_h,
              axis_length=0.3, line_width=3, colors=("red", "green", "blue")):
    """Draw projected 3D orientation axes on the image."""
    R = quat_to_dcm(q).T
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    sx = display_w / img_w
    sy = display_h / img_h

    origin = t
    axes_3d = [
        t + R[:, 0] * axis_length,  # not R columns — use body axes projected
        t + R[:, 1] * axis_length,
        t + R[:, 2] * axis_length,
    ]
    # Actually: axes in camera frame. Body X/Y/Z axes endpoints in cam frame:
    axes_3d = [t + R @ (np.eye(3)[i] * axis_length) for i in range(3)]

    def proj(pt3d):
        if pt3d[2] <= 0:
            return None
        u = (fx * pt3d[0] / pt3d[2] + cx) * sx
        v = (fy * pt3d[1] / pt3d[2] + cy) * sy
        return (u, v)

    o2d = proj(origin)
    if o2d is None:
        return

    for ax, color in zip(axes_3d, colors):
        e2d = proj(ax)
        if e2d is not None:
            draw.line([o2d, e2d], fill=color, width=line_width)


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------

def parse_yolo_keypoints(label_path, img_w, img_h):
    """Parse YOLO keypoint label. Returns (keypoints_px, visibility, bbox)."""
    with open(label_path) as f:
        line = f.readline().strip()
    parts = line.split()
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    bbox = np.array([
        (cx - bw / 2) * img_w, (cy - bh / 2) * img_h,
        (cx + bw / 2) * img_w, (cy + bh / 2) * img_h,
    ])
    kp_data = parts[5:]
    num_kp = len(kp_data) // 3
    keypoints = np.zeros((num_kp, 2))
    visibility = np.zeros(num_kp)
    for i in range(num_kp):
        keypoints[i] = [float(kp_data[i*3]) * img_w, float(kp_data[i*3+1]) * img_h]
        visibility[i] = int(kp_data[i*3+2])
    return keypoints, visibility, bbox


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_keypoints_and_wireframe(image, keypoints, visibility, radius=5,
                                 draw_indices=True, wireframe_color="white",
                                 kp_colors=None):
    """Draw keypoints + wireframe. Returns new image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Wireframe
    for i, j in WIREFRAME_EDGES:
        if i < len(keypoints) and j < len(keypoints):
            if visibility[i] > 0 and visibility[j] > 0:
                draw.line([tuple(keypoints[i]), tuple(keypoints[j])],
                          fill=wireframe_color, width=1)

    # Keypoints
    for i in range(len(keypoints)):
        if visibility[i] == 0:
            continue
        x, y = keypoints[i]
        if kp_colors is not None:
            color = kp_colors[i % len(kp_colors)]
        else:
            color = KP_COLORS[i % len(KP_COLORS)]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=color, outline="black", width=1)
        if draw_indices:
            draw.text((x + radius + 2, y - radius), str(i), fill=color)

    return img


def add_label(image, text, position="top"):
    """Add a text label bar."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    bar_h = 24
    w = img.width
    if position == "top":
        draw.rectangle([0, 0, w, bar_h], fill=(0, 0, 0))
        draw.text((4, 4), text, fill="white")
    else:
        h = img.height
        draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
        draw.text((4, h - bar_h + 4), text, fill="white")
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare style-transferred images: 4-panel with keypoints + orientation"
    )
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--stylized_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True,
                        help="YOLO labels for original (content) images")
    parser.add_argument("--style_dir", type=str, required=True,
                        help="Directory with style reference images (e.g. sunlamp)")
    parser.add_argument("--style_label_dir", type=str, required=True,
                        help="YOLO labels for style reference images")
    parser.add_argument("--content_poses", type=str, required=True,
                        help="Pose JSON for content images (e.g. synthetic/train.json)")
    parser.add_argument("--style_poses", type=str, required=True,
                        help="Pose JSON for style images (e.g. sunlamp/test.json)")
    parser.add_argument("--camera", type=str, default="camera.json",
                        help="Camera intrinsics JSON")
    parser.add_argument("--points_3d", type=str, default="tango3Dpoints.json",
                        help="3D keypoints JSON")
    parser.add_argument("--match_log", type=str, default=None,
                        help="Path to match_log.json (auto-detected from stylized_dir/)")
    parser.add_argument("--output_dir", type=str, default="./outputs/style_transfer_comparison")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--display_width", type=int, default=480,
                        help="Width of each panel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load camera + 3D points
    with open(args.camera) as f:
        cam = json.load(f)
    camera_matrix = np.array(cam["cameraMatrix"], dtype=np.float64)
    img_w_orig, img_h_orig = cam["Nu"], cam["Nv"]  # 1920x1200

    with open(args.points_3d) as f:
        pts3d = json.load(f)
    points_3d = np.array(pts3d["points"], dtype=np.float64)

    # Load poses
    content_poses = load_pose_dict(args.content_poses)
    style_poses = load_pose_dict(args.style_poses)

    # Load match log
    match_log_path = args.match_log
    if match_log_path is None:
        auto_path = os.path.join(args.stylized_dir, "match_log.json")
        if os.path.exists(auto_path):
            match_log_path = auto_path
    match_log = {}
    if match_log_path and os.path.exists(match_log_path):
        with open(match_log_path) as f:
            for entry in json.load(f):
                match_log[entry["content"]] = entry
        print(f"Loaded match log: {len(match_log)} entries")

    # Collect stylized images
    stylized_files = sorted([
        f for f in os.listdir(args.stylized_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if 0 < args.num_images < len(stylized_files):
        stylized_files = random.sample(stylized_files, args.num_images)
        stylized_files.sort()

    print(f"Generating 4-panel comparisons for {len(stylized_files)} images")

    # Green-ish palette for content keypoints
    CONTENT_KP_COLORS = ["#00FF00"] * 11
    # Orange-ish palette for style ref keypoints
    STYLE_KP_COLORS = ["#FF8800"] * 11

    grid_tiles = []

    for idx, fname in enumerate(stylized_files):
        stem = Path(fname).stem
        log_entry = match_log.get(fname, {})
        style_ref_fname = log_entry.get("style")
        pose_dist = log_entry.get("pose_distance")
        rot_dist = log_entry.get("rotation_distance_deg")

        if not style_ref_fname:
            print(f"  [SKIP] {fname}: no match log entry")
            continue

        orig_path = os.path.join(args.original_dir, fname)
        aug_path = os.path.join(args.stylized_dir, fname)
        ref_path = os.path.join(args.style_dir, style_ref_fname)
        label_path = os.path.join(args.label_dir, stem + ".txt")
        ref_label_path = os.path.join(args.style_label_dir,
                                       Path(style_ref_fname).stem + ".txt")

        # Check all files exist
        missing = [p for p in [orig_path, aug_path, ref_path, label_path, ref_label_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] {fname}: missing {[os.path.basename(m) for m in missing]}")
            continue

        # Load images
        original = Image.open(orig_path).convert("RGB")
        augmented = Image.open(aug_path).convert("RGB")
        style_ref = Image.open(ref_path).convert("RGB")
        o_w, o_h = original.size
        r_w, r_h = style_ref.size

        # Resize augmented to match original
        if augmented.size != original.size:
            augmented = augmented.resize(original.size, Image.LANCZOS)

        # Display sizes
        scale_x = args.display_width / o_w
        display_h = int(o_h * scale_x)
        display_size = (args.display_width, display_h)

        orig_disp = original.resize(display_size, Image.LANCZOS)
        aug_disp = augmented.resize(display_size, Image.LANCZOS)
        ref_disp = style_ref.resize(display_size, Image.LANCZOS)

        # Parse keypoints
        content_kp, content_vis, _ = parse_yolo_keypoints(label_path, o_w, o_h)
        ref_kp, ref_vis, _ = parse_yolo_keypoints(ref_label_path, r_w, r_h)

        # Scale to display coords
        ckp = content_kp.copy()
        ckp[:, 0] *= args.display_width / o_w
        ckp[:, 1] *= display_h / o_h

        rkp = ref_kp.copy()
        rkp[:, 0] *= args.display_width / r_w
        rkp[:, 1] *= display_h / r_h

        # Get poses
        c_pose = content_poses.get(fname)
        s_pose = style_poses.get(style_ref_fname)

        # Common args for axes drawing
        axes_kwargs = dict(
            camera_matrix=camera_matrix,
            img_w=img_w_orig, img_h=img_h_orig,
            display_w=args.display_width, display_h=display_h,
            axis_length=0.3, line_width=3,
        )

        # === Panel 1: Original + content keypoints (green) + content axes ===
        p1 = draw_keypoints_and_wireframe(
            orig_disp, ckp, content_vis, radius=5,
            wireframe_color="#00FF00", kp_colors=CONTENT_KP_COLORS)
        if c_pose:
            draw = ImageDraw.Draw(p1)
            draw_axes(draw, c_pose["q"], c_pose["t"],
                      colors=("#00CC00", "#00CC00", "#00FF88"), **axes_kwargs)
        p1 = add_label(p1, f"1. Original: {fname}")

        # === Panel 2: Style ref + ref keypoints (orange) + ref axes ===
        p2 = draw_keypoints_and_wireframe(
            ref_disp, rkp, ref_vis, radius=5,
            wireframe_color="#FF8800", kp_colors=STYLE_KP_COLORS)
        if s_pose:
            draw = ImageDraw.Draw(p2)
            draw_axes(draw, s_pose["q"], s_pose["t"],
                      colors=("#FF8800", "#FF8800", "#FFAA44"), **axes_kwargs)
        dist_str = ""
        if rot_dist is not None:
            dist_str = f"  (rot={rot_dist:.1f}deg, dist={pose_dist:.2f})"
        p2 = add_label(p2, f"2. Style ref: {style_ref_fname}{dist_str}")

        # === Panel 3: Augmented + content keypoints (green) + content axes ===
        # This is the key check: original labels overlaid on augmented image
        p3 = draw_keypoints_and_wireframe(
            aug_disp, ckp, content_vis, radius=5,
            wireframe_color="#00FF00", kp_colors=CONTENT_KP_COLORS)
        if c_pose:
            draw = ImageDraw.Draw(p3)
            draw_axes(draw, c_pose["q"], c_pose["t"],
                      colors=("#00CC00", "#00CC00", "#00FF88"), **axes_kwargs)
        p3 = add_label(p3, "3. Augmented + original labels (green)")

        # === Panel 4: Augmented + BOTH keypoints + BOTH axes ===
        p4 = aug_disp.copy()
        # Draw style ref keypoints first (orange, behind)
        if s_pose:
            # Project 3D points using style pose -> these are where the ref's kp land
            # on the augmented image's coordinate system (they won't match, that's the point)
            pass
        p4 = draw_keypoints_and_wireframe(
            p4, rkp, ref_vis, radius=4,
            wireframe_color="#FF8800", kp_colors=STYLE_KP_COLORS, draw_indices=False)
        if s_pose:
            draw = ImageDraw.Draw(p4)
            draw_axes(draw, s_pose["q"], s_pose["t"],
                      colors=("#FF6600", "#FF6600", "#FFAA44"), **axes_kwargs)
        # Draw content keypoints on top (green, foreground)
        p4 = draw_keypoints_and_wireframe(
            p4, ckp, content_vis, radius=5,
            wireframe_color="#00FF00", kp_colors=CONTENT_KP_COLORS)
        if c_pose:
            draw = ImageDraw.Draw(p4)
            draw_axes(draw, c_pose["q"], c_pose["t"],
                      colors=("#00CC00", "#00CC00", "#00FF88"), **axes_kwargs)
        p4 = add_label(p4, "4. Augmented + both (green=orig, orange=ref)")

        # === Assemble 2x2 grid ===
        pw = args.display_width
        grid_w = pw * 2
        grid_h = display_h * 2
        combined = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        combined.paste(p1, (0, 0))
        combined.paste(p2, (pw, 0))
        combined.paste(p3, (0, display_h))
        combined.paste(p4, (pw, display_h))

        combined.save(out_dir / f"compare_{stem}.png")
        extra = f"  rot={rot_dist:.1f}deg" if rot_dist is not None else ""
        print(f"  [{idx+1}/{len(stylized_files)}] compare_{stem}.png{extra}")

        grid_tiles.append(combined)

    # Summary grid: stack all comparisons vertically
    if grid_tiles:
        tile_w = grid_tiles[0].width
        tile_h = grid_tiles[0].height
        header_h = 30
        grid = Image.new("RGB", (tile_w, tile_h * len(grid_tiles) + header_h), (20, 20, 20))
        draw = ImageDraw.Draw(grid)
        draw.text((8, 6),
                  "Green = original labels/pose | Orange = style ref labels/pose | "
                  "Panel 3 = consistency check",
                  fill="white")
        for i, tile in enumerate(grid_tiles):
            grid.paste(tile, (0, i * tile_h + header_h))
        grid.save(out_dir / "summary_grid.png")
        print(f"\nSummary grid: {out_dir / 'summary_grid.png'}")

    print(f"\nDone! Output in {out_dir}/")
    print("How to read:")
    print("  Panel 1: Original image with its GT keypoints + orientation (green)")
    print("  Panel 2: Style reference with its GT keypoints + orientation (orange)")
    print("  Panel 3: KEY CHECK — augmented image with original labels overlaid")
    print("           Keypoints should still land on correct spacecraft features")
    print("  Panel 4: Both label sets on augmented — see pose difference between")
    print("           content and style ref (smaller diff = better transfer)")


if __name__ == "__main__":
    main()
