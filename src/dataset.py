import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SpeedPlusKeypointDataset(Dataset):
    """SPEED+ dataset for keypoint regression from bbox-cropped images.

    Label format (YOLO keypoint):
        class cx cy w h x1 y1 v1 x2 y2 v2 ... x11 y11 v11
    All coordinates are normalized [0,1] relative to the full image.
    Visibility: 0=not labeled, 1=labeled+occluded, 2=labeled+visible.

    Optionally loads pose labels (quaternion + translation) from a JSON file.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        num_keypoints: int = 11,
        bbox_pad_ratio: float = 0.1,
        transform=None,
        pose_json: str | None = None,
        include_list: set[str] | None = None,
        fda_pool=None,
        fda_prob: float = 0.0,
        bg_labels: dict[str, str] | None = None,
        no_crop: bool = False,
        gt_crop: bool = False,
        gt_crop_margin: float = 0.15,
        gt_crop_min_size: float = 64.0,
        resize_first: int = 0,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_keypoints = num_keypoints
        self.bbox_pad_ratio = bbox_pad_ratio
        self.transform = transform
        self.no_crop = no_crop
        self.gt_crop = gt_crop
        self.gt_crop_margin = gt_crop_margin
        self.gt_crop_min_size = gt_crop_min_size
        self.resize_first = resize_first

        # FDA augmentation (optional)
        self.fda_pool = fda_pool
        self.fda_prob = fda_prob
        self.bg_labels = bg_labels or {}

        # Load pose labels if provided
        self.pose_data = {}
        if pose_json is not None:
            with open(pose_json, "r") as f:
                entries = json.load(f)
            for entry in entries:
                self.pose_data[entry["filename"]] = {
                    "q": np.array(entry["q_vbs2tango_true"], dtype=np.float32),
                    "t": np.array(entry["r_Vo2To_vbs_true"], dtype=np.float32),
                }

        # Collect samples: match image <-> label by stem name
        self.samples = []
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if include_list is not None and img_path.name not in include_list:
                continue
            label_path = self.label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def _parse_label(self, label_path: Path):
        """Parse a YOLO keypoint label file.

        Returns:
            bbox: (cx, cy, w, h) normalized
            keypoints: (num_keypoints, 2) normalized (x, y)
            visibility: (num_keypoints,) int
        """
        with open(label_path, "r") as f:
            line = f.readline().strip()
        tokens = line.split()

        # tokens[0] = class, tokens[1:5] = bbox
        bbox = np.array([float(t) for t in tokens[1:5]], dtype=np.float32)

        # Remaining tokens: triplets of (x, y, vis)
        kp_tokens = tokens[5:]
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        visibility = np.zeros(self.num_keypoints, dtype=np.int64)

        for i in range(self.num_keypoints):
            idx = i * 3
            keypoints[i, 0] = float(kp_tokens[idx])
            keypoints[i, 1] = float(kp_tokens[idx + 1])
            visibility[i] = int(kp_tokens[idx + 2])

        return bbox, keypoints, visibility

    def _crop_bbox(self, image: Image.Image, bbox: np.ndarray):
        """Crop image to bbox with padding. Returns crop and pixel-space bbox.

        Args:
            image: PIL Image
            bbox: (cx, cy, w, h) normalized [0,1]

        Returns:
            crop: PIL Image (cropped region)
            crop_box: (x1, y1, x2, y2) in pixel coordinates
        """
        img_w, img_h = image.size
        cx, cy, w, h = bbox

        # Convert to pixel coords
        cx_px = cx * img_w
        cy_px = cy * img_h
        w_px = w * img_w
        h_px = h * img_h

        # Add padding
        pad_w = w_px * self.bbox_pad_ratio
        pad_h = h_px * self.bbox_pad_ratio

        x1 = max(0, int(cx_px - w_px / 2 - pad_w))
        y1 = max(0, int(cy_px - h_px / 2 - pad_h))
        x2 = min(img_w, int(cx_px + w_px / 2 + pad_w))
        y2 = min(img_h, int(cy_px + h_px / 2 + pad_h))

        crop = image.crop((x1, y1, x2, y2))
        crop_box = np.array([x1, y1, x2, y2], dtype=np.float32)

        return crop, crop_box

    def _remap_keypoints(
        self, keypoints: np.ndarray, crop_box: np.ndarray, img_w: int, img_h: int
    ):
        """Remap keypoints from full-image-relative to crop-relative [0,1].

        Args:
            keypoints: (N, 2) normalized coords in full image
            crop_box: (x1, y1, x2, y2) pixel coords of the crop
            img_w, img_h: original image dimensions

        Returns:
            (N, 2) keypoints in crop-relative [0,1] coords
        """
        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Convert from normalized full-image to pixel
        kp_px = keypoints.copy()
        kp_px[:, 0] *= img_w
        kp_px[:, 1] *= img_h

        # Remap to crop-relative
        kp_crop = np.zeros_like(kp_px)
        kp_crop[:, 0] = (kp_px[:, 0] - x1) / crop_w
        kp_crop[:, 1] = (kp_px[:, 1] - y1) / crop_h

        return kp_crop

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image (grayscale -> convert to RGB for DINOv3)
        image = Image.open(img_path)
        orig_w, orig_h = image.size
        if image.mode == "L":
            image = image.convert("RGB")

        # Optionally resize to target size before cropping
        # (matching colleague's pipeline: resize 1920x1200 → 512x512, then crop)
        if self.resize_first > 0:
            image = image.resize(
                (self.resize_first, self.resize_first), Image.BILINEAR)

        img_w, img_h = image.size  # working dimensions (may be resized)

        # Parse label
        bbox, keypoints, visibility = self._parse_label(label_path)

        # Crop to bbox (or use full image for models trained on full images)
        if self.no_crop:
            crop = image
            crop_box = np.array([0, 0, img_w, img_h], dtype=np.float32)
            kp_crop = keypoints  # already [0,1] in full-image space
        elif self.gt_crop:
            # Crop around GT keypoints (matching colleague's pipeline)
            kp_px = keypoints.copy()
            kp_px[:, 0] *= img_w
            kp_px[:, 1] *= img_h
            valid = visibility > 0
            if valid.sum() >= 2:
                xs, ys = kp_px[valid, 0], kp_px[valid, 1]
                cx = (xs.min() + xs.max()) / 2
                cy = (ys.min() + ys.max()) / 2
                bw = max(float(xs.max() - xs.min()), 1.0) * (1 + self.gt_crop_margin)
                bh = max(float(ys.max() - ys.min()), 1.0) * (1 + self.gt_crop_margin)
                side = max(bw, bh, self.gt_crop_min_size)
                x1 = max(0, int(cx - side / 2))
                y1 = max(0, int(cy - side / 2))
                x2 = min(img_w, int(cx + side / 2))
                y2 = min(img_h, int(cy + side / 2))
                crop = image.crop((x1, y1, x2, y2))
                crop_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                kp_crop = self._remap_keypoints(keypoints, crop_box, img_w, img_h)
            else:
                # Fallback to full image if too few visible keypoints
                crop = image
                crop_box = np.array([0, 0, img_w, img_h], dtype=np.float32)
                kp_crop = keypoints
        else:
            crop, crop_box = self._crop_bbox(image, bbox)
            kp_crop = self._remap_keypoints(keypoints, crop_box, img_w, img_h)

        # FDA style transfer (before other augmentations)
        if self.fda_pool is not None and random.random() < self.fda_prob:
            bg_type = self.bg_labels.get(img_path.name, "black")
            crop = self.fda_pool.apply(crop, bg_type)

        # Apply transforms (resize, normalize, augmentation)
        if self.transform is not None:
            crop, kp_crop, visibility = self.transform(crop, kp_crop, visibility)

        sample = {
            "image": crop,
            "keypoints": torch.as_tensor(kp_crop, dtype=torch.float32),
            "visibility": torch.as_tensor(visibility, dtype=torch.long),
            "crop_box": torch.as_tensor(crop_box, dtype=torch.float32),
            "img_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
        }

        # Add pose labels if available
        filename = img_path.name
        if filename in self.pose_data:
            pose = self.pose_data[filename]
            sample["quaternion"] = torch.as_tensor(pose["q"], dtype=torch.float32)
            sample["translation"] = torch.as_tensor(pose["t"], dtype=torch.float32)
        else:
            # Provide zeros so batching works; trainer masks by has_pose
            sample["quaternion"] = torch.zeros(4, dtype=torch.float32)
            sample["translation"] = torch.zeros(3, dtype=torch.float32)

        sample["has_pose"] = torch.tensor(filename in self.pose_data, dtype=torch.bool)
        sample["filename"] = filename

        return sample
