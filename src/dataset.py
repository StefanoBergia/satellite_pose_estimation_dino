import json
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
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_keypoints = num_keypoints
        self.bbox_pad_ratio = bbox_pad_ratio
        self.transform = transform

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
        img_w, img_h = image.size
        if image.mode == "L":
            image = image.convert("RGB")

        # Parse label
        bbox, keypoints, visibility = self._parse_label(label_path)

        # Crop to bbox
        crop, crop_box = self._crop_bbox(image, bbox)

        # Remap keypoints to crop-relative coordinates
        kp_crop = self._remap_keypoints(keypoints, crop_box, img_w, img_h)

        # Apply transforms (resize, normalize, augmentation)
        if self.transform is not None:
            crop, kp_crop, visibility = self.transform(crop, kp_crop, visibility)

        sample = {
            "image": crop,
            "keypoints": torch.as_tensor(kp_crop, dtype=torch.float32),
            "visibility": torch.as_tensor(visibility, dtype=torch.long),
            "crop_box": torch.as_tensor(crop_box, dtype=torch.float32),
            "img_size": torch.tensor([img_w, img_h], dtype=torch.float32),
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

        return sample
