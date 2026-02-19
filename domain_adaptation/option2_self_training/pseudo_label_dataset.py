"""Dataset for real images with model-generated pseudo-labels.

Used in self-training: the model's own predictions on unlabeled real images
are used as training targets with confidence-based weighting.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PseudoLabelDataset(Dataset):
    """Dataset that loads real images with pseudo-labels (predicted keypoints).

    Each sample contains:
        - image: cropped satellite image
        - keypoints: predicted keypoint coords in crop-relative [0,1]
        - visibility: all set to 2 (visible) since we predicted them
        - confidence: per-keypoint confidence from heatmap peaks
        - pseudo_weight: sample weight based on mean confidence

    Args:
        image_dir: path to real images (e.g., sunlamp)
        label_dir: path to YOLO labels (for bbox cropping only)
        pseudo_labels: dict {filename: {"keypoints": (K,2), "confidence": (K,)}}
        confidence_threshold: min mean confidence to include sample
        num_keypoints: number of keypoints
        bbox_pad_ratio: bbox padding ratio
        transform: KeypointTransform instance
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        pseudo_labels: dict,
        confidence_threshold: float = 0.3,
        num_keypoints: int = 11,
        bbox_pad_ratio: float = 0.1,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_keypoints = num_keypoints
        self.bbox_pad_ratio = bbox_pad_ratio
        self.transform = transform

        # Filter by confidence
        self.samples = []
        n_filtered = 0
        for fname, pl in pseudo_labels.items():
            mean_conf = np.mean(pl["confidence"])
            if mean_conf < confidence_threshold:
                n_filtered += 1
                continue

            img_path = self.image_dir / fname
            label_path = self.label_dir / (Path(fname).stem + ".txt")
            if img_path.exists() and label_path.exists():
                self.samples.append({
                    "image_path": img_path,
                    "label_path": label_path,
                    "keypoints": pl["keypoints"],     # (K, 2) crop-relative
                    "confidence": pl["confidence"],    # (K,)
                    "weight": float(mean_conf),
                })

        print(f"PseudoLabelDataset: {len(self.samples)} samples "
              f"({n_filtered} filtered by confidence < {confidence_threshold})")

    def __len__(self):
        return len(self.samples)

    def _parse_bbox(self, label_path):
        """Parse just the bbox from a YOLO label file."""
        with open(label_path) as f:
            line = f.readline().strip()
        tokens = line.split()
        return np.array([float(t) for t in tokens[1:5]], dtype=np.float32)

    def _crop_bbox(self, image, bbox):
        """Crop image to bbox with padding."""
        img_w, img_h = image.size
        cx, cy, w, h = bbox
        cx_px, cy_px = cx * img_w, cy * img_h
        w_px, h_px = w * img_w, h * img_h
        pad_w = w_px * self.bbox_pad_ratio
        pad_h = h_px * self.bbox_pad_ratio

        x1 = max(0, int(cx_px - w_px / 2 - pad_w))
        y1 = max(0, int(cy_px - h_px / 2 - pad_h))
        x2 = min(img_w, int(cx_px + w_px / 2 + pad_w))
        y2 = min(img_h, int(cy_px + h_px / 2 + pad_h))

        crop = image.crop((x1, y1, x2, y2))
        crop_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        return crop, crop_box

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = Image.open(s["image_path"])
        img_w, img_h = image.size
        if image.mode == "L":
            image = image.convert("RGB")

        bbox = self._parse_bbox(s["label_path"])
        crop, crop_box = self._crop_bbox(image, bbox)

        # Use pseudo-label keypoints (already in crop-relative [0,1])
        kp_crop = s["keypoints"].copy()
        visibility = np.full(self.num_keypoints, 2, dtype=np.int64)  # all "visible"

        if self.transform is not None:
            crop, kp_crop, visibility = self.transform(crop, kp_crop, visibility)

        sample = {
            "image": crop,
            "keypoints": torch.as_tensor(kp_crop, dtype=torch.float32),
            "visibility": torch.as_tensor(visibility, dtype=torch.long),
            "crop_box": torch.as_tensor(crop_box, dtype=torch.float32),
            "img_size": torch.tensor([img_w, img_h], dtype=torch.float32),
            "pseudo_weight": torch.tensor(s["weight"], dtype=torch.float32),
            "confidence": torch.as_tensor(s["confidence"], dtype=torch.float32),
            # No pose labels for pseudo-labeled data
            "quaternion": torch.zeros(4, dtype=torch.float32),
            "translation": torch.zeros(3, dtype=torch.float32),
            "has_pose": torch.tensor(False, dtype=torch.bool),
        }
        return sample
