"""Wheat canopy organ-segmentation dataset (paper Section 2.2).

Loads the multi-view wheat canopy RGB images and their organ-level annotation
masks used to train WheatScopeNet.  Every sample is resized to the paper's
2048x2048 working resolution (bilinear for the image, nearest for the mask so
class indices are never interpolated) and, for the training split, passed
through the augmentation pipeline described in Section 2.2.

Class index convention: 0 = background, 1 = spike, 2 = leaf.
"""

from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF

__all__ = ["WheatCanopyDataset"]

# Accepted image file extensions inside <data_root>/images/<split>.
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
# Annotation masks are single-channel PNGs storing the class index per pixel.
MASK_EXTENSION: str = ".png"

# Augmentation probabilities (paper Section 2.2; identical to the legacy loader).
P_HFLIP: float = 0.5
P_VFLIP: float = 0.5
P_ROTATE: float = 0.5
P_SCALE_CROP: float = 0.5
P_COLOR_JITTER: float = 0.5
P_GAUSSIAN_BLUR: float = 0.2

ROTATION_ANGLES: Tuple[int, ...] = (90, 180, 270)
SCALE_CROP_RANGE: Tuple[float, float] = (0.8, 1.0)
BLUR_RADIUS_RANGE: Tuple[float, float] = (0.5, 1.5)

COLOR_JITTER_BRIGHTNESS: float = 0.2
COLOR_JITTER_CONTRAST: float = 0.2
COLOR_JITTER_SATURATION: float = 0.2
COLOR_JITTER_HUE: float = 0.05

# Number of missing masks listed in the error message before it is truncated.
MAX_REPORTED_MISSING: int = 10


class WheatCanopyDataset(Dataset):
    """Organ-level wheat canopy segmentation dataset.

    The only supported on-disk layout is::

        <data_root>/images/<split>/*.jpg
        <data_root>/masks/<split>/*.png

    An image and its mask are paired by file basename (stem).

    Args:
        data_root: Dataset root directory holding ``images/`` and ``masks/``.
        split: One of ``"train"``, ``"val"`` or ``"test"`` (the sub-directory name).
        input_size: ``(height, width)`` the image and mask are resized to
            (paper Section 2.2 uses 2048x2048).
        augment: Enable the training augmentation pipeline. ``None`` (default)
            means "augment only when ``split == 'train'``".

    Returns per item:
        ``(image, mask)`` where ``image`` is a float32 tensor ``[3, H, W]``
        scaled to ``[0, 1]`` and ``mask`` is an int64 tensor ``[H, W]`` of class
        indices.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        input_size: Tuple[int, int] = (2048, 2048),
        augment: bool | None = None,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.augment = (split == "train") if augment is None else bool(augment)

        self.image_dir = os.path.join(data_root, "images", split)
        self.mask_dir = os.path.join(data_root, "masks", split)

        missing = [d for d in (self.image_dir, self.mask_dir) if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(
                "WheatCanopyDataset expects the layout "
                "'<data_root>/images/<split>' and '<data_root>/masks/<split>'. "
                "Missing directory(ies): " + ", ".join(missing)
            )

        self.image_files: List[str] = self._list_images(self.image_dir)
        if not self.image_files:
            raise FileNotFoundError(
                f"No image files with extensions {IMAGE_EXTENSIONS} found in '{self.image_dir}'."
            )

        self.mask_files: List[str] = self._match_masks(self.image_files)

        self.color_jitter = ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE,
        )

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _list_images(image_dir: str) -> List[str]:
        """Return the sorted image file names present in ``image_dir``."""
        names = [
            name
            for name in os.listdir(image_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        return sorted(names)

    def _match_masks(self, image_files: Sequence[str]) -> List[str]:
        """Pair every image with ``<stem>.png`` in the mask directory."""
        mask_files: List[str] = []
        unmatched: List[str] = []
        for name in image_files:
            mask_name = os.path.splitext(name)[0] + MASK_EXTENSION
            if os.path.isfile(os.path.join(self.mask_dir, mask_name)):
                mask_files.append(mask_name)
            else:
                unmatched.append(mask_name)

        if unmatched:
            listed = ", ".join(unmatched[:MAX_REPORTED_MISSING])
            suffix = "" if len(unmatched) <= MAX_REPORTED_MISSING else f", ... (+{len(unmatched) - MAX_REPORTED_MISSING} more)"
            raise FileNotFoundError(
                f"{len(unmatched)} image(s) in '{self.image_dir}' have no matching mask in "
                f"'{self.mask_dir}'. Every image '<stem>.<ext>' needs a mask '<stem>{MASK_EXTENSION}'. "
                f"Missing: {listed}{suffix}"
            )
        return mask_files

    # -------------------------------------------------------------- pipeline

    def _resize(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Resize to ``input_size``: bilinear for RGB, nearest for class indices."""
        height, width = self.input_size
        image = TF.resize(image, (height, width), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (height, width), interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Training augmentation of paper Section 2.2 (geometry then photometry)."""
        height, width = self.input_size

        if random.random() < P_HFLIP:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < P_VFLIP:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < P_ROTATE:
            angle = random.choice(ROTATION_ANGLES)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < P_SCALE_CROP:
            scale = random.uniform(*SCALE_CROP_RANGE)
            new_h, new_w = int(height * scale), int(width * scale)
            top = random.randint(0, height - new_h)
            left = random.randint(0, width - new_w)
            image = TF.crop(image, top, left, new_h, new_w)
            mask = TF.crop(mask, top, left, new_h, new_w)
            image, mask = self._resize(image, mask)

        if random.random() < P_COLOR_JITTER:
            # Jitter through a uint8 tensor: torchvision's PIL hue implementation
            # overflows for negative hue factors under NumPy >= 2, while the
            # tensor path is pure PyTorch and numerically equivalent.
            image = TF.to_pil_image(self.color_jitter(TF.pil_to_tensor(image)))

        if random.random() < P_GAUSSIAN_BLUR:
            radius = random.uniform(*BLUR_RADIUS_RANGE)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image, mask

    # --------------------------------------------------------------- dataset

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask).astype(np.int64)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous().float()
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor, mask_tensor
