#!/usr/bin/env python3
"""Inference driver for WheatScopeNet.

Runs a trained checkpoint over one image or a folder of images and writes one
organ-level segmentation mask per input.

Run from the repository root, e.g.::

    python tools/predict.py --checkpoint checkpoints/organ.pth --input images/
    python tools/predict.py --checkpoint checkpoints/organ.pth --input a.jpg --overlay

Preprocessing mirrors ``WheatCanopyDataset``: the RGB image is resized to the
working resolution with bilinear interpolation and scaled to [0, 1]. The logits
are resized back to the original image resolution before ``argmax``, so the
stored class indices are never interpolated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Make "import wheatscopenet" and "from configs... import Config" resolve when
# this file is executed as "python tools/predict.py" from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.wheatscopenet import Config  # noqa: E402
from wheatscopenet.models import build_wheatscopenet  # noqa: E402
from wheatscopenet.utils import count_parameters  # noqa: E402

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Visualisation palette (0 = background, 1 = spike, 2 = leaf).
DEFAULT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),        # background
    (255, 193, 7),    # spike
    (76, 175, 80),    # leaf
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WheatScopeNet inference on wheat canopy RGB images.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained WheatScopeNet checkpoint (.pth).")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image file or a directory of images.")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Directory the predicted masks are written to.")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes. Default: taken from the checkpoint config, else 3.")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Square inference resolution. Default: 2048.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--overlay", action="store_true",
                        help="Also write a colour overlay of the prediction on the input image.")
    parser.add_argument("--overlay-alpha", type=float, default=0.5,
                        help="Blending weight of the colour mask in the overlay.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision on CUDA.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    """Return the requested device, defaulting to CUDA when it is available."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_images(path: str) -> List[Path]:
    """Return the sorted list of image files addressed by ``path``."""
    target = Path(path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Input path does not exist: '{path}'")
    files = sorted(p for p in target.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {IMAGE_EXTENSIONS} found in '{path}'."
        )
    return files


def load_model(checkpoint_path: str, device: torch.device,
               num_classes: int | None) -> Tuple[torch.nn.Module, int]:
    """Build WheatScopeNet from the checkpoint's own configuration and load its weights.

    Accepts both a bare ``state_dict`` and the training checkpoint written by
    ``tools/train.py`` (``{"state_dict": ..., "config": ...}``).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    saved_config: dict = {}
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if isinstance(checkpoint.get("config"), dict):
            saved_config = checkpoint["config"]

    # Prefer the configuration the checkpoint was trained with, so the
    # architecture always matches the weights.
    cfg = Config()
    for name, value in saved_config.items():
        if hasattr(cfg, name) and not callable(getattr(cfg, name)):
            setattr(cfg, name, value)
    if num_classes is not None:
        cfg.num_classes = num_classes

    model = build_wheatscopenet(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not match the model.\n"
            f"  missing keys:    {list(missing)[:8]}\n"
            f"  unexpected keys: {list(unexpected)[:8]}"
        )
    return model.to(device).eval(), int(cfg.num_classes)


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    """Resize to ``size`` x ``size`` (bilinear) and scale to [0, 1] -> [3, H, W]."""
    resized = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def colourise(labels: np.ndarray, palette: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Map a [H, W] label map to an [H, W, 3] uint8 RGB image."""
    lut = np.zeros((max(len(palette), int(labels.max()) + 1), 3), dtype=np.uint8)
    for index, colour in enumerate(palette):
        lut[index] = colour
    return lut[labels]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    size = args.input_size or Config.input_size[0]
    use_amp = device.type == "cuda" and not args.no_amp

    files = collect_images(args.input)
    model, num_classes = load_model(args.checkpoint, device, args.num_classes)
    n_params = count_parameters(model)
    print(f"WheatScopeNet | {n_params:,} parameters ({n_params / 1e6:.3f} M) | "
          f"{num_classes} classes | device: {device} | resolution: {size}x{size}")

    output_dir = Path(args.output)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    if args.overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, len(files), args.batch_size), desc="Predicting", ncols=100):
        chunk = files[start:start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in chunk]
        batch = torch.stack([preprocess(image, size) for image in images]).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(batch)
        logits = logits.float()

        for i, (path, image) in enumerate(zip(chunk, images)):
            # Upsample the logits to the original resolution before argmax, then
            # store the label map as a single-channel PNG of class indices.
            width, height = image.size
            resized = F.interpolate(logits[i: i + 1], size=(height, width),
                                    mode="bilinear", align_corners=False)
            labels = resized.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

            stem = path.stem
            Image.fromarray(labels, mode="L").save(mask_dir / f"{stem}.png")

            if args.overlay:
                colour = colourise(labels, DEFAULT_PALETTE).astype(np.float32)
                base = np.asarray(image, dtype=np.float32)
                alpha = float(args.overlay_alpha)
                blended = (1.0 - alpha) * base + alpha * colour
                Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(
                    overlay_dir / f"{stem}.png"
                )

    print(f"Wrote {len(files)} mask(s) to {mask_dir}"
          + (f" and overlay(s) to {overlay_dir}" if args.overlay else ""))


if __name__ == "__main__":
    main()
