# WheatScopeNet

**A lightweight organ-level semantic segmentation network for field wheat canopy RGB images.**

[![Paper](https://img.shields.io/badge/ISPRS%20J.%20Photogramm.%20Remote%20Sens.-10.1016%2Fj.isprsjprs.2026.08.039-blue)](https://doi.org/10.1016/j.isprsjprs.2026.08.039)

WheatScopeNet segments near-ground RGB images of wheat plots into three classes — `background`,
`spike` and `leaf`. It combines **parallel hybrid spatial modeling** (local multi-scale depthwise
convolutions running alongside a 2D selective state-space operator) with **cross-scale feature
fusion**, so that thin, heavily occluded and visually similar canopy organs stay separable while
the network remains small enough to train and deploy on a single GPU.

The architecture is a five-stage encoder–decoder with a cross-scale bridge:

* **Encoder** — two plain convolution stages capture low-level detail, then three PHS-Block stages
  model multi-scale spatial detail and long-range dependencies in parallel.
* **Bottleneck** — PHS Blocks at the deepest resolution enlarge the effective receptive field.
* **Cross-scale bridge** — SCAB applies spatial and channel attention to the five skip features,
  then LightCSF compresses, fuses and redistributes them across scales.
* **Decoder** — bilinear upsampling with SE-recalibrated skip connections, followed by a 1x1
  convolution producing the per-pixel class logits.

---

## Contents

1. [Installation](#1-installation)
2. [Pretrained weights](#2-pretrained-weights)
3. [Inference](#3-inference)
4. [Training on your own data](#4-training-on-your-own-data)
5. [Repository layout](#5-repository-layout)
6. [Citation](#6-citation)
7. [License and acknowledgements](#7-license-and-acknowledgements)

---

## 1. Installation

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
git clone <this-repository> WheatScopeNet
cd WheatScopeNet

conda create -n wheatscopenet python=3.10 -y
conda activate wheatscopenet

# Install PyTorch matching your CUDA toolkit first, e.g. for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Optional but recommended: the fused CUDA selective-scan kernel

```bash
pip install causal-conv1d
pip install mamba-ssm
```

`mamba-ssm` provides the fused CUDA kernel behind the selective state-space operator. It is
**optional**: if the import fails, the model automatically falls back to a pure-PyTorch
implementation that produces the same results and also runs on CPU.

> The fallback evaluates the state-space recurrence as a Python loop over every scan position, so
> its cost grows linearly with image size. That is fine for inference on small images and for quick
> checks, but full-resolution training without `mamba-ssm` is impractically slow. Install the
> kernel before training.

Verify the installation:

```bash
python -c "from configs.wheatscopenet import Config; \
from wheatscopenet.models import build_wheatscopenet; \
print(type(build_wheatscopenet(Config())).__name__, 'ready')"
```

---

## 2. Pretrained weights

 Download the checkpoint from Google Drive:

**https://drive.google.com/file/d/10JGo8v9BQsEw9BeU0fsE0uDcQ2n8CvjF/view?usp=sharing**

Place it in a `checkpoints/` directory at the repository root:

```bash
mkdir -p checkpoints
# move the downloaded file into checkpoints/
```

```
WheatScopeNet/
└── checkpoints/
    └── organ.pth
```

The checkpoint stores its training configuration alongside the weights, so `tools/predict.py`
rebuilds the matching architecture automatically — no architecture flags are needed.

---

## 3. Inference

`tools/predict.py` runs a checkpoint over a single image or a whole folder and writes one
segmentation mask per input image.

```bash
# a single image
python tools/predict.py \
    --checkpoint checkpoints/organ.pth \
    --input path/to/image.jpg \
    --output predictions

# a whole folder, on the GPU
python tools/predict.py \
    --checkpoint checkpoints/organ.pth \
    --input path/to/images/ \
    --output predictions \
    --batch-size 4 \
    --device cuda

# also write colour overlays of the mask on top of the input image
python tools/predict.py \
    --checkpoint checkpoints/organ.pth \
    --input path/to/images/ \
    --output predictions \
    --overlay

# CPU-only run
python tools/predict.py \
    --checkpoint checkpoints/organ.pth \
    --input path/to/image.jpg \
    --output predictions \
    --device cpu
```

### Output

```
predictions/
├── masks/          # one single-channel PNG per input image
└── overlays/       # only written with --overlay
```

Each mask is a single-channel PNG holding the class index of every pixel, at the **original**
resolution of the input image:

| Pixel value | Class |
|---|---|
| `0` | background |
| `1` | spike |
| `2` | leaf |

Because the stored values are class indices (0, 1, 2), the masks look almost black in an image
viewer. Use `--overlay` for a human-readable visualisation, or read them programmatically:

```python
import numpy as np
from PIL import Image

mask = np.array(Image.open("predictions/masks/image.png"))
spike_ratio = (mask == 1).mean()
leaf_ratio = (mask == 2).mean()
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to the `.pth` checkpoint |
| `--input` | *(required)* | An image file or a directory of images |
| `--output` | `predictions` | Output directory |
| `--batch-size` | `1` | Images processed per forward pass |
| `--input-size` | `2048` | Square resolution the model runs at |
| `--overlay` | off | Also write colour overlays |
| `--overlay-alpha` | `0.5` | Blending weight of the overlay colour |
| `--device` | auto | `cuda`, `cuda:0`, `cpu`, … |
| `--no-amp` | off | Disable mixed precision on CUDA |

---

## 4. Training on your own data

### Data layout

Images and masks are matched by file basename:

```
<data_root>/
├── images/
│   ├── train/   *.jpg
│   ├── val/     *.jpg
│   └── test/    *.jpg
└── masks/
    ├── train/   *.png
    ├── val/     *.png
    └── test/    *.png
```

Masks are single-channel `uint8` PNGs using the class indices above (`0` background, `1` spike,
`2` leaf). Pixels valued `255` are ignored by both the loss and the metrics.

Images are resized to the working resolution with bilinear interpolation and masks with
nearest-neighbour interpolation, so class labels are never interpolated. When several images come
from repeated observations of the same plot, assign all images of a plot to a single split so that
temporally adjacent images cannot leak between training and evaluation.

### Run training

```bash
python tools/train.py --data-root /path/to/dataset

# common overrides
python tools/train.py \
    --data-root /path/to/dataset \
    --work-dir results/my_run \
    --epochs 300 \
    --batch-size 2 \
    --lr 1e-3 \
    --device cuda

# resume an interrupted run
python tools/train.py \
    --data-root /path/to/dataset \
    --work-dir results/my_run \
    --resume results/my_run/checkpoints/latest.pth
```

Training writes into the run directory:

```
results/<run>/
├── checkpoints/
│   ├── best.pth        # best validation score
│   └── latest.pth      # last finished epoch
├── config.json         # resolved configuration
├── metrics_history.json / .csv
└── train.info.log
```

`best.pth` can be passed straight to `tools/predict.py`.

Defaults live in `configs/wheatscopenet.py` and can be edited there or overridden on the command
line.

---

## 5. Repository layout

```
WheatScopeNet/
├── configs/
│   └── wheatscopenet.py          # model / data / training configuration
├── wheatscopenet/
│   ├── models/
│   │   ├── layers.py             # shared building blocks
│   │   ├── ss2d.py               # 2D selective scan operator
│   │   ├── phs_block.py          # parallel hybrid spatial block
│   │   ├── bridge.py             # SCAB and LightCSF cross-scale bridge
│   │   └── wheatscopenet.py      # the network
│   ├── data/
│   │   └── wheat_canopy.py       # dataset and augmentation
│   ├── losses.py                 # cross-entropy + Dice loss
│   ├── metrics.py                # IoU / Dice / precision / recall / accuracy
│   ├── engine.py                 # train and evaluation loops
│   └── utils.py                  # seeding, logging, optimizer, scheduler
├── tools/
│   ├── train.py                  # training entry point
│   └── predict.py                # inference entry point
├── requirements.txt
└── LICENSE
```

The dataset is not included in this repository.

---

## 6. Citation

If you use this code, please cite:

```bibtex
@article{deng2026wheatscoper,
  title   = {WheatScoper: A lightweight organ-based framework for multi-view wheat
             phenotyping using time-series RGB images},
  author  = {Deng, Haotian and Tian, Xiaomiao and Zhang, Yufeng and Ren, Yong and
             Yuan, Guoqiang and Qin, Bingxi and Zhou, Honghao and Chen, Jiawei and
             Wang, Xiao and Zhou, Qin and Cai, Jian and Zhong, Yingxin and
             Huang, Mei and Sun, Qixin and Jiang, Dong and Yao, Yingyin and Li, Qing},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.08.039},
  url     = {https://doi.org/10.1016/j.isprsjprs.2026.08.039}
}
```

---

## 7. License and acknowledgements

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This work is licensed under the Creative Commons
**Attribution-NonCommercial-NoDerivatives 4.0 International** licence
(CC BY-NC-ND 4.0) — see [LICENSE](LICENSE).

You may share the material for non-commercial purposes with attribution.
Commercial use, and distribution of modified versions, require the authors' permission.

The selective state-space operator follows the SS2D design of
[VMamba](https://github.com/MzeroMiko/VMamba) (Y. Liu et al., 2024).
