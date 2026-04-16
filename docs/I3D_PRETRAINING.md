# I3D Pretraining for MC-TTA

This document explains how to use Kinetics-pretrained I3D weights with MC-TTA, including options for using the local `kinetics-i3d` repository.

---

## Overview

The MC-TTA paper uses I3D backbones "initialized with the ImageNet dataset and the Kinetics dataset pre-trained models." The paper reports:
- **RGB-I3D**: 71.1% top-1 on Kinetics (ImageNet + Kinetics)
- **Flow-I3D**: 63.4% top-1 on Kinetics (ImageNet + Kinetics)

---

## Option 1: Pre-converted PyTorch Weights (Recommended)

The **piergiaj/pytorch-i3d** repository provides pre-converted PyTorch weights that work with MC-TTA's I3D architecture.

### Download

```bash
mkdir -p pretrained
wget -P pretrained/ https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt
wget -P pretrained/ https://github.com/piergiaj/pytorch-i3d/raw/master/models/flow_imagenet.pt
```

### Usage

```bash
python scripts/pretrain.py \
    --config config/default.yaml \
    --source_domain ucf \
    --data_root ./data/ucf_hmdb_full \
    --num_classes 12 \
    --i3d_rgb_pretrained pretrained/rgb_imagenet.pt \
    --i3d_flow_pretrained pretrained/flow_imagenet.pt
```

Weights are loaded via `models.weights.load_i3d_weights` (strict=False, supports nested `state_dict`).

### Note

- These are the same weights as the official `kinetics-i3d` TensorFlow checkpoints, converted to PyTorch.
- If `wget` fails (e.g. GitHub raw URLs), use a browser or `curl` to download from the [pytorch-i3d models directory](https://github.com/piergiaj/pytorch-i3d/tree/master/models).

---

## Option 2: Convert from Local kinetics-i3d (TensorFlow)

If you have the official **kinetics-i3d** repository locally (e.g. at `../kinetics-i3d`), you can convert its TensorFlow checkpoints to PyTorch.

### Prerequisites

- kinetics-i3d repo with checkpoints in `data/checkpoints/`:
  - `rgb_imagenet/` — ImageNet + Kinetics (recommended)
  - `flow_imagenet/` — ImageNet + Kinetics (recommended)

### Method A: hassony2 conversion + key remapping (recommended)

1. Clone [hassony2/kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)
2. Copy kinetics-i3d checkpoints into hassony2's expected paths
3. Run `python i3d_tf_to_pt.py --rgb --flow` (requires TensorFlow 1.x + Sonnet)
4. Remap the output to MC-TTA format:

```bash
cd /path/to/mc_tta
python scripts/convert_kinetics_i3d_to_pytorch.py \
    --hassony2_rgb /path/to/kinetics_i3d_pytorch/model/model_rgb.pth \
    --hassony2_flow /path/to/kinetics_i3d_pytorch/model/model_flow.pth \
    --output_dir pretrained
```

### Method B: Direct TF conversion

```bash
python scripts/convert_kinetics_i3d_to_pytorch.py \
    --kinetics_i3d_path /path/to/kinetics-i3d \
    --output_dir pretrained
```

Requires TensorFlow. The direct TF→PyTorch mapping may be incomplete; Method A is more reliable.

---

## Option 3: hassony2/kinetics_i3d_pytorch

The [hassony2/kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch) repo has a conversion script that converts TF → PyTorch.

### Steps

1. Clone the repo:
   ```bash
   git clone https://github.com/hassony2/kinetics_i3d_pytorch.git
   cd kinetics_i3d_pytorch
   ```

2. Copy or symlink kinetics-i3d checkpoints:
   ```bash
   mkdir -p model/tf_rgb_imagenet model/tf_flow_imagenet
   cp /path/to/kinetics-i3d/data/checkpoints/rgb_imagenet/* model/tf_rgb_imagenet/
   cp /path/to/kinetics-i3d/data/checkpoints/flow_imagenet/* model/tf_flow_imagenet/
   ```

3. Run conversion (requires TensorFlow 1.x + Sonnet):
   ```bash
   python i3d_tf_to_pt.py --rgb --flow
   ```

4. **Important**: The output format (`model_rgb.pth`, `model_flow.pth`) uses hassony2's I3D structure, which differs from MC-TTA's. You may need a key-remapping step to load into MC-TTA (e.g. `conv3d_1a_7x7` → `Conv3d_1a_7x7`, `batch3d` → `bn`).

---

## Summary

| Option | Effort | Compatibility |
|--------|--------|----------------|
| **1. piergiaj/pytorch-i3d** | Low (download) | Direct |
| **2. Custom conversion script** | Medium | Direct |
| **3. hassony2 conversion** | Medium | Requires key mapping |

**Recommendation**: Use Option 1 (piergiaj weights) if possible. Use Option 2 if you need to convert from your local kinetics-i3d checkpoints, or if Option 1 downloads fail.
