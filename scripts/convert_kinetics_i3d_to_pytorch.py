#!/usr/bin/env python3
"""
Convert kinetics-i3d TensorFlow checkpoints to PyTorch format for MC-TTA.

This script converts the official DeepMind kinetics-i3d checkpoints
(rgb_imagenet, flow_imagenet) to PyTorch .pt files compatible with
MC-TTA's InceptionI3d backbone.

Requirements:
  - TensorFlow (1.x or 2.x)
  - PyTorch
  - kinetics-i3d repo with checkpoints in data/checkpoints/

Usage:
  python scripts/convert_kinetics_i3d_to_pytorch.py \\
      --kinetics_i3d_path /path/to/kinetics-i3d \\
      --output_dir pretrained

Alternative: If you have pre-converted weights from hassony2/kinetics_i3d_pytorch,
use --hassony2_rgb and --hassony2_flow to remap keys to MC-TTA format.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mapping from hassony2 I3D state_dict keys to MC-TTA I3D keys
# hassony2 uses lowercase (conv3d_1a_7x7, mixed_3b) and batch3d
# MC-TTA uses Mixed case (Conv3d_1a_7x7, Mixed_3b) and bn
HASSONY2_TO_MCTTA = {
    'conv3d_1a_7x7': 'Conv3d_1a_7x7',
    'conv3d_2b_1x1': 'Conv3d_2b_1x1',
    'conv3d_2c_3x3': 'Conv3d_2c_3x3',
    'mixed_3b': 'Mixed_3b',
    'mixed_3c': 'Mixed_3c',
    'mixed_4b': 'Mixed_4b',
    'mixed_4c': 'Mixed_4c',
    'mixed_4d': 'Mixed_4d',
    'mixed_4e': 'Mixed_4e',
    'mixed_4f': 'Mixed_4f',
    'mixed_5b': 'Mixed_5b',
    'mixed_5c': 'Mixed_5c',
}

# Inception branch mapping: hassony2 branch_0 -> MC-TTA b0, etc.
BRANCH_MAP = {
    'branch_0': 'b0',
    'branch_1.0': 'b1a',
    'branch_1.1': 'b1b',
    'branch_2.0': 'b2a',
    'branch_2.1': 'b2b',
    'branch_3.1': 'b3b',  # branch_3.0 is MaxPool3d (no params)
}


def remap_hassony2_to_mctta(state_dict):
    """
    Remap hassony2/kinetics_i3d_pytorch state_dict to MC-TTA format.
    MC-TTA I3D stops at Mixed_5c (no Logits), so we drop conv3d_0c_1x1.
    """
    new_state = {}
    for key, value in state_dict.items():
        if 'conv3d_0c_1x1' in key:  # Logits layer - skip for feature extractor
            continue
        new_key = key
        for old_prefix, new_prefix in HASSONY2_TO_MCTTA.items():
            if new_key.startswith(old_prefix + '.'):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        # batch3d -> bn
        new_key = new_key.replace('.batch3d.', '.bn.')
        # Mixed branch names
        for old_b, new_b in BRANCH_MAP.items():
            new_key = new_key.replace('.' + old_b + '.', '.' + new_b + '.')
        new_state[new_key] = value
    return new_state


def convert_from_hassony2(hassony2_path, output_path):
    """Load hassony2 .pth and save in MC-TTA format."""
    state = torch.load(hassony2_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    new_state = remap_hassony2_to_mctta(state)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(new_state, output_path)
    print(f"Saved {output_path} ({len(new_state)} keys)")
    return new_state


def convert_from_tf_checkpoint(tf_ckpt_path, output_path, modality='rgb'):
    """
    Convert TensorFlow checkpoint to PyTorch using tf.train.load_checkpoint.
    Works with TF 2.x. For TF 1.x, use the alternative reader.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is required for TF checkpoint conversion.")
        print("Install with: pip install tensorflow")
        print("Alternatively, use --hassony2_rgb/--hassony2_flow with pre-converted weights.")
        return None

    # TF 2.x uses tf.train.load_checkpoint
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    var_map = reader.get_variable_to_shape_map()

    prefix = 'RGB' if modality == 'rgb' else 'Flow'
    # kinetics-i3d uses RGB/ or Flow/ scope; Sonnet may add inception_i3d/
    # Try both naming conventions
    def find_var(suffix):
        for name in var_map.keys():
            if prefix in name and suffix in name:
                return name
        return None

    # Build state dict by mapping TF vars to MC-TTA keys
    # This is a simplified mapping - full implementation would need
    # complete TF variable name parsing
    state_dict = {}
    for tf_name, shape in var_map.items():
        if prefix not in tf_name:
            continue
        # Parse TF name and convert to PyTorch tensor
        try:
            var = reader.get_tensor(tf_name)
            arr = np.array(var)
            # TF conv: (kH, kW, kD, in_c, out_c) -> PyTorch (out_c, in_c, kD, kH, kW)
            if 'conv' in tf_name.lower() and 'w' in tf_name.lower():
                if len(arr.shape) == 5:
                    arr = np.transpose(arr, (4, 3, 0, 1, 2))
            tensor = torch.from_numpy(arr).float()
            # Map to MC-TTA key (simplified - may need full mapping)
            pt_key = tf_name.replace(prefix + '/', '').replace(':0', '')
            if 'inception_i3d/' in pt_key:
                pt_key = pt_key.replace('inception_i3d/', '')
            pt_key = pt_key.replace('/', '.')
            pt_key = pt_key.replace('conv_3d.w', 'conv3d.weight')
            pt_key = pt_key.replace('batch_norm.moving_mean', 'bn.running_mean')
            pt_key = pt_key.replace('batch_norm.moving_variance', 'bn.running_var')
            pt_key = pt_key.replace('batch_norm.beta', 'bn.bias')
            state_dict[pt_key] = tensor
        except Exception as e:
            pass  # Skip vars we can't map

    if len(state_dict) < 10:
        print("WARNING: TF conversion produced few keys. TF variable names may differ.")
        print("Consider using hassony2/kinetics_i3d_pytorch conversion instead.")
        return None

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"Saved {output_path} ({len(state_dict)} keys)")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Convert kinetics-i3d to PyTorch for MC-TTA')
    parser.add_argument('--kinetics_i3d_path', type=str,
                        help='Path to kinetics-i3d repo (e.g. ../kinetics-i3d)')
    parser.add_argument('--output_dir', type=str, default='pretrained')
    parser.add_argument('--hassony2_rgb', type=str,
                        help='Path to hassony2 model_rgb.pth (pre-converted)')
    parser.add_argument('--hassony2_flow', type=str,
                        help='Path to hassony2 model_flow.pth (pre-converted)')
    args = parser.parse_args()

    if args.hassony2_rgb or args.hassony2_flow:
        if args.hassony2_rgb:
            convert_from_hassony2(args.hassony2_rgb,
                                  os.path.join(args.output_dir, 'rgb_imagenet.pt'))
        if args.hassony2_flow:
            convert_from_hassony2(args.hassony2_flow,
                                  os.path.join(args.output_dir, 'flow_imagenet.pt'))
        return

    if args.kinetics_i3d_path:
        rgb_base = os.path.join(args.kinetics_i3d_path, 'data', 'checkpoints', 'rgb_imagenet', 'model.ckpt')
        if os.path.exists(rgb_base + '.index') or os.path.exists(rgb_base):
            convert_from_tf_checkpoint(rgb_base, os.path.join(args.output_dir, 'rgb_imagenet.pt'), 'rgb')
        else:
            print(f"RGB checkpoint not found at {rgb_base}")
        flow_base = os.path.join(args.kinetics_i3d_path, 'data', 'checkpoints', 'flow_imagenet', 'model.ckpt')
        if os.path.exists(flow_base + '.index') or os.path.exists(flow_base):
            convert_from_tf_checkpoint(flow_base, os.path.join(args.output_dir, 'flow_imagenet.pt'), 'flow')
        else:
            print(f"Flow checkpoint not found at {flow_base}")
        return

    print("Usage:")
    print("  1. From kinetics-i3d: --kinetics_i3d_path /path/to/kinetics-i3d")
    print("  2. From hassony2:      --hassony2_rgb model_rgb.pth --hassony2_flow model_flow.pth")
    print("\nRecommended: Download pre-converted weights from piergiaj/pytorch-i3d:")
    print("  wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt")
    print("  wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/flow_imagenet.pt")


if __name__ == '__main__':
    main()
