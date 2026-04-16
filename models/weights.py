"""
Load I3D backbone checkpoints (Kinetics / ImageNet / raw state_dict).

Supports:
  - Plain state_dict (.pt)
  - dict with 'state_dict' key
  - Partial key match via strict=False
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import torch
import torch.nn as nn


def load_i3d_weights(module: nn.Module, path: str, device: torch.device | str = 'cpu') -> Tuple[list, list]:
    """
    Load weights into an InceptionI3d (or compatible) module.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    ckpt: Any = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('Conv3d_') or k.startswith('Mixed_') for k in ckpt):
        state = ckpt
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f'Expected state dict in {path}, got {type(state)}')

    return module.load_state_dict(state, strict=False)
