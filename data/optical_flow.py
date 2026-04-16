"""
Optical flow computation using OpenCV (Farneback or TV-L1).

Produces 2-channel (u, v) flow fields for each consecutive frame pair.
"""

import cv2
import numpy as np
import torch


def compute_optical_flow(frames, method='tvl1'):
    """
    Compute dense optical flow for a sequence of frames.

    Args:
        frames: list of numpy arrays (H, W, 3) in BGR or RGB, uint8.
        method: 'farneback' or 'tvl1'.

    Returns:
        flows: numpy array (T-1, H, W, 2) — u,v components normalized to [-1, 1].
    """
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                   for f in frames]
    flows = []

    if method == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create() if hasattr(cv2, 'optflow') \
            else cv2.optflow.createOptFlow_DualTVL1() if hasattr(cv2, 'optflow') \
            else None
    else:
        tvl1 = None

    for i in range(len(gray_frames) - 1):
        prev, curr = gray_frames[i], gray_frames[i + 1]

        if tvl1 is not None and method == 'tvl1':
            flow = tvl1.calc(prev, curr, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Normalize to [-1, 1] using a fixed bound
        flow = np.clip(flow / 20.0, -1.0, 1.0)
        flows.append(flow)

    if len(flows) == 0:
        H, W = gray_frames[0].shape[:2]
        return np.zeros((1, H, W, 2), dtype=np.float32)

    return np.stack(flows, axis=0).astype(np.float32)


def flow_frames_to_tensor(flow_frames):
    """
    Convert flow frames to PyTorch tensor in (C, T, H, W) format.

    Args:
        flow_frames: (T, H, W, 2) numpy array.

    Returns:
        tensor: (2, T, H, W) float32 tensor.
    """
    tensor = torch.from_numpy(flow_frames).permute(3, 0, 1, 2).float()
    return tensor
