"""
Video dataset loaders for MC-TTA.

Supports:
  - UCF-HMDB_small, UCF-HMDB_full, UCF-Olympic, Epic-Kitchens
  - Pre-computed optical flow or on-the-fly computation
  - Segmenting each video into T clips of D frames each

Expected directory structure (example for UCF-HMDB_full):
  data_root/
    UCF-HMDB_full/
      ucf/
        train/
          <class_name>/<video_name>.avi
        test/
          ...
      hmdb/
        train/
          ...
        test/
          ...
      flow/           (optional, pre-computed)
        ucf/train/<class_name>/<video_name>/
          flow_x_00001.jpg, flow_y_00001.jpg, ...
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.optical_flow import compute_optical_flow, flow_frames_to_tensor

# ImageNet normalization for I3D (Kinetics-pretrained expects this)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoDataset(Dataset):
    """
    Base video dataset that loads RGB frames and optical flow clips.

    Each video is segmented into T clips of clip_len frames.
    Returns: rgb_clips (T, 3, clip_len, H, W), flow_clips (T, 2, clip_len, H, W), label
    """

    def __init__(self, video_list, label_list, class_to_idx,
                 num_clips=10, clip_len=8, resolution=224,
                 flow_dir=None, compute_flow_online=True,
                 flow_method='tvl1',
                 is_train=True):
        """
        Args:
            video_list: list of paths to video files.
            label_list: list of integer labels.
            class_to_idx: dict mapping class name to index.
            num_clips: T (number of clips per video).
            clip_len: D (frames per clip).
            resolution: spatial resolution (H=W).
            flow_dir: path to pre-computed flow (None = compute online).
            compute_flow_online: whether to compute flow on the fly.
            is_train: training mode (with augmentation) or test mode.
        """
        self.video_list = video_list
        self.label_list = label_list
        self.class_to_idx = class_to_idx
        self.num_clips = num_clips
        self.clip_len = clip_len
        self.resolution = resolution
        self.flow_dir = flow_dir
        self.compute_flow_online = compute_flow_online
        self.flow_method = flow_method
        self.is_train = is_train

    def __len__(self):
        return len(self.video_list)

    def _load_video_frames(self, video_path):
        """Load all frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to slightly larger for random crop augmentation (train) or direct (test)
            if self.is_train:
                frame = cv2.resize(frame, (256, 256))  # Will random crop to 224
            else:
                frame = cv2.resize(frame, (self.resolution, self.resolution))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read video: {video_path}")
        return frames

    def _segment_into_clips(self, total_frames):
        """
        Segment video into T clips, each of clip_len frames.
        Returns list of (start, end) indices.
        """
        T = self.num_clips
        D = self.clip_len
        n_frames = total_frames

        if n_frames < T * D:
            indices = list(range(n_frames))
            while len(indices) < T * D:
                indices.append(indices[-1])
            n_frames = len(indices)
        else:
            indices = None

        segment_len = n_frames // T
        clips = []
        for i in range(T):
            start = i * segment_len
            if self.is_train:
                max_start = min(start + segment_len - D, n_frames - D)
                clip_start = random.randint(start, max(start, max_start))
            else:
                clip_start = start + (segment_len - D) // 2
                clip_start = max(0, min(clip_start, n_frames - D))
            clips.append((clip_start, clip_start + D))

        return clips, indices

    def _frames_to_rgb_tensor(self, frames, clip_start_h=0, clip_start_w=0):
        """
        Convert list of BGR frames to tensor (3, T, H, W).
        Uses ImageNet normalization for I3D compatibility.
        If clip_start_* are set, crops 224x224 from that position (for augmentation).
        """
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
        arr = arr[:, :, :, ::-1].copy()  # BGR → RGB

        # Random crop for training (frames are 256x256)
        if self.is_train and arr.shape[1] >= self.resolution and arr.shape[2] >= self.resolution:
            arr = arr[:, clip_start_h:clip_start_h + self.resolution,
                      clip_start_w:clip_start_w + self.resolution, :]

        # ImageNet normalization (expected by Kinetics-pretrained I3D)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float()  # (3, T, H, W)
        return tensor

    def _spatial_crop_bgr_frames(self, frames, clip_start_h=0, clip_start_w=0):
        """
        Same 224×224 window as _frames_to_rgb_tensor so RGB and flow stay aligned.
        Training loads 256×256 then crops; test loads 224×224 (crop is full frame).
        """
        if not frames:
            return frames
        h0, w0 = frames[0].shape[:2]
        if h0 >= self.resolution and w0 >= self.resolution:
            return [
                f[
                    clip_start_h : clip_start_h + self.resolution,
                    clip_start_w : clip_start_w + self.resolution,
                ]
                for f in frames
            ]
        return frames

    def _compute_flow_tensor(self, frames, clip_start_h=0, clip_start_w=0):
        """
        Compute optical flow and return tensor (2, D, H, W).
        If clip has D frames, flow has D-1 pairs; pad last frame.
        """
        frames = self._spatial_crop_bgr_frames(frames, clip_start_h, clip_start_w)
        flow = compute_optical_flow(frames, method=self.flow_method)
        if flow.shape[0] < len(frames):
            last = flow[-1:] if flow.shape[0] > 0 else np.zeros_like(flow[:1])
            flow = np.concatenate([flow, last], axis=0)
        flow_tensor = flow_frames_to_tensor(flow[:len(frames)])  # (2, D, H, W)
        return flow_tensor

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.label_list[idx]

        frames = self._load_video_frames(video_path)
        clips_indices, padded_indices = self._segment_into_clips(len(frames))

        if padded_indices is not None:
            frames = [frames[min(i, len(frames) - 1)] for i in padded_indices]

        # Random crop position (shared across all clips for temporal consistency)
        if self.is_train and len(frames[0].shape) >= 2:
            H, W = frames[0].shape[:2]
            if H >= self.resolution and W >= self.resolution:
                max_h = H - self.resolution
                max_w = W - self.resolution
                clip_start_h = random.randint(0, max_h) if max_h > 0 else 0
                clip_start_w = random.randint(0, max_w) if max_w > 0 else 0
            else:
                clip_start_h = clip_start_w = 0
        else:
            clip_start_h = clip_start_w = 0

        # Random horizontal flip (shared across all clips)
        do_flip = self.is_train and random.random() > 0.5

        rgb_clips = []
        flow_clips = []

        for (start, end) in clips_indices:
            clip_frames = frames[start:end]
            if do_flip:
                clip_frames = [cv2.flip(f, 1) for f in clip_frames]

            rgb_tensor = self._frames_to_rgb_tensor(clip_frames, clip_start_h, clip_start_w)
            flow_tensor = self._compute_flow_tensor(
                clip_frames, clip_start_h, clip_start_w
            )
            if do_flip:
                # Flip flow: negate u-component and flip spatial dim (C, T, H, W)
                flow_tensor = flow_tensor.clone()
                flow_tensor[0] = -flow_tensor[0]
                flow_tensor = flow_tensor.flip(-1)  # flip W dimension

            rgb_clips.append(rgb_tensor)
            flow_clips.append(flow_tensor)

        rgb_clips = torch.stack(rgb_clips, dim=0)    # (T, 3, D, H, W)
        flow_clips = torch.stack(flow_clips, dim=0)  # (T, 2, D, H, W)

        return rgb_clips, flow_clips, label


def build_dataset_from_directory(data_root, domain, split, class_names=None):
    """
    Build video list and labels from a directory structure:
      data_root/<domain>/<split>/<class_name>/<video_file>

    Args:
        data_root: root directory.
        domain: e.g., 'ucf', 'hmdb', 'olympic'.
        split: 'train' or 'test'.
        class_names: optional list to filter/order classes.

    Returns:
        video_list, label_list, class_to_idx
    """
    base_dir = os.path.join(data_root, domain, split)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    if class_names is None:
        class_names = sorted(os.listdir(base_dir))
        class_names = [c for c in class_names if os.path.isdir(os.path.join(base_dir, c))]

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    video_list = []
    label_list = []

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        videos = sorted(glob.glob(os.path.join(class_dir, '*')))
        for v in videos:
            ext = os.path.splitext(v)[1].lower()
            if ext in ('.avi', '.mp4', '.mkv', '.mov', '.webm'):
                video_list.append(v)
                label_list.append(class_to_idx[class_name])

    return video_list, label_list, class_to_idx


class SourceVideoDataset(VideoDataset):
    """Convenience wrapper for source domain training (with labels)."""
    pass


class TargetVideoDataset(VideoDataset):
    """
    Target domain dataset for adaptation.
    Labels are only used for evaluation, not during adaptation.
    """
    pass
