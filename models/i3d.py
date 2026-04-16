"""
Inception-I3D (Inflated 3D ConvNet) backbone.

Based on: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
by Carreira & Zisserman (CVPR 2017).

Supports both RGB (in_channels=3) and Optical Flow (in_channels=2) inputs.
Output: 1024-dimensional feature vector per clip.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    """MaxPool3d with 'SAME' padding to match TensorFlow behavior."""

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    """Basic Conv3d + BN + ReLU unit with SAME padding."""

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name='unit_3d'):
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """3D Inception module with four parallel branches."""

    def __init__(self, in_channels, out_channels, name=''):
        super().__init__()
        # out_channels: [b0, b1a, b1b, b2a, b2b, b3]
        self.b0 = Unit3D(
            in_channels, out_channels[0], kernel_shape=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(
            in_channels, out_channels[1], kernel_shape=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(
            out_channels[1], out_channels[2], kernel_shape=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(
            in_channels, out_channels[3], kernel_shape=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(
            out_channels[3], out_channels[4], kernel_shape=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels, out_channels[5], kernel_shape=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    Inception-I3D model for video feature extraction.

    Args:
        in_channels: 3 for RGB, 2 for optical flow.
        final_endpoint: which layer to stop at (for feature extraction).
    """

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1',
        'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c',
        'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d',
        'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b',
        'Mixed_5c', 'Logits', 'Predictions',
    )

    def __init__(self, in_channels=3, final_endpoint='Mixed_5c',
                 name='inception_i3d'):
        super().__init__()
        self._in_channels = in_channels
        self._final_endpoint = final_endpoint
        self.feature_dim = 1024

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown endpoint {self._final_endpoint}')

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(
            in_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2),
            padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(
            64, 64, kernel_shape=(1, 1, 1), name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(
            64, 192, kernel_shape=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            480, [192, 96, 208, 16, 48, 64], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            512, [160, 112, 224, 24, 64, 64], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            512, [128, 128, 256, 24, 64, 64], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            512, [112, 144, 288, 32, 64, 64], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            528, [256, 160, 320, 32, 128, 128], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            832, [256, 160, 320, 32, 128, 128], name=name + end_point)
        if self._final_endpoint == end_point:
            self._build(); return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            832, [384, 192, 384, 48, 128, 128], name=name + end_point)
        self._build()

    def _build(self):
        for k in self.end_points:
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) tensor — C=3 for RGB, C=2 for flow.
        Returns:
            features: (B, 1024) feature vector.
        """
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
            if end_point == self._final_endpoint:
                break

        # Global average pooling over (T, H, W)
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        return x

    def extract_features(self, x):
        """Alias for forward — returns (B, 1024) features."""
        return self.forward(x)
