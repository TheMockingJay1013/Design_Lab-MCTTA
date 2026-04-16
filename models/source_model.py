"""Full source-pretrained dual-stream I3D + three classifiers (paper §3.2.1)."""

from __future__ import annotations

import torch.nn as nn

from models.i3d import InceptionI3d
from models.classifiers import SingleModalityClassifier, MultimodalClassifier


class SourceModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.rgb_backbone = InceptionI3d(in_channels=3)
        self.flow_backbone = InceptionI3d(in_channels=2)
        self.rgb_classifier = SingleModalityClassifier(
            feature_dim, hidden_dim, num_classes, dropout=dropout
        )
        self.flow_classifier = SingleModalityClassifier(
            feature_dim, hidden_dim, num_classes, dropout=dropout
        )
        self.multi_classifier = MultimodalClassifier(
            feature_dim * 2, hidden_dim, num_classes, dropout=dropout
        )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
