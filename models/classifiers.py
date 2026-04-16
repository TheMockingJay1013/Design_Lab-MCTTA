"""
Classifiers for MC-TTA.

Per the paper:
  - RGB classifier ψ_r: d → hidden → C
  - Flow classifier ψ_o: d → hidden → C
  - Multimodal classifier ψ_m: 2d → hidden → C  (concat of RGB + flow features)
  - All are two-layer perceptrons with ReLU activation and Softmax output.
  - d = 1024, hidden = 1024
"""

import torch
import torch.nn as nn


class SingleModalityClassifier(nn.Module):
    """Two-layer MLP classifier for a single modality (RGB or optical flow)."""

    def __init__(self, input_dim=1024, hidden_dim=1024, num_classes=12, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Returns raw logits (before softmax)."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MultimodalClassifier(nn.Module):
    """Two-layer MLP classifier for concatenated RGB + flow features."""

    def __init__(self, input_dim=2048, hidden_dim=1024, num_classes=12, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Returns raw logits (before softmax)."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
