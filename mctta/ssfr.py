"""
Self-Assembled Source-Friendly Feature Reconstruction (SSFR), paper §3.2.2.

Default behavior matches paper *prose* (low entropy ≈ source-like):
  - pairwise_metric=c cosine_distance: d = 1 - cos(p̂_r, p̂_o), select d <= alpha
  - entropy_mode=low_entropy_threshold: H(p̂) <= beta for both modalities

Optional toggles for PDF / reproduction experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SSFRConfig:
    alpha: float = 0.3
    beta: float = 0.6
    pairwise_metric: str = 'cosine_distance'
    entropy_mode: str = 'low_entropy_threshold'


class SSFR:
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.6,
        pairwise_metric: str = 'cosine_distance',
        entropy_mode: str = 'low_entropy_threshold',
    ):
        self.alpha = alpha
        self.beta = beta
        self.pairwise_metric = pairwise_metric
        self.entropy_mode = entropy_mode

    def _pairwise_ok(self, rgb_logits: torch.Tensor, flow_logits: torch.Tensor) -> torch.Tensor:
        """Per-clip mask (T,) bool."""
        if self.pairwise_metric == 'cosine_distance':
            cos_sim = F.cosine_similarity(rgb_logits, flow_logits, dim=-1)
            d = 1.0 - cos_sim
            return d <= self.alpha
        if self.pairwise_metric == 'cosine_similarity_pdf':
            cos_sim = F.cosine_similarity(rgb_logits, flow_logits, dim=-1)
            return cos_sim <= self.alpha
        raise ValueError(f'Unknown pairwise_metric: {self.pairwise_metric}')

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1)

    def _confident_mask(self, rgb_logits: torch.Tensor, flow_logits: torch.Tensor) -> torch.Tensor:
        h_r = self._entropy(rgb_logits)
        h_o = self._entropy(flow_logits)
        if self.entropy_mode == 'low_entropy_threshold':
            return (h_r <= self.beta) & (h_o <= self.beta)
        if self.entropy_mode == 'high_entropy_threshold':
            return (h_r >= self.beta) & (h_o >= self.beta)
        raise ValueError(f'Unknown entropy_mode: {self.entropy_mode}')

    @torch.no_grad()
    def select_mask(
        self, rgb_clip_logits: torch.Tensor, flow_clip_logits: torch.Tensor
    ) -> torch.Tensor:
        """Return (T,) bool — source-friendly clips."""
        ok_pair = self._pairwise_ok(rgb_clip_logits, flow_clip_logits)
        ok_conf = self._confident_mask(rgb_clip_logits, flow_clip_logits)
        return ok_pair & ok_conf

    @torch.no_grad()
    def reconstruct(
        self,
        rgb_clip_features: torch.Tensor,
        flow_clip_features: torch.Tensor,
        rgb_clip_logits: torch.Tensor,
        flow_clip_logits: torch.Tensor,
    ):
        """
        Returns:
            rgb_video_feat (d,), flow_video_feat (d,), used_ssfr: bool
        """
        mask = self.select_mask(rgb_clip_logits, flow_clip_logits)
        if mask.sum() == 0:
            return (
                rgb_clip_features.mean(dim=0),
                flow_clip_features.mean(dim=0),
                False,
            )
        return (
            rgb_clip_features[mask].mean(dim=0),
            flow_clip_features[mask].mean(dim=0),
            True,
        )
