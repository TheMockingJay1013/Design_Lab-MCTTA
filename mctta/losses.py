"""
MC-TTA losses — paper Eq.5–9.

L_ma: sum of squared L2 norms (no extra normalization; λ1 scales in Eq.9).
L_cmr: symmetric KL on row-softmax(-Euclidean) relationship matrices (Eq.6–7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Eq.8: sum of CE for RGB, flow, multimodal heads vs pseudo-label."""

    def forward(
        self,
        rgb_logits: torch.Tensor,
        flow_logits: torch.Tensor,
        multi_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        return (
            F.cross_entropy(rgb_logits, pseudo_labels)
            + F.cross_entropy(flow_logits, pseudo_labels)
            + F.cross_entropy(multi_logits, pseudo_labels)
        )


class MultimodalPrototypeAlignmentLoss(nn.Module):
    """Eq.5 — L2 squared summed over classes and modalities."""

    def forward(
        self,
        pseudo_rgb: torch.Tensor,
        pseudo_flow: torch.Tensor,
        target_rgb: torch.Tensor,
        target_flow: torch.Tensor,
    ) -> torch.Tensor:
        loss_rgb = (pseudo_rgb - target_rgb).pow(2).sum()
        loss_flow = (pseudo_flow - target_flow).pow(2).sum()
        return loss_rgb + loss_flow


class CrossModalRelativeConsistencyLoss(nn.Module):
    """Eq.6–7 — KL on row-normalized similarity from negative distances."""

    def forward(self, target_rgb: torch.Tensor, target_flow: torch.Tensor) -> torch.Tensor:
        a_r = torch.cdist(target_rgb, target_rgb, p=2)
        a_o = torch.cdist(target_flow, target_flow, p=2)
        p_r = F.softmax(-a_r, dim=-1).clamp_min(1e-8)
        p_o = F.softmax(-a_o, dim=-1).clamp_min(1e-8)
        kl_ro = F.kl_div(p_r.log(), p_o, reduction='batchmean')
        kl_or = F.kl_div(p_o.log(), p_r, reduction='batchmean')
        return kl_ro + kl_or


class MCTTALoss(nn.Module):
    """Eq.9: L = L_CE + λ1 L_ma + λ2 L_cmr."""

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        normalize_l_ma: bool = False,
        num_classes: int = 12,
        feature_dim: int = 1024,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.normalize_l_ma = normalize_l_ma
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ce_loss = ClassificationLoss()
        self.mpa_loss = MultimodalPrototypeAlignmentLoss()
        self.cmr_loss = CrossModalRelativeConsistencyLoss()

    def forward(
        self,
        rgb_logits: torch.Tensor,
        flow_logits: torch.Tensor,
        multi_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        pseudo_rgb_proto: torch.Tensor,
        pseudo_flow_proto: torch.Tensor,
        target_rgb_proto: torch.Tensor,
        target_flow_proto: torch.Tensor,
    ):
        l_ce = self.ce_loss(rgb_logits, flow_logits, multi_logits, pseudo_labels)
        l_ma = self.mpa_loss(
            pseudo_rgb_proto,
            pseudo_flow_proto,
            target_rgb_proto,
            target_flow_proto,
        )
        if self.normalize_l_ma:
            denom = float(2 * self.num_classes * self.feature_dim)
            l_ma = l_ma / max(denom, 1.0)
        l_cmr = self.cmr_loss(target_rgb_proto, target_flow_proto)
        total = l_ce + self.lambda1 * l_ma + self.lambda2 * l_cmr
        return total, {
            'L_CE': l_ce.item(),
            'L_ma': l_ma.item(),
            'L_cmr': l_cmr.item(),
            'total': total.item(),
        }
