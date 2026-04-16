"""
Teacher / student memory banks — ring buffer per class on GPU (FIFO).

Teacher: SSFR video-level features + logits; pseudo-prototypes = mean of top-K
lowest-entropy stored samples per class (paper §3.2.2).

Student: running average of mean-clip features per class (§3.2.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


class TeacherMemoryBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        top_k: int = 5,
        max_per_class: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.max_per_class = max_per_class

        self.register_buffer('_rgb', torch.zeros(num_classes, max_per_class, feature_dim))
        self.register_buffer('_flow', torch.zeros(num_classes, max_per_class, feature_dim))
        self.register_buffer(
            '_rgb_log',
            torch.zeros(num_classes, max_per_class, num_classes),
        )
        self.register_buffer(
            '_flow_log',
            torch.zeros(num_classes, max_per_class, num_classes),
        )
        self.register_buffer('_count', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('_head', torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def initialize_from_classifier(self, rgb_classifier: nn.Module, flow_classifier: nn.Module):
        w_r = rgb_classifier.fc2.weight.data.clone()
        w_o = flow_classifier.fc2.weight.data.clone()
        for c in range(self.num_classes):
            self._rgb[c, 0] = w_r[c]
            self._flow[c, 0] = w_o[c]
            self._count[c] = 1

    @torch.no_grad()
    def push(
        self,
        c: int,
        rgb_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        rgb_logit: torch.Tensor,
        flow_logit: torch.Tensor,
    ):
        """Append one video-level sample for class c (FIFO)."""
        rgb_feat = rgb_feat.reshape(-1).to(self._rgb.dtype)
        flow_feat = flow_feat.reshape(-1).to(self._flow.dtype)
        rgb_logit = rgb_logit.reshape(self.num_classes).to(self._rgb_log.dtype)
        flow_logit = flow_logit.reshape(self.num_classes).to(self._flow_log.dtype)

        M = self.max_per_class
        cnt = int(self._count[c].item())
        if cnt < M:
            idx = cnt
            self._count[c] = cnt + 1
        else:
            idx = int(self._head[c].item())
            self._head[c] = (idx + 1) % M

        self._rgb[c, idx] = rgb_feat
        self._flow[c, idx] = flow_feat
        self._rgb_log[c, idx] = rgb_logit
        self._flow_log[c, idx] = flow_logit

    def _gather_entries(self, c: int):
        n = int(self._count[c].item())
        if n == 0:
            return None, None, None, None
        if n < self.max_per_class:
            return (
                self._rgb[c, :n],
                self._flow[c, :n],
                self._rgb_log[c, :n],
                self._flow_log[c, :n],
            )
        h = int(self._head[c].item())
        dev = self._rgb.device
        idx = torch.cat([torch.arange(h, self.max_per_class, device=dev), torch.arange(0, h, device=dev)])
        return (
            self._rgb[c, idx],
            self._flow[c, idx],
            self._rgb_log[c, idx],
            self._flow_log[c, idx],
        )

    @torch.no_grad()
    def get_pseudo_prototypes(self, device: torch.device):
        """Returns (C, d), (C, d) on device."""
        c_dim = self.num_classes
        d = self.feature_dim
        rgb_p = torch.zeros(c_dim, d, device=device, dtype=self._rgb.dtype)
        flow_p = torch.zeros(c_dim, d, device=device, dtype=self._flow.dtype)

        for c in range(c_dim):
            gathered = self._gather_entries(c)
            if gathered is None:
                continue
            rgb_e, flow_e, rl, fl = gathered
            rgb_e = rgb_e.to(device)
            flow_e = flow_e.to(device)
            rl = rl.to(device)
            fl = fl.to(device)
            n = rgb_e.size(0)
            if n <= self.top_k:
                rgb_p[c] = rgb_e.mean(0)
                flow_p[c] = flow_e.mean(0)
                continue
            h_r = _entropy_from_logits(rl)
            h_o = _entropy_from_logits(fl)
            avg_h = (h_r + h_o) * 0.5
            k = min(self.top_k, n)
            _, ind = torch.topk(avg_h, k, largest=False)
            rgb_p[c] = rgb_e[ind].mean(0)
            flow_p[c] = flow_e[ind].mean(0)

        return rgb_p, flow_p


class StudentMemoryBank(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.register_buffer('rgb_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('flow_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('counts', torch.zeros(num_classes, dtype=torch.float32))

    @torch.no_grad()
    def update(self, rgb_feat: torch.Tensor, flow_feat: torch.Tensor, pseudo_label: int):
        c = pseudo_label
        rgb_feat = rgb_feat.detach().reshape(-1)
        flow_feat = flow_feat.detach().reshape(-1)
        n = self.counts[c].item() + 1.0
        self.rgb_prototypes[c].mul_((n - 1) / n).add_(rgb_feat / n)
        self.flow_prototypes[c].mul_((n - 1) / n).add_(flow_feat / n)
        self.counts[c] = n

    def get_target_prototypes(
        self,
        rgb_feat: torch.Tensor | None = None,
        flow_feat: torch.Tensor | None = None,
        pseudo_label: int | None = None,
    ):
        """Include current (differentiable) sample in class row for gradients."""
        rgb_proto = self.rgb_prototypes.clone()
        flow_proto = self.flow_prototypes.clone()
        if rgb_feat is None or flow_feat is None or pseudo_label is None:
            return rgb_proto, flow_proto

        c = int(pseudo_label)
        n = float(self.counts[c].item())
        rgb_feat = rgb_feat.reshape(-1)
        flow_feat = flow_feat.reshape(-1)
        rgb_proto = rgb_proto.clone()
        flow_proto = flow_proto.clone()
        rgb_proto[c] = (n / (n + 1.0)) * self.rgb_prototypes[c] + rgb_feat / (n + 1.0)
        flow_proto[c] = (n / (n + 1.0)) * self.flow_prototypes[c] + flow_feat / (n + 1.0)
        return rgb_proto, flow_proto
