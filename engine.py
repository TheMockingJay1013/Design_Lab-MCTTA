"""
MC-TTA engine — teacher (EMA) + student, SSFR, memory banks, Eq.9–10.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.source_model import SourceModel
from mctta.memory_banks import StudentMemoryBank, TeacherMemoryBank
from mctta.ssfr import SSFR
from mctta.losses import MCTTALoss


class MCTTAEngine:
    def __init__(
        self,
        source_model: SourceModel,
        num_classes: int,
        feature_dim: int = 1024,
        alpha: float = 0.3,
        beta: float = 0.6,
        top_k: int = 5,
        max_bank_per_class: int = 512,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        gamma: float = 0.999,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        device: str | torch.device = 'cuda',
        ssfr_pairwise_metric: str = 'cosine_distance',
        ssfr_entropy_mode: str = 'low_entropy_threshold',
        normalize_l_ma: bool = False,
        grad_clip_norm: float | None = None,
    ):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm

        self.teacher = copy.deepcopy(source_model).to(self.device)
        self.student = copy.deepcopy(source_model).to(self.device)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self._freeze_student_fc2()
        self._set_student_train_mode()

        self.teacher_bank = TeacherMemoryBank(
            num_classes, feature_dim, top_k=top_k, max_per_class=max_bank_per_class
        ).to(self.device)
        self.teacher_bank.initialize_from_classifier(
            source_model.rgb_classifier, source_model.flow_classifier
        )

        self.student_bank = StudentMemoryBank(num_classes, feature_dim).to(self.device)

        self.ssfr = SSFR(
            alpha=alpha,
            beta=beta,
            pairwise_metric=ssfr_pairwise_metric,
            entropy_mode=ssfr_entropy_mode,
        )
        self.criterion = MCTTALoss(
            lambda1=lambda1,
            lambda2=lambda2,
            normalize_l_ma=normalize_l_ma,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

        trainable = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def _freeze_student_fc2(self):
        for mod in (
            self.student.rgb_classifier.fc2,
            self.student.flow_classifier.fc2,
            self.student.multi_classifier.fc2,
        ):
            for p in mod.parameters():
                p.requires_grad = False

    def _set_student_train_mode(self):
        self.student.train()

    @staticmethod
    def _backbone_clips(backbone: nn.Module, clips: torch.Tensor) -> torch.Tensor:
        """clips (T, C, D, H, W) -> (T, d)."""
        return backbone(clips)

    @torch.no_grad()
    def _teacher_forward_video(
        self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        rgb_clip_feats = self._backbone_clips(self.teacher.rgb_backbone, rgb_clips)
        flow_clip_feats = self._backbone_clips(self.teacher.flow_backbone, flow_clips)
        rgb_clip_logits = self.teacher.rgb_classifier(rgb_clip_feats)
        flow_clip_logits = self.teacher.flow_classifier(flow_clip_feats)

        rgb_v, flow_v, _ = self.ssfr.reconstruct(
            rgb_clip_feats,
            flow_clip_feats,
            rgb_clip_logits,
            flow_clip_logits,
        )
        multi_feat = torch.cat([rgb_v, flow_v], dim=-1).unsqueeze(0)
        multi_logits = self.teacher.multi_classifier(multi_feat)
        pseudo_label = int(multi_logits.argmax(dim=-1).item())

        rgb_v_logits = self.teacher.rgb_classifier(rgb_v.unsqueeze(0)).squeeze(0)
        flow_v_logits = self.teacher.flow_classifier(flow_v.unsqueeze(0)).squeeze(0)
        return rgb_v, flow_v, pseudo_label, rgb_v_logits, flow_v_logits

    def _student_forward_video(
        self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_clip_feats = self._backbone_clips(self.student.rgb_backbone, rgb_clips)
        flow_clip_feats = self._backbone_clips(self.student.flow_backbone, flow_clips)
        rgb_v = rgb_clip_feats.mean(dim=0)
        flow_v = flow_clip_feats.mean(dim=0)
        rgb_logits = self.student.rgb_classifier(rgb_v.unsqueeze(0))
        flow_logits = self.student.flow_classifier(flow_v.unsqueeze(0))
        multi_logits = self.student.multi_classifier(
            torch.cat([rgb_v, flow_v], dim=-1).unsqueeze(0)
        )
        return rgb_v, flow_v, rgb_logits, flow_logits, multi_logits

    @torch.no_grad()
    def _ema_update_teacher(self):
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(self.gamma).add_(sp.data, alpha=1.0 - self.gamma)

    def _adapt_step_impl(
        self,
        rgb_clips: torch.Tensor,
        flow_clips: torch.Tensor,
        accumulate_only: bool = False,
    ):
        rgb_clips = rgb_clips.to(self.device)
        flow_clips = flow_clips.to(self.device)

        t_rgb, t_flow, pseudo_label, t_rgb_log, t_flow_log = self._teacher_forward_video(
            rgb_clips, flow_clips
        )
        self.teacher_bank.push(pseudo_label, t_rgb, t_flow, t_rgb_log, t_flow_log)

        s_rgb, s_flow, s_rgb_l, s_flow_l, s_multi_l = self._student_forward_video(
            rgb_clips, flow_clips
        )

        pseudo_rgb_p, pseudo_flow_p = self.teacher_bank.get_pseudo_prototypes(self.device)
        tgt_rgb_p, tgt_flow_p = self.student_bank.get_target_prototypes(
            s_rgb, s_flow, pseudo_label
        )
        self.student_bank.update(s_rgb, s_flow, pseudo_label)

        y = torch.tensor([pseudo_label], device=self.device, dtype=torch.long)
        total, loss_dict = self.criterion(
            s_rgb_l,
            s_flow_l,
            s_multi_l,
            y,
            pseudo_rgb_p,
            pseudo_flow_p,
            tgt_rgb_p,
            tgt_flow_p,
        )

        if not accumulate_only:
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            self._ema_update_teacher()

        return pseudo_label, loss_dict, total

    def adapt_step(self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor):
        pred, loss_dict, _ = self._adapt_step_impl(
            rgb_clips, flow_clips, accumulate_only=False
        )
        return pred, loss_dict

    def adapt_accumulate_step(
        self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor
    ) -> Tuple[int, Dict[str, float], torch.Tensor]:
        return self._adapt_step_impl(rgb_clips, flow_clips, accumulate_only=True)

    def optimizer_step_from_accumulated(self, scaled_loss: torch.Tensor):
        """Legacy: backward on a *sum* of losses — do not use (keeps huge graphs). Prefer backward_microbatch."""
        self.optimizer.zero_grad()
        scaled_loss.backward()
        self.optimizer.step()
        self._ema_update_teacher()

    def optimizer_step_with_ema(self):
        """After micro-batch backward() calls have accumulated grads, apply update + EMA."""
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad],
                self.grad_clip_norm,
            )
        self.optimizer.step()
        self._ema_update_teacher()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_multimodal_student(self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor) -> int:
        rgb_clips = rgb_clips.to(self.device)
        flow_clips = flow_clips.to(self.device)
        self.student.eval()
        rgb_f = self.student.rgb_backbone(rgb_clips).mean(dim=0)
        flow_f = self.student.flow_backbone(flow_clips).mean(dim=0)
        logits = self.student.multi_classifier(
            torch.cat([rgb_f, flow_f], dim=-1).unsqueeze(0)
        )
        self._set_student_train_mode()
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def predict_multimodal_teacher(self, rgb_clips: torch.Tensor, flow_clips: torch.Tensor) -> int:
        rgb_clips = rgb_clips.to(self.device)
        flow_clips = flow_clips.to(self.device)
        self.teacher.eval()
        _, _, pl, _, _ = self._teacher_forward_video(rgb_clips, flow_clips)
        return pl
