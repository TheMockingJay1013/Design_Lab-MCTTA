"""Unit tests for SSFR, losses, memory banks (paper §3)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mctta.losses import MCTTALoss, MultimodalPrototypeAlignmentLoss
from mctta.memory_banks import StudentMemoryBank, TeacherMemoryBank
from mctta.ssfr import SSFR


def test_ssfr_mask_low_entropy():
    """Confident + agreeing logits -> mask True."""
    t = 4
    c = 5
    # Same peak on rgb and flow -> cosine sim 1 -> distance 0
    rgb = torch.zeros(t, c)
    flow = torch.zeros(t, c)
    rgb[:, 0] = 10.0
    flow[:, 0] = 10.0
    ssfr = SSFR(alpha=0.3, beta=0.6, entropy_mode='low_entropy_threshold')
    m = ssfr.select_mask(rgb, flow)
    assert m.all()


def test_ssfr_high_entropy_mode():
    rgb = torch.randn(3, 4)
    flow = torch.randn(3, 4)
    ssfr = SSFR(alpha=1.0, beta=0.01, entropy_mode='high_entropy_threshold')
    _ = ssfr.select_mask(rgb, flow)
    assert _.shape == (3,)


def test_l_ma_sum_squares():
    m = MultimodalPrototypeAlignmentLoss()
    pr = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pf = pr.clone()
    tr = torch.zeros_like(pr)
    tf = torch.zeros_like(pr)
    loss = m(pr, pf, tr, tf)
    assert loss.item() == pytest.approx(4.0, rel=1e-5)


def test_mctta_loss_forward():
    crit = MCTTALoss(lambda1=1.0, lambda2=0.5)
    c = 3
    d = 4
    rgb_l = torch.randn(1, c, requires_grad=True)
    fl = torch.randn(1, c, requires_grad=True)
    ml = torch.randn(1, c, requires_grad=True)
    y = torch.tensor([1])
    pr = torch.randn(c, d, requires_grad=True)
    pf = torch.randn(c, d, requires_grad=True)
    tr = torch.randn(c, d, requires_grad=True)
    tf = torch.randn(c, d, requires_grad=True)
    total, _ = crit(rgb_l, fl, ml, y, pr, pf, tr, tf)
    total.backward()
    assert rgb_l.grad is not None


def test_teacher_bank_topk():
    torch.manual_seed(0)
    c = 3
    d = 8
    bank = TeacherMemoryBank(c, d, top_k=2, max_per_class=16)
    from models.classifiers import SingleModalityClassifier

    rgb_c = SingleModalityClassifier(d, d, c)
    flow_c = SingleModalityClassifier(d, d, c)
    bank.initialize_from_classifier(rgb_c, flow_c)

    for _ in range(6):
        feat = torch.randn(d)
        log = torch.randn(c)
        bank.push(1, feat, feat.clone(), log, log.clone())

    pr, pf = bank.get_pseudo_prototypes(torch.device('cpu'))
    assert pr.shape == (c, d)
    assert not torch.isnan(pr).any()


def test_student_running_avg():
    b = StudentMemoryBank(2, 4)
    f1 = torch.ones(4)
    b.update(f1, f1 * 2, 0)
    b.update(f1 * 3, f1 * 4, 0)
    assert b.counts[0].item() == pytest.approx(2.0)
    assert b.rgb_prototypes[0, 0].item() == pytest.approx(2.0)
