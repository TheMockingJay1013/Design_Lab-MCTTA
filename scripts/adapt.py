"""
Test-time adaptation — paper §4.1: SGD lr=0.01, effective batch 64, λ1=1, λ2=0.5.

Logs:
  - Source-only: student multimodal (no adapt)
  - After adapt — during pass: teacher pseudo-label accuracy (online signal)
  - Post-adapt eval: **student** and **teacher** multimodal accuracy (report both; paper protocol may use either)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.video_dataset import VideoDataset, build_dataset_from_directory
from engine import MCTTAEngine
from models.source_model import SourceModel
from utils import accuracy, AverageMeter, get_logger, load_checkpoint, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description='MC-TTA adaptation')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--source_checkpoint', type=str, required=True)
    parser.add_argument('--target_domain', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_clips', type=int, default=None)
    parser.add_argument('--clip_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--accum_steps', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--max_bank_per_class', type=int, default=None)
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--shuffle_target', type=int, default=None, help='1=true 0=false')
    parser.add_argument(
        '--grad_clip_norm',
        type=float,
        default=None,
        help='Clip grad norm before each optimizer step (overrides config if set)',
    )
    parser.add_argument(
        '--normalize_l_ma',
        type=int,
        default=None,
        help='1=divide L_ma by (2*C*d) for scale; 0=paper raw L_ma',
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        alt = os.path.join(os.path.dirname(__file__), '..', args.config)
        cfg_path = alt if os.path.isfile(alt) else args.config

    cfg = load_config(cfg_path) if os.path.isfile(cfg_path) else {}
    dcfg = cfg.get('data', {})
    acfg = cfg.get('adaptation', {})
    sscfg = cfg.get('ssfr', {})

    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    set_seed(seed)
    logger = get_logger('adapt')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    num_clips = args.num_clips if args.num_clips is not None else dcfg.get('num_clips', 10)
    clip_len = args.clip_len if args.clip_len is not None else dcfg.get('clip_len', 8)
    resolution = dcfg.get('resolution', 224)
    flow_method = dcfg.get('flow_method', 'tvl1')

    ckpt = load_checkpoint(args.source_checkpoint, device='cpu')
    class_to_idx = ckpt['class_to_idx']
    num_classes = ckpt['num_classes']

    source_model = SourceModel(num_classes=num_classes)
    source_model.load_state_dict(ckpt['state_dict'])
    logger.info(f'Loaded {args.source_checkpoint} (best_acc={ckpt.get("best_acc", "n/a")})')

    target_videos, target_labels, _ = build_dataset_from_directory(
        args.data_root,
        args.target_domain,
        'test',
        class_names=list(class_to_idx.keys()),
    )
    target_ds = VideoDataset(
        target_videos,
        target_labels,
        class_to_idx,
        num_clips=num_clips,
        clip_len=clip_len,
        resolution=resolution,
        flow_method=flow_method,
        is_train=False,
    )

    shuffle = acfg.get('shuffle_target', True)
    if args.shuffle_target is not None:
        shuffle = bool(args.shuffle_target)
    target_loader = DataLoader(
        target_ds,
        batch_size=args.batch_size if args.batch_size is not None else acfg.get('batch_size', 1),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )

    accum_steps = (
        args.accum_steps if args.accum_steps is not None else acfg.get('accum_steps', 64)
    )
    accum_steps = min(accum_steps, len(target_ds)) if len(target_ds) else 1

    gcn = acfg.get('grad_clip_norm')
    if gcn is not None:
        try:
            gcn = float(gcn)
        except (TypeError, ValueError):
            gcn = None
        if gcn is not None and gcn <= 0:
            gcn = None
    if args.grad_clip_norm is not None:
        gcn = args.grad_clip_norm if args.grad_clip_norm > 0 else None

    nma = bool(acfg.get('normalize_l_ma', False))
    if args.normalize_l_ma is not None:
        nma = bool(args.normalize_l_ma)

    def _eng():
        return MCTTAEngine(
            source_model=source_model,
            num_classes=num_classes,
            alpha=args.alpha if args.alpha is not None else acfg.get('alpha', 0.3),
            beta=args.beta if args.beta is not None else acfg.get('beta', 0.6),
            top_k=args.top_k if args.top_k is not None else acfg.get('top_k', 5),
            max_bank_per_class=(
                args.max_bank_per_class
                if args.max_bank_per_class is not None
                else acfg.get('max_bank_per_class', 512)
            ),
            lambda1=args.lambda1 if args.lambda1 is not None else acfg.get('lambda1', 1.0),
            lambda2=args.lambda2 if args.lambda2 is not None else acfg.get('lambda2', 0.5),
            gamma=args.gamma if args.gamma is not None else acfg.get('gamma', 0.999),
            lr=args.lr if args.lr is not None else acfg.get('lr', 0.01),
            momentum=acfg.get('momentum', 0.9),
            weight_decay=acfg.get('weight_decay', 1e-4),
            device=device,
            ssfr_pairwise_metric=sscfg.get('pairwise_metric', 'cosine_distance'),
            ssfr_entropy_mode=sscfg.get('entropy_mode', 'low_entropy_threshold'),
            normalize_l_ma=nma,
            grad_clip_norm=gcn,
        )

    # --- Source-only (student, multimodal) ---
    logger.info('=== Source-only (student multimodal, no adaptation) ===')
    eng0 = _eng()
    src_preds, src_labels = [], []
    for rgb, flow, y in tqdm(target_loader, desc='Source-only'):
        for i in range(rgb.size(0)):
            src_preds.append(eng0.predict_multimodal_student(rgb[i], flow[i]))
            src_labels.append(y[i].item())
    source_acc = accuracy(src_preds, src_labels)
    logger.info(f'Source-only accuracy: {source_acc:.2f}%')

    del eng0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # --- Adaptation ---
    engine = _eng()
    logger.info(
        f'=== MC-TTA (accum_steps={accum_steps}, λ1={engine.criterion.lambda1}, λ2={engine.criterion.lambda2}) ==='
    )
    during_teacher_preds, all_labels = [], []
    loss_meter = AverageMeter()
    accum_count = 0
    # True gradient accumulation: backward each micro-batch so only one graph
    # lives at a time. Summing 64 losses then backward once retains 64 graphs (~OOM).
    engine.optimizer.zero_grad(set_to_none=True)

    for rgb, flow, labels in tqdm(target_loader, desc='Adapting'):
        for i in range(rgb.size(0)):
            pred, loss_dict, total = engine.adapt_accumulate_step(rgb[i], flow[i])
            during_teacher_preds.append(pred)
            all_labels.append(labels[i].item())
            loss_meter.update(loss_dict['total'])

            (total / accum_steps).backward()
            accum_count += 1

            if accum_count >= accum_steps:
                engine.optimizer_step_with_ema()
                accum_count = 0

        step = len(during_teacher_preds)
        if step and step % 50 == 0:
            logger.info(
                f'Step {step}/{len(target_ds)} loss_avg={loss_meter.avg:.4f} '
                f'teacher_pseudo_acc={accuracy(during_teacher_preds, all_labels):.2f}%'
            )

    if accum_count > 0:
        engine.optimizer_step_with_ema()

    during_teacher_acc = accuracy(during_teacher_preds, all_labels)
    logger.info(f'During adaptation (teacher pseudo-label) accuracy: {during_teacher_acc:.2f}%')

    # --- Post-adapt: student + teacher multimodal ---
    logger.info('=== Post-adaptation evaluation ===')
    stu_preds, tea_preds, ev_labels = [], [], []
    for rgb, flow, y in tqdm(target_loader, desc='Post-adapt eval'):
        for i in range(rgb.size(0)):
            stu_preds.append(engine.predict_multimodal_student(rgb[i], flow[i]))
            tea_preds.append(engine.predict_multimodal_teacher(rgb[i], flow[i]))
            ev_labels.append(y[i].item())

    acc_stu = accuracy(stu_preds, ev_labels)
    acc_tea = accuracy(tea_preds, ev_labels)
    logger.info(f'Post-adapt student multimodal accuracy: {acc_stu:.2f}%')
    logger.info(f'Post-adapt teacher (EMA) multimodal accuracy: {acc_tea:.2f}%')
    logger.info(
        f'Summary: source_only={source_acc:.2f}% | post_student={acc_stu:.2f}% | post_teacher={acc_tea:.2f}%'
    )


if __name__ == '__main__':
    main()
