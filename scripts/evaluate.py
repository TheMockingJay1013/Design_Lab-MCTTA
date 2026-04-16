"""Evaluate a checkpoint on a domain split (multimodal head, mean clips)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.video_dataset import VideoDataset, build_dataset_from_directory
from models.source_model import SourceModel
from utils import accuracy, get_logger, load_checkpoint, load_config, set_seed


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    labels_all = []

    for rgb_clips, flow_clips, labels in tqdm(dataloader, desc='Evaluating'):
        b = rgb_clips.size(0)
        rgb_clips = rgb_clips.to(device)
        flow_clips = flow_clips.to(device)
        t = rgb_clips.size(1)
        rgb_f = model.rgb_backbone(rgb_clips.reshape(b * t, 3, *rgb_clips.shape[3:])).view(b, t, -1).mean(1)
        flow_f = model.flow_backbone(flow_clips.reshape(b * t, 2, *flow_clips.shape[3:])).view(b, t, -1).mean(1)
        logits = model.multi_classifier(torch.cat([rgb_f, flow_f], dim=-1))
        pred = logits.argmax(dim=-1)
        for i in range(b):
            predictions.append(pred[i].item())
            labels_all.append(labels[i].item())

    return predictions, labels_all


def main():
    parser = argparse.ArgumentParser(description='MC-TTA evaluation')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_domain', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_clips', type=int, default=None)
    parser.add_argument('--clip_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        alt = os.path.join(os.path.dirname(__file__), '..', args.config)
        cfg_path = alt if os.path.isfile(alt) else args.config
    cfg = load_config(cfg_path) if os.path.isfile(cfg_path) else {}
    dcfg = cfg.get('data', {})

    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    set_seed(seed)
    logger = get_logger('evaluate')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt = load_checkpoint(args.checkpoint, device='cpu')
    class_to_idx = ckpt['class_to_idx']
    num_classes = ckpt['num_classes']

    model = SourceModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    logger.info(f'Loaded {args.checkpoint}')

    target_videos, target_labels, _ = build_dataset_from_directory(
        args.data_root,
        args.target_domain,
        args.split,
        class_names=list(class_to_idx.keys()),
    )

    num_clips = args.num_clips if args.num_clips is not None else dcfg.get('num_clips', 10)
    clip_len = args.clip_len if args.clip_len is not None else dcfg.get('clip_len', 8)
    resolution = dcfg.get('resolution', 224)
    flow_method = dcfg.get('flow_method', 'tvl1')

    dataset = VideoDataset(
        target_videos,
        target_labels,
        class_to_idx,
        num_clips=num_clips,
        clip_len=clip_len,
        resolution=resolution,
        flow_method=flow_method,
        is_train=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    preds, labels_all = evaluate_model(model, dataloader, device)
    acc = accuracy(preds, labels_all)
    logger.info(f'Multimodal accuracy: {acc:.2f}%')

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_correct = {}
    class_total = {}
    for p, l in zip(preds, labels_all):
        cls = idx_to_class[l]
        class_total[cls] = class_total.get(cls, 0) + 1
        if p == l:
            class_correct[cls] = class_correct.get(cls, 0) + 1

    logger.info('Per-class accuracy:')
    for cls in sorted(class_total.keys()):
        c = class_correct.get(cls, 0)
        t = class_total[cls]
        logger.info(f'  {cls}: {c}/{t} = {c / t * 100:.1f}%')


if __name__ == '__main__':
    main()
