"""
Source-domain pre-training with batched clip I3D forward (B*T clips per step).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.video_dataset import VideoDataset, build_dataset_from_directory
from models.source_model import SourceModel
from models.weights import load_i3d_weights
from utils import AverageMeter, get_logger, load_config, save_checkpoint, set_seed


def _video_level_features(model: SourceModel, rgb_clips: torch.Tensor, flow_clips: torch.Tensor):
    """
    rgb_clips, flow_clips: (B, T, C, D, H, W) on device.
    Returns (B, d), (B, d).
    """
    b, t, _, d, h, w = rgb_clips.shape
    rgb_flat = rgb_clips.reshape(b * t, 3, d, h, w)
    flow_flat = flow_clips.reshape(b * t, 2, d, h, w)
    rgb_f = model.rgb_backbone(rgb_flat).view(b, t, -1).mean(dim=1)
    flow_f = model.flow_backbone(flow_flat).view(b, t, -1).mean(dim=1)
    return rgb_f, flow_f


def train_one_epoch(model, dataloader, optimizer, device, logger):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for rgb_clips, flow_clips, labels in tqdm(dataloader, desc='Training'):
        rgb_clips = rgb_clips.to(device, non_blocking=True)
        flow_clips = flow_clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        rgb_f, flow_f = _video_level_features(model, rgb_clips, flow_clips)
        rgb_logits = model.rgb_classifier(rgb_f)
        flow_logits = model.flow_classifier(flow_f)
        multi_logits = model.multi_classifier(torch.cat([rgb_f, flow_f], dim=-1))

        loss = (
            F.cross_entropy(rgb_logits, labels)
            + F.cross_entropy(flow_logits, labels)
            + F.cross_entropy(multi_logits, labels)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = multi_logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

    acc = correct / total * 100.0 if total else 0.0
    logger.info(f'Train Loss: {loss_meter.avg:.4f}, Train Acc: {acc:.2f}%')
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for rgb_clips, flow_clips, labels in tqdm(dataloader, desc='Evaluating'):
        rgb_clips = rgb_clips.to(device, non_blocking=True)
        flow_clips = flow_clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        rgb_f, flow_f = _video_level_features(model, rgb_clips, flow_clips)
        logits = model.multi_classifier(torch.cat([rgb_f, flow_f], dim=-1))
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100.0 if total else 0.0


def main():
    parser = argparse.ArgumentParser(description='MC-TTA source pre-training')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--source_domain', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--num_clips', type=int, default=None)
    parser.add_argument('--clip_len', type=int, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--i3d_rgb_pretrained', type=str, default=None)
    parser.add_argument('--i3d_flow_pretrained', type=str, default=None)
    parser.add_argument('--early_stop_patience', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        alt = os.path.join(os.path.dirname(__file__), '..', args.config)
        cfg_path = alt if os.path.isfile(alt) else args.config

    cfg = load_config(cfg_path) if os.path.isfile(cfg_path) else {}
    m = cfg.get('model', {})
    dcfg = cfg.get('data', {})
    pcfg = cfg.get('pretrain', {})

    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    set_seed(seed)
    logger = get_logger('pretrain')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    num_clips = args.num_clips if args.num_clips is not None else dcfg.get('num_clips', 10)
    clip_len = args.clip_len if args.clip_len is not None else dcfg.get('clip_len', 8)
    resolution = dcfg.get('resolution', 224)
    flow_method = dcfg.get('flow_method', 'tvl1')
    epochs = args.epochs if args.epochs is not None else pcfg.get('epochs', 30)
    batch_size = args.batch_size if args.batch_size is not None else pcfg.get('batch_size', 8)
    lr = args.lr if args.lr is not None else pcfg.get('lr', 0.01)
    wd = args.weight_decay if args.weight_decay is not None else pcfg.get('weight_decay', 1e-4)
    early_stop = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else pcfg.get('early_stop_patience', 5)
    )

    video_list, label_list, class_to_idx = build_dataset_from_directory(
        args.data_root, args.source_domain, 'train'
    )
    num_classes = (
        args.num_classes if args.num_classes is not None else len(class_to_idx)
    )

    logger.info(f'Source domain: {args.source_domain}, classes: {num_classes}, videos: {len(video_list)}')

    train_ds = VideoDataset(
        video_list,
        label_list,
        class_to_idx,
        num_clips=num_clips,
        clip_len=clip_len,
        resolution=resolution,
        flow_method=flow_method,
        is_train=True,
    )
    test_videos, test_labels, _ = build_dataset_from_directory(
        args.data_root,
        args.source_domain,
        'test',
        class_names=list(class_to_idx.keys()),
    )
    test_ds = VideoDataset(
        test_videos,
        test_labels,
        class_to_idx,
        num_clips=num_clips,
        clip_len=clip_len,
        resolution=resolution,
        flow_method=flow_method,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    dropout = m.get('dropout', 0.5)
    model = SourceModel(
        num_classes=num_classes,
        feature_dim=m.get('feature_dim', 1024),
        hidden_dim=m.get('hidden_dim', 1024),
        dropout=dropout,
    ).to(device)

    if args.i3d_rgb_pretrained:
        miss, unexp = load_i3d_weights(model.rgb_backbone, args.i3d_rgb_pretrained, device)
        logger.info(f'RGB I3D loaded: missing={len(miss)}, unexpected={len(unexp)}')
    if args.i3d_flow_pretrained:
        miss, unexp = load_i3d_weights(model.flow_backbone, args.i3d_flow_pretrained, device)
        logger.info(f'Flow I3D loaded: missing={len(miss)}, unexpected={len(unexp)}')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best_acc = 0.0
    stall = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_acc = ckpt.get('best_acc', 0.0)
        logger.info(f'Resumed epoch {start_epoch - 1}, best_acc={best_acc:.2f}%')

    for epoch in range(start_epoch, epochs + 1):
        logger.info(f'--- Epoch {epoch}/{epochs} ---')
        train_one_epoch(model, train_loader, optimizer, device, logger)
        test_acc = evaluate(model, test_loader, device)
        logger.info(f'Test Acc: {test_acc:.2f}%')
        scheduler.step()

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'class_to_idx': class_to_idx,
                'num_classes': num_classes,
                'best_acc': best_acc,
                'config_path': cfg_path,
            },
            os.path.join(args.output_dir, f'source_{args.source_domain}_latest.pth'),
        )

        if test_acc > best_acc:
            best_acc = test_acc
            stall = 0
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'class_to_idx': class_to_idx,
                    'num_classes': num_classes,
                    'best_acc': best_acc,
                    'config_path': cfg_path,
                },
                os.path.join(args.output_dir, f'source_{args.source_domain}_best.pth'),
            )
            logger.info(f'Saved best ({best_acc:.2f}%)')
        else:
            stall += 1
            if stall >= early_stop:
                logger.info(f'Early stop after {early_stop} epochs without improvement')
                break

    logger.info(f'Best source test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
