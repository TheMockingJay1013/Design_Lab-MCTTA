"""
Print dataset statistics vs paper §4.1 (video counts / splits).

Usage:
  python scripts/verify_dataset.py --data_root /path/to/data --domain ucf --split train
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.video_dataset import build_dataset_from_directory

# Reference counts from paper §4.1 (for manual comparison; paths must match your layout).
REFERENCE = {
    'ucf_olympic_ucf_train': 432,
    'ucf_olympic_ucf_test': 168,
    'ucf_olympic_olympic_train': 260,
    'ucf_olympic_olympic_test': 55,
    'ucf_hmdb_small_ucf_train': 432,
    'ucf_hmdb_small_ucf_test': 168,
    'ucf_hmdb_small_hmdb_train': 482,
    'ucf_hmdb_small_hmdb_test': 189,
    'ucf_hmdb_full_ucf_train': 1438,
    'ucf_hmdb_full_ucf_test': 571,
    'ucf_hmdb_full_hmdb_train': 840,
    'ucf_hmdb_full_hmdb_test': 360,
}


def main():
    parser = argparse.ArgumentParser(description='Verify dataset layout and counts')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True, help='e.g. ucf, hmdb, olympic')
    parser.add_argument('--split', type=str, default='train', choices=('train', 'test'))
    args = parser.parse_args()

    videos, labels, class_to_idx = build_dataset_from_directory(
        args.data_root, args.domain, args.split
    )
    n_cls = len(class_to_idx)
    print(f'data_root={args.data_root}')
    print(f'domain={args.domain} split={args.split}')
    print(f'num_classes={n_cls} num_videos={len(videos)}')
    print('classes:', list(class_to_idx.keys()))

    key = f'{args.domain}_{args.split}'
    ref_note = ' (compare to paper §4.1 if using official UCF-HMDB / Olympic splits)'
    print(f'\nReference keys in script: UCF-Olympic / UCF-HMDB small / full counts in REFERENCE dict{ref_note}')
    print(f'Current key hint: {args.domain}_{args.split}_videos={len(videos)}')

    from collections import Counter

    cnt = Counter(labels)
    print('\nPer-class video counts:')
    idx_to_name = {i: n for n, i in class_to_idx.items()}
    for lab in sorted(cnt.keys()):
        print(f'  {idx_to_name[lab]}: {cnt[lab]}')


if __name__ == '__main__':
    main()
