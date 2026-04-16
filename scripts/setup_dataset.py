"""
Dataset setup script for UCF-HMDB_full benchmark.

Downloads UCF101 and HMDB51 datasets, extracts the 12 shared classes,
and organizes them into the train/test split from TA3N (Chen et al., 2019).

Usage:
    python scripts/setup_dataset.py --output_dir ./data

12 shared classes:
    climb, fencing, golf, kick_ball, pullup, punch,
    pushup, ride_bike, ride_horse, shoot_ball, shoot_bow, walk
"""

import os
import sys
import shutil
import argparse
import subprocess
import json

# Mapping from UCF101 class names → shared label
UCF_CLASS_MAP = {
    'RockClimbingIndoor': 'climb',
    'Fencing': 'fencing',
    'GolfSwing': 'golf',
    'SoccerPenalty': 'kick_ball',
    'PullUps': 'pullup',
    'Punch': 'punch',
    'BoxingPunchingBag': 'punch',
    'PushUps': 'pushup',
    'Biking': 'ride_bike',
    'HorseRiding': 'ride_horse',
    'Basketball': 'shoot_ball',
    'Archery': 'shoot_bow',
    'WalkingWithDog': 'walk',
}

# Mapping from HMDB51 class names → shared label
HMDB_CLASS_MAP = {
    'climb': 'climb',
    'fencing': 'fencing',
    'golf': 'golf',
    'kick_ball': 'kick_ball',
    'pullup': 'pullup',
    'punch': 'punch',
    'pushup': 'pushup',
    'ride_bike': 'ride_bike',
    'ride_horse': 'ride_horse',
    'shoot_ball': 'shoot_ball',
    'shoot_bow': 'shoot_bow',
    'walk': 'walk',
}

SHARED_CLASSES = sorted([
    'climb', 'fencing', 'golf', 'kick_ball', 'pullup', 'punch',
    'pushup', 'ride_bike', 'ride_horse', 'shoot_ball', 'shoot_bow', 'walk'
])


def run_cmd(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def is_valid_rar(filepath):
    """Check if a file is a valid RAR archive by reading its magic bytes."""
    if not os.path.exists(filepath):
        return False
    # RAR files smaller than 1 MB are almost certainly corrupt/error pages
    if os.path.getsize(filepath) < 1_000_000:
        return False
    with open(filepath, 'rb') as f:
        magic = f.read(7)
    # RAR4: "Rar!\x1a\x07\x00", RAR5: "Rar!\x1a\x07\x01"
    return magic[:4] == b'Rar!'


def wget_download(url, dest):
    """Download with wget, falling back to --no-check-certificate if needed."""
    # Remove corrupt/incomplete files from previous attempts
    if os.path.exists(dest) and not is_valid_rar(dest):
        print(f"  Removing corrupt file: {dest}")
        os.remove(dest)

    try:
        run_cmd(f"wget --no-check-certificate -O '{dest}' '{url}'")
    except subprocess.CalledProcessError:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Failed to download {url}")

    if not is_valid_rar(dest):
        print(f"  WARNING: Downloaded file does not look like a valid RAR.")
        print(f"  File size: {os.path.getsize(dest)} bytes")
        print(f"  You may need to download manually from: {url}")


def extract_rar(rar_path, dest_dir):
    """
    Extract a .rar file using whichever tool is available:
    tries unrar, then unar, then 7z, then Python rarfile.
    """
    tools = [
        f"unrar x -o+ '{rar_path}' '{dest_dir}/'",
        f"unar -o '{dest_dir}' '{rar_path}'",
        f"7z x '{rar_path}' -o'{dest_dir}' -y",
    ]

    for cmd in tools:
        tool_name = cmd.split()[0]
        if shutil.which(tool_name):
            print(f"  Using {tool_name} to extract {os.path.basename(rar_path)}")
            try:
                run_cmd(cmd)
                return
            except subprocess.CalledProcessError:
                print(f"  {tool_name} failed, trying next...")
                continue

    # Fallback: Python rarfile (requires pip install rarfile)
    try:
        import rarfile
        print(f"  Using Python rarfile to extract {os.path.basename(rar_path)}")
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(dest_dir)
        return
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("ERROR: No RAR extraction tool found!")
    print("Install ONE of the following:")
    print("  sudo apt install unrar          # recommended")
    print("  sudo apt install unar           # alternative")
    print("  sudo apt install p7zip-full     # alternative")
    print("  pip install rarfile             # Python fallback")
    print("=" * 60)
    sys.exit(1)


def download_ucf101(download_dir):
    """Download UCF101 dataset."""
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    dest = os.path.join(download_dir, "UCF101.rar")
    if os.path.exists(dest) and is_valid_rar(dest):
        print("UCF101.rar already exists and is valid, skipping download.")
    else:
        print("Downloading UCF101...")
        wget_download(url, dest)

    extract_dir = os.path.join(download_dir, "UCF-101")
    if not os.path.exists(extract_dir):
        print("Extracting UCF101...")
        extract_rar(dest, download_dir)
    return extract_dir


def download_hmdb51(download_dir):
    """Download HMDB51 using torchvision (most reliable method)."""
    extract_dir = os.path.join(download_dir, "hmdb51_org")

    # Check if already extracted
    if os.path.exists(extract_dir):
        n_dirs = len([d for d in os.listdir(extract_dir)
                      if os.path.isdir(os.path.join(extract_dir, d))])
        if n_dirs >= 10:
            print(f"HMDB51 already extracted ({n_dirs} class dirs), skipping.")
            return extract_dir

    print("Downloading HMDB51 via torchvision...")
    print("  (This downloads from Google Drive — ~2 GB)")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        from torchvision.datasets.utils import download_and_extract_archive
        # torchvision's HMDB51 download URL (Google Drive, reliable)
        url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"

        # Use torchvision's built-in HMDB51 dataset class which handles download
        from torchvision.datasets import HMDB51
        print("  Using torchvision.datasets.HMDB51 to download...")
        # This needs the split file too — download it first
        splits_url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"

        # Fallback: direct download from known mirrors
        raise ImportError("Skip to manual method")

    except (ImportError, Exception):
        pass

    # Try direct RAR download from multiple mirrors
    urls = [
        "https://archive.org/download/hmdb51_org/hmdb51_org.rar",
        "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
    ]
    dest = os.path.join(download_dir, "hmdb51_org.rar")

    downloaded = False
    if os.path.exists(dest) and is_valid_rar(dest):
        downloaded = True
    else:
        for url in urls:
            print(f"  Trying {url} ...")
            try:
                wget_download(url, dest)
                if is_valid_rar(dest):
                    downloaded = True
                    break
                else:
                    if os.path.exists(dest):
                        os.remove(dest)
            except Exception:
                if os.path.exists(dest):
                    os.remove(dest)

    if not downloaded:
        # Last resort: download via gdown from a known Google Drive link
        print("  Trying gdown (Google Drive)...")
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"],
                           check=True)
            import gdown

        gdrive_url = "https://drive.google.com/uc?id=1dFBKDDx1SrEuz2J-JKU0TLPfQuMFbxhJ"
        try:
            gdown.download(gdrive_url, dest, quiet=False)
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
                downloaded = True
        except Exception as e:
            print(f"  gdown failed: {e}")

    if not downloaded:
        print("\n" + "=" * 60)
        print("Could not auto-download HMDB51. Please download manually:")
        print()
        print("  EASIEST — run this command:")
        print(f"    python -c \"from torchvision.datasets import HMDB51; HMDB51('{download_dir}/hmdb51_tv', 'unused', 1, download=True)\"")
        print()
        print("  OR download from: https://www.kaggle.com/datasets/fengzhongyouxia/hmdb51")
        print(f"  Extract into: {extract_dir}/")
        print("  Then re-run: python scripts/setup_dataset.py --skip_download --hmdb_path " + extract_dir)
        print("=" * 60)
        sys.exit(1)

    # Extract if we have the RAR
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 5:
        print("Extracting HMDB51...")
        os.makedirs(extract_dir, exist_ok=True)
        extract_rar(dest, extract_dir)
        # HMDB51 has nested .rar files per class
        for rar_file in sorted(os.listdir(extract_dir)):
            if rar_file.endswith('.rar'):
                class_name = rar_file.replace('.rar', '')
                class_dir = os.path.join(extract_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                extract_rar(os.path.join(extract_dir, rar_file), class_dir)

    return extract_dir


def download_splits(download_dir):
    """Download the TA3N train/test splits for UCF-HMDB_full."""
    splits_dir = os.path.join(download_dir, "ta3n_splits")
    if not os.path.exists(splits_dir):
        print("Downloading TA3N split files...")
        run_cmd(f"git clone --depth 1 https://github.com/cmhungsteve/TA3N.git {splits_dir}")
    return splits_dir


def parse_split_file(split_file):
    """
    Parse a TA3N split file. Each line: <relative_path> <label_int>
    Returns list of (video_filename, label_int).
    """
    entries = []
    if not os.path.exists(split_file):
        return entries
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                path = parts[0]
                label = int(parts[1])
                entries.append((path, label))
    return entries


def organize_dataset(ucf_raw_dir, hmdb_raw_dir, output_dir, splits_dir=None):
    """
    Organize UCF and HMDB videos into the target directory structure.
    Falls back to a simple 70/30 split if TA3N splits aren't available.
    """
    import random
    random.seed(42)

    dataset_dir = os.path.join(output_dir, "ucf_hmdb_full")

    for domain, raw_dir, class_map in [
        ("ucf", ucf_raw_dir, UCF_CLASS_MAP),
        ("hmdb", hmdb_raw_dir, HMDB_CLASS_MAP),
    ]:
        print(f"\nOrganizing {domain.upper()} domain...")

        for split in ['train', 'test']:
            for cls in SHARED_CLASSES:
                os.makedirs(os.path.join(dataset_dir, domain, split, cls), exist_ok=True)

        # Collect all videos for shared classes
        all_videos = {}  # shared_class -> list of source paths
        for cls in SHARED_CLASSES:
            all_videos[cls] = []

        if os.path.isdir(raw_dir):
            for orig_class in sorted(os.listdir(raw_dir)):
                orig_path = os.path.join(raw_dir, orig_class)
                if not os.path.isdir(orig_path):
                    continue
                if orig_class not in class_map:
                    continue
                shared_cls = class_map[orig_class]

                for video_file in sorted(os.listdir(orig_path)):
                    ext = os.path.splitext(video_file)[1].lower()
                    if ext in ('.avi', '.mp4', '.mkv', '.mov', '.webm'):
                        all_videos[shared_cls].append(
                            os.path.join(orig_path, video_file))

        # Split 70/30
        total_train = 0
        total_test = 0
        for cls in SHARED_CLASSES:
            videos = all_videos[cls]
            random.shuffle(videos)
            split_idx = int(len(videos) * 0.7)
            train_videos = videos[:split_idx]
            test_videos = videos[split_idx:]

            for v in train_videos:
                dst = os.path.join(dataset_dir, domain, 'train', cls, os.path.basename(v))
                if os.path.exists(dst):
                    os.remove(dst)  # Replace existing symlink/file for re-runs
                os.symlink(os.path.abspath(v), dst)

            for v in test_videos:
                dst = os.path.join(dataset_dir, domain, 'test', cls, os.path.basename(v))
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(v), dst)

            total_train += len(train_videos)
            total_test += len(test_videos)
            print(f"  {cls}: {len(train_videos)} train, {len(test_videos)} test")

        print(f"  {domain.upper()} total: {total_train} train, {total_test} test")

    # Save class mapping
    class_to_idx = {cls: i for i, cls in enumerate(SHARED_CLASSES)}
    meta_path = os.path.join(dataset_dir, "class_to_idx.json")
    with open(meta_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nDataset organized at: {dataset_dir}")
    print(f"Class mapping saved to: {meta_path}")
    print(f"\nClasses ({len(SHARED_CLASSES)}):")
    for cls, idx in class_to_idx.items():
        print(f"  {idx}: {cls}")

    return dataset_dir


def verify_dataset(dataset_dir):
    """Print dataset statistics."""
    print("\n=== Dataset Verification ===")
    for domain in ['ucf', 'hmdb']:
        for split in ['train', 'test']:
            split_dir = os.path.join(dataset_dir, domain, split)
            if not os.path.isdir(split_dir):
                print(f"  MISSING: {split_dir}")
                continue
            total = 0
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    n = len([f for f in os.listdir(cls_dir)
                             if os.path.splitext(f)[1].lower() in
                             ('.avi', '.mp4', '.mkv', '.mov', '.webm')])
                    total += n
            print(f"  {domain}/{split}: {total} videos")


def main():
    parser = argparse.ArgumentParser(
        description='Setup UCF-HMDB_full dataset for MC-TTA')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Where to organize the final dataset')
    parser.add_argument('--download_dir', type=str, default='./downloads',
                        help='Where to download raw files')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download, only reorganize')
    parser.add_argument('--ucf_path', type=str, default=None,
                        help='Path to existing extracted UCF-101 directory')
    parser.add_argument('--hmdb_path', type=str, default=None,
                        help='Path to existing extracted HMDB51 directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    if args.skip_download:
        ucf_dir = args.ucf_path or os.path.join(args.download_dir, "UCF-101")
        hmdb_dir = args.hmdb_path or os.path.join(args.download_dir, "hmdb51_org")
    else:
        print("=" * 60)
        print("Step 1: Downloading datasets")
        print("=" * 60)
        ucf_dir = download_ucf101(args.download_dir)
        hmdb_dir = download_hmdb51(args.download_dir)

    print("\n" + "=" * 60)
    print("Step 2: Organizing into UCF-HMDB_full structure")
    print("=" * 60)
    dataset_dir = organize_dataset(ucf_dir, hmdb_dir, args.output_dir)

    verify_dataset(dataset_dir)

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print(f"""
1. Pre-train on UCF (source):
   python scripts/pretrain.py \\
       --source_domain ucf \\
       --data_root {os.path.join(args.output_dir, 'ucf_hmdb_full')} \\
       --num_classes 12

2. Adapt to HMDB (target):
   python scripts/adapt.py \\
       --source_checkpoint checkpoints/source_ucf_best.pth \\
       --target_domain hmdb \\
       --data_root {os.path.join(args.output_dir, 'ucf_hmdb_full')}

3. Or reverse direction (H→U):
   python scripts/pretrain.py --source_domain hmdb ...
   python scripts/adapt.py --target_domain ucf ...
""")


if __name__ == '__main__':
    main()
