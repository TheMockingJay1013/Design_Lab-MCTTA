"""
Download HMDB51 videos from Hugging Face and organize into class folders.

Usage:
    pip install datasets
    python scripts/download_hmdb51.py --output_dir ./downloads/hmdb51_org
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./downloads/hmdb51_org')
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading HMDB51 from Hugging Face (Samoed/hmdb51)...")
    print("This may take a while (~2 GB download)...\n")

    try:
        ds = load_dataset("Samoed/hmdb51", trust_remote_code=True)
    except Exception as e:
        print(f"Samoed/hmdb51 failed: {e}")
        print("Trying alternative: mteb/hmdb51...")
        try:
            ds = load_dataset("mteb/hmdb51", trust_remote_code=True)
        except Exception as e2:
            print(f"mteb/hmdb51 also failed: {e2}")
            print("\nPlease download manually from:")
            print("  https://huggingface.co/datasets/Samoed/hmdb51")
            print("  or https://www.kaggle.com/datasets/fengzhongyouxia/hmdb51")
            sys.exit(1)

    # Figure out the splits and label info
    print(f"Dataset splits: {list(ds.keys())}")
    for split_name in ds:
        print(f"  {split_name}: {len(ds[split_name])} samples")
        print(f"  Columns: {ds[split_name].column_names}")
        break

    total_saved = 0

    for split_name in ds:
        split_data = ds[split_name]
        columns = split_data.column_names

        # Detect column names (varies by dataset version)
        video_col = None
        label_col = None
        for c in columns:
            if 'video' in c.lower() or 'path' in c.lower() or 'file' in c.lower():
                video_col = c
            if 'label' in c.lower() or 'class' in c.lower() or 'action' in c.lower():
                label_col = c

        if video_col is None:
            video_col = columns[0]
        if label_col is None:
            label_col = columns[-1]

        print(f"\nProcessing split '{split_name}': video='{video_col}', label='{label_col}'")

        # Get class names
        if hasattr(split_data.features[label_col], 'names'):
            class_names = split_data.features[label_col].names
        else:
            class_names = None

        for i, sample in enumerate(split_data):
            # Get label/class name
            label = sample[label_col]
            if class_names and isinstance(label, int):
                class_name = class_names[label]
            elif isinstance(label, str):
                class_name = label
            else:
                class_name = str(label)

            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Save video
            video_data = sample[video_col]

            if isinstance(video_data, dict) and 'path' in video_data:
                # Video stored as file reference
                src_path = video_data.get('path', '')
                ext = os.path.splitext(src_path)[1] if src_path else '.avi'
                dst = os.path.join(class_dir, f"video_{i:05d}{ext}")
                if 'bytes' in video_data and video_data['bytes'] is not None:
                    with open(dst, 'wb') as f:
                        f.write(video_data['bytes'])
                    total_saved += 1
                elif os.path.exists(src_path):
                    import shutil
                    shutil.copy2(src_path, dst)
                    total_saved += 1
            elif isinstance(video_data, bytes):
                dst = os.path.join(class_dir, f"video_{i:05d}.avi")
                with open(dst, 'wb') as f:
                    f.write(video_data)
                total_saved += 1
            elif isinstance(video_data, str) and os.path.exists(video_data):
                import shutil
                dst = os.path.join(class_dir, os.path.basename(video_data))
                shutil.copy2(video_data, dst)
                total_saved += 1

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1} videos...")

    print(f"\nDone! Saved {total_saved} videos to {output_dir}")

    # Verify
    class_dirs = [d for d in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, d))]
    print(f"Class folders: {len(class_dirs)}")
    for cd in sorted(class_dirs)[:5]:
        n = len(os.listdir(os.path.join(output_dir, cd)))
        print(f"  {cd}: {n} videos")
    if len(class_dirs) > 5:
        print(f"  ... and {len(class_dirs)-5} more classes")


if __name__ == '__main__':
    main()
