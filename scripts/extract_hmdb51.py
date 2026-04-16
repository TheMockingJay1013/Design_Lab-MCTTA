"""
Extract HMDB51 videos from HuggingFace cache to class folders.

Usage:
    python scripts/extract_hmdb51.py --output_dir ./downloads/hmdb51_org
"""

import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./downloads/hmdb51_org')
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = args.output_dir

    print("Loading HMDB51 from HuggingFace cache...")
    ds = load_dataset('mteb/hmdb51', trust_remote_code=True)

    class_names = ds['train'].features['label'].names
    print(f"Classes: {len(class_names)}")

    total_saved = 0

    for split_name in ds:
        split_data = ds[split_name]
        print(f"\nProcessing '{split_name}' split: {len(split_data)} videos")

        for i in range(len(split_data)):
            sample = split_data[i]
            label_idx = sample['label']
            class_name = class_names[label_idx]
            video_reader = sample['video']

            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            dst_path = os.path.join(class_dir, f"{split_name}_{i:05d}.avi")
            if os.path.exists(dst_path):
                total_saved += 1
                continue

            try:
                # Get all frames from the video reader
                frames = []
                for frame_idx in range(len(video_reader)):
                    frame = video_reader[frame_idx].asnumpy()
                    frames.append(frame)

                if len(frames) == 0:
                    continue

                # Save as AVI using OpenCV
                import cv2
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(dst_path, fourcc, 30.0, (w, h))
                for frame in frames:
                    # RGB to BGR for OpenCV
                    writer.write(frame[:, :, ::-1])
                writer.release()
                total_saved += 1

            except Exception as e:
                print(f"  Error on {split_name}[{i}]: {e}")
                continue

            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(split_data)} videos...")

    print(f"\nDone! Saved {total_saved} videos to {output_dir}")

    # Verify
    class_dirs = [d for d in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, d))]
    total_files = 0
    for cd in sorted(class_dirs):
        n = len([f for f in os.listdir(os.path.join(output_dir, cd))
                 if f.endswith(('.avi', '.mp4'))])
        total_files += n
    print(f"Class folders: {len(class_dirs)}, Total videos: {total_files}")


if __name__ == '__main__':
    main()
