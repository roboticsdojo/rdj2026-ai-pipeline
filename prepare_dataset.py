"""
prepare_dataset.py

Takes raw captured/downloaded images sorted into:
    dataset/raw/organic/
    dataset/raw/metals/
    dataset/raw/plastics/

and produces a clean, split, augmented dataset at:
    dataset/final/train/<class>/...
    dataset/final/val/<class>/...
    dataset/final/test/<class>/...

Steps performed:
    1. Validate every image opens correctly (drop corrupted files).
    2. Remove exact duplicate images (by file hash).
    3. Resize all images to a consistent size.
    4. Split into train/val/test (default 70/20/10).
    5. Augment ONLY the training set (rotation, flip, brightness) to
       multiply the number of training examples.

REQUIREMENTS:
    pip install opencv-python pillow numpy albumentations

USAGE:
    python3 prepare_dataset.py --raw dataset/raw --out dataset/final --augment 3

    --augment 3   means: for every real training image, generate 3
                  extra augmented versions (so 100 real images -> 400
                  total training images). Set to 0 to skip augmentation.
"""

import argparse
import hashlib
import os
import random
import shutil

import cv2

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

VALID_CLASSES = ["organic", "metals", "plastics"]
IMG_SIZE = (224, 224)  # standard input size for lightweight models like MobileNetV2


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_and_validate_images(raw_class_dir: str):
    """Return a list of valid, de-duplicated image paths for one class."""
    valid_paths = []
    seen_hashes = set()
    dropped_corrupt = 0
    dropped_dupe = 0

    for fname in os.listdir(raw_class_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(raw_class_dir, fname)

        img = cv2.imread(fpath)
        if img is None:
            dropped_corrupt += 1
            continue

        h = file_hash(fpath)
        if h in seen_hashes:
            dropped_dupe += 1
            continue
        seen_hashes.add(h)
        valid_paths.append(fpath)

    print(f"  -> {len(valid_paths)} valid images "
          f"({dropped_corrupt} corrupted dropped, {dropped_dupe} duplicates dropped)")
    return valid_paths


def split_paths(paths, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(paths)
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }


def save_resized(src_path: str, dst_path: str):
    img = cv2.imread(src_path)
    img = cv2.resize(img, IMG_SIZE)
    cv2.imwrite(dst_path, img)


def get_augmentation_pipeline():
    return A.Compose([
        A.Rotate(limit=25, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    ])


def augment_and_save(src_path: str, out_dir: str, class_name: str, base_idx: int, num_augments: int, pipeline):
    img = cv2.imread(src_path)
    img = cv2.resize(img, IMG_SIZE)
    for i in range(num_augments):
        augmented = pipeline(image=img)["image"]
        out_name = f"{class_name}_aug_{base_idx:04d}_{i}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), augmented)


def main():
    parser = argparse.ArgumentParser(description="Clean, split, and augment the waste dataset.")
    parser.add_argument("--raw", default="dataset/raw", help="Path to raw sorted images (dataset/raw/<class>/)")
    parser.add_argument("--out", default="dataset/final", help="Output path for the prepared dataset")
    parser.add_argument("--augment", type=int, default=3, help="Number of augmented copies per training image (0 = skip)")
    args = parser.parse_args()

    if args.augment > 0 and not HAS_ALBUMENTATIONS:
        print("WARNING: albumentations is not installed, so augmentation will be skipped.")
        print("Install it with: pip install albumentations --break-system-packages")
        args.augment = 0

    pipeline = get_augmentation_pipeline() if args.augment > 0 else None

    for split in ["train", "val", "test"]:
        for cls in VALID_CLASSES:
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    summary = {}

    for cls in VALID_CLASSES:
        raw_class_dir = os.path.join(args.raw, cls)
        if not os.path.isdir(raw_class_dir):
            print(f"Skipping '{cls}': no folder found at {raw_class_dir}")
            continue

        print(f"\nProcessing class: {cls}")
        valid_paths = load_and_validate_images(raw_class_dir)
        if not valid_paths:
            print(f"  No valid images found for '{cls}', skipping.")
            continue

        splits = split_paths(valid_paths)
        summary[cls] = {k: len(v) for k, v in splits.items()}

        # Copy resized originals into train/val/test
        for split_name, paths in splits.items():
            split_dir = os.path.join(args.out, split_name, cls)
            for idx, p in enumerate(paths):
                out_name = f"{cls}_{split_name}_{idx:04d}.jpg"
                save_resized(p, os.path.join(split_dir, out_name))

        # Augment ONLY the training split
        if args.augment > 0:
            train_dir = os.path.join(args.out, "train", cls)
            print(f"  Augmenting training images x{args.augment}...")
            for idx, p in enumerate(splits["train"]):
                augment_and_save(p, train_dir, cls, idx, args.augment, pipeline)
            summary[cls]["train_after_augment"] = summary[cls]["train"] * (1 + args.augment)

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for cls, counts in summary.items():
        print(f"{cls}: {counts}")
    print(f"\nFinal dataset ready at: {args.out}")


if __name__ == "__main__":
    main()
