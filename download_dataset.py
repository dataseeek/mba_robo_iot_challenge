"""
Download a person-detection dataset from COCO 2017.

Downloads only the specific images needed (not the full 18GB dataset) by:
1. Fetching the COCO 2017 val/train annotations JSON
2. Identifying images WITH person annotations → dataset/PESSOA/
3. Identifying images WITHOUT any person → dataset/NENHUM/
4. Downloading a balanced subset of each

Usage:
  python download_dataset.py                          # 1000 per class from val set
  python download_dataset.py --count 2000 --split train  # 2000 per class from train set
  python download_dataset.py --count 500 --workers 8     # faster with more threads
"""

import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ANNO_URLS = {
    "val": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
ANNO_FILES = {
    "val": "annotations/instances_val2017.json",
    "train": "annotations/instances_train2017.json",
}
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{filename}"

PERSON_CATEGORY_ID = 1  # COCO category ID for "person"


def download_file(url: str, dest: str):
    """Download a file with progress indication."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    tmp = dest + ".tmp"

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
    print()
    os.rename(tmp, dest)


def get_annotations(cache_dir: str, split: str) -> dict:
    """Download and extract COCO annotations, return parsed JSON."""
    os.makedirs(cache_dir, exist_ok=True)
    anno_file = ANNO_FILES[split]
    anno_path = os.path.join(cache_dir, anno_file)

    if not os.path.exists(anno_path):
        zip_path = os.path.join(cache_dir, "annotations_trainval2017.zip")
        download_file(ANNO_URLS["val"], zip_path)

        print("  Extracting annotations...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only the annotation files we need
            members = [m for m in zf.namelist() if m.startswith("annotations/instances_")]
            zf.extractall(cache_dir, members)

    print(f"  Loading {anno_path} ...")
    with open(anno_path, "r") as f:
        return json.load(f)


def split_person_images(annotations: dict):
    """Split image IDs into person vs no-person sets."""
    # Find all image IDs that contain at least one person annotation
    person_image_ids = set()
    for ann in annotations["annotations"]:
        if ann["category_id"] == PERSON_CATEGORY_ID:
            person_image_ids.add(ann["image_id"])

    all_image_ids = {img["id"] for img in annotations["images"]}
    no_person_image_ids = all_image_ids - person_image_ids

    # Build id→filename mapping
    id_to_info = {img["id"]: img for img in annotations["images"]}

    person_images = [id_to_info[i] for i in person_image_ids]
    no_person_images = [id_to_info[i] for i in no_person_image_ids]

    return person_images, no_person_images


def download_image(url: str, dest: str) -> bool:
    """Download a single image. Returns True on success."""
    if os.path.exists(dest):
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"\n  WARN: Failed to download {url}: {e}")
        return False


def download_images(images: list[dict], dest_dir: str, split: str, workers: int):
    """Download images in parallel with progress."""
    os.makedirs(dest_dir, exist_ok=True)

    # Check how many already exist
    existing = sum(1 for img in images if os.path.exists(os.path.join(dest_dir, img["file_name"])))
    if existing == len(images):
        print(f"  All {len(images)} images already downloaded in {dest_dir}")
        return

    print(f"  Downloading {len(images)} images to {dest_dir} ({existing} already exist)...")
    completed = existing
    failed = 0

    split_name = "val2017" if split == "val" else "train2017"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for img in images:
            dest = os.path.join(dest_dir, img["file_name"])
            if os.path.exists(dest):
                continue
            url = IMAGE_URL_TEMPLATE.format(split=split_name, filename=img["file_name"])
            futures[pool.submit(download_image, url, dest)] = img["file_name"]

        for future in as_completed(futures):
            if future.result():
                completed += 1
            else:
                failed += 1
            total = len(images)
            print(f"\r  Progress: {completed}/{total} downloaded, {failed} failed", end="", flush=True)

    print(f"\n  Done: {completed} images in {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download COCO person-detection dataset")
    parser.add_argument(
        "--output", default="./dataset",
        help="Output directory (default: ./dataset)",
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Number of images per class (default: 1000)",
    )
    parser.add_argument(
        "--split", choices=["val", "train"], default="val",
        help="COCO split to use (default: val — 5k images, smaller download)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--cache", default="./.coco_cache",
        help="Directory to cache annotation files (default: ./.coco_cache)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Get annotations
    print(f"\n[1/3] Fetching COCO 2017 {args.split} annotations...")
    annotations = get_annotations(args.cache, args.split)

    # Step 2: Split into person / no-person
    print("\n[2/3] Identifying person vs no-person images...")
    person_images, no_person_images = split_person_images(annotations)
    print(f"  Total images:     {len(person_images) + len(no_person_images)}")
    print(f"  With person:      {len(person_images)}")
    print(f"  Without person:   {len(no_person_images)}")

    # Sample balanced subsets
    count = args.count
    if count > len(person_images):
        print(f"\n  WARNING: Requested {count} person images but only {len(person_images)} available.")
        count = min(len(person_images), len(no_person_images))
        print(f"  Using {count} per class instead.")

    if count > len(no_person_images):
        print(f"\n  WARNING: Requested {count} no-person images but only {len(no_person_images)} available.")
        count = min(count, len(no_person_images))
        print(f"  Using {count} per class instead.")

    person_sample = random.sample(person_images, count)
    no_person_sample = random.sample(no_person_images, count)

    # Step 3: Download images
    print(f"\n[3/3] Downloading {count} images per class ({count * 2} total)...")
    pessoa_dir = os.path.join(args.output, "PESSOA")
    nenhum_dir = os.path.join(args.output, "NENHUM")

    download_images(person_sample, pessoa_dir, args.split, args.workers)
    download_images(no_person_sample, nenhum_dir, args.split, args.workers)

    print(f"\n{'='*50}")
    print(f"Dataset ready at: {args.output}")
    print(f"  PESSOA/ : {len(os.listdir(pessoa_dir))} images")
    print(f"  NENHUM/ : {len(os.listdir(nenhum_dir))} images")
    print(f"\nNext step:")
    print(f"  python train.py --dataset {args.output} --output ./output")
    print()


if __name__ == "__main__":
    main()
