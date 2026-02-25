"""
sample_dataset.py

Downloads a random subset of images and saves them locally so
generate_concept_labels.py can process them.

COCO is downloaded directly from cocodataset.org (avoids broken HF mirrors).
Other datasets are loaded via HuggingFace.

Usage:
    # COCO val — 5000 random images (recommended)
    python sample_dataset.py --dataset coco --n 5000 --output sampled_images

    # Flickr30K — 2000 random images
    python sample_dataset.py --dataset flickr30k --n 2000 --output sampled_images

    # Visual Genome — 3000 images (streaming)
    python sample_dataset.py --dataset visual_genome --n 3000 --output sampled_images --stream
"""

import argparse
import random
import json
import io
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image


# ── Dataset registry ───────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "coco": {
        "source": "direct",
        "zip_url": "http://images.cocodataset.org/zips/val2017.zip",
        "approx_size": 5_000,
        "notes": "COCO 2017 val — 5K images, ~778 MB zip. Downloaded from cocodataset.org.",
    },
    "flickr30k": {
        "source": "huggingface",
        "hf_name": "nlphuji/flickr30k",
        "hf_split": "test",
        "image_key": "image",
        "approx_size": 31_783,
        "notes": "31K diverse real-world photos.",
    },
    "visual_genome": {
        "source": "huggingface",
        "hf_name": "visual_genome",
        "hf_config": "region_descriptions_v1.2.0",
        "hf_split": "train",
        "image_key": "image",
        "approx_size": 108_249,
        "notes": "108K images with dense region annotations. Use --stream.",
    },
    "laion_lvis": {
        "source": "huggingface",
        "hf_name": "laion/220k-GPT4Vision-captions-from-LIVIS",
        "hf_split": "train",
        "image_key": "image",
        "approx_size": 220_000,
        "notes": "220K high-quality GPT-4V captioned images. Use --stream.",
    },
}


def list_datasets():
    print("\nAvailable datasets:")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"\n  [{name}]")
        print(f"    ~Size  : {cfg['approx_size']:,}")
        print(f"    Notes  : {cfg['notes']}")


# ── COCO direct download ───────────────────────────────────────────────────────

def download_coco_direct(n, output_dir, seed, fmt, manifest_path):
    """
    Download val2017.zip from cocodataset.org, extract a random subset of
    n images into output_dir without keeping the full zip on disk permanently.
    """
    url = DATASET_CONFIGS["coco"]["zip_url"]
    zip_cache = output_dir.parent / "val2017.zip"

    # Download if not already cached
    if not zip_cache.exists():
        print(f"Downloading COCO val2017 (~778 MB) from {url}")
        print(f"Saving to: {zip_cache}  (cached for future runs)\n")

        def progress(count, block_size, total):
            mb_done = count * block_size / 1e6
            mb_total = total / 1e6
            pct = min(100, 100 * mb_done / mb_total)
            print(f"  {mb_done:.1f} / {mb_total:.1f} MB  ({pct:.1f}%)", end="\r")

        urllib.request.urlretrieve(url, zip_cache, reporthook=progress)
        print(f"\nDownload complete.")
    else:
        print(f"Using cached zip: {zip_cache}")

    # Get list of image names inside the zip
    print("Reading zip index...")
    with zipfile.ZipFile(zip_cache, "r") as zf:
        all_names = [
            name for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png")) and not name.endswith("/")
        ]

    print(f"Found {len(all_names):,} images in zip")

    rng = random.Random(seed)
    if n >= len(all_names):
        print(f"  Requested {n} but zip only has {len(all_names)} — using all.")
        selected = all_names
    else:
        selected = rng.sample(all_names, n)
    selected.sort()

    # Extract selected images
    print(f"Extracting {len(selected)} images to {output_dir}/ ...")
    manifest = []
    saved = 0
    errors = 0

    with zipfile.ZipFile(zip_cache, "r") as zf:
        for i, name in enumerate(selected):
            out_name = f"coco_{i:05d}.{fmt}"
            out_path = output_dir / out_name
            try:
                data = zf.read(name)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                save_kwargs = {"quality": 92, "optimize": True} if fmt == "jpg" else {}
                img.save(out_path, **save_kwargs)
                saved += 1
                manifest.append({
                    "file": str(out_path),
                    "index": i,
                    "original_name": Path(name).name,
                })
            except Exception as e:
                errors += 1
                print(f"  [warn] Skipped {name}: {e}")

            if (i + 1) % 200 == 0 or (i + 1) == len(selected):
                print(f"  {i+1}/{len(selected)}  saved={saved}  errors={errors}", end="\r")

    print()
    _save_manifest(manifest_path, "cocodataset.org/val2017", "val", n, saved, seed, manifest)
    return saved


# ── HuggingFace download ───────────────────────────────────────────────────────

def reservoir_sample(iterator, n, seed=42):
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(iterator):
        if i < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = item
    return reservoir


def random_indices_sample(dataset, n, seed=42):
    rng = random.Random(seed)
    total = len(dataset)
    if n >= total:
        print(f"  Requested {n} but dataset only has {total} — using all.")
        return list(range(total))
    return sorted(rng.sample(range(total), n))


def extract_image(row, image_key):
    img = row[image_key]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    raise ValueError(f"Unrecognised image type: {type(img)}")


def download_huggingface(cfg, split, n, output_dir, seed, fmt, stream, manifest_path, dataset_name):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install with:  pip install datasets pillow")
        return 0

    load_kwargs = {}
    if "hf_config" in cfg:
        load_kwargs["name"] = cfg["hf_config"]

    if stream:
        print("Loading dataset in streaming mode...")
        ds = load_dataset(cfg["hf_name"], split=split, streaming=True, **load_kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        print("Reservoir sampling (streaming through dataset)...")
        rows = reservoir_sample(ds, n, seed=seed)
    else:
        print("Downloading dataset (cached by HuggingFace after first run)...")
        ds = load_dataset(cfg["hf_name"], split=split, **load_kwargs)
        print(f"Dataset loaded: {len(ds):,} total images")
        indices = random_indices_sample(ds, n, seed=seed)
        rows = [ds[i] for i in indices]

    image_key = cfg["image_key"]
    manifest = []
    saved = 0
    errors = 0

    print(f"\nSaving {len(rows)} images...")
    for i, row in enumerate(rows):
        out_name = f"{dataset_name}_{i:05d}.{fmt}"
        out_path = output_dir / out_name
        try:
            img = extract_image(row, image_key)
            save_kwargs = {"quality": 92, "optimize": True} if fmt == "jpg" else {}
            img.save(out_path, **save_kwargs)
            saved += 1
            entry = {"file": str(out_path), "index": i}
            for meta_key in ("file_name", "caption", "id", "image_id", "url"):
                if meta_key in row and not isinstance(row[meta_key], (Image.Image, bytes)):
                    entry[meta_key] = row[meta_key]
            manifest.append(entry)
        except Exception as e:
            errors += 1
            print(f"  [warn] Skipped image {i}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"  {i+1}/{len(rows)}  saved={saved}  errors={errors}", end="\r")

    print()
    _save_manifest(manifest_path, cfg["hf_name"], split, n, saved, seed, manifest)
    return saved


# ── Shared ────────────────────────────────────────────────────────────────────

def _save_manifest(path, source, split, n_requested, n_saved, seed, images):
    with open(path, "w") as f:
        json.dump({
            "source": source,
            "split": split,
            "n_requested": n_requested,
            "n_saved": n_saved,
            "seed": seed,
            "images": images,
        }, f, indent=2)
    print(f"Manifest written to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sample a random subset of images for concept labeling")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), help="Dataset to use (see --list)")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument("--n", type=int, default=1000, help="Number of images to sample")
    parser.add_argument("--split", default=None, help="Override default split")
    parser.add_argument("--output", default="sampled_images", help="Output directory for images")
    parser.add_argument("--stream", action="store_true", help="HuggingFace streaming mode (large datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--manifest", default="sample_manifest.json")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset is None:
        parser.error("Provide --dataset or use --list to see options.")

    cfg = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest

    print(f"Dataset : {args.dataset}  ({cfg['notes']})")
    print(f"Sampling: {args.n} images  (seed={args.seed})")
    print(f"Output  : {output_dir}/\n")

    if cfg["source"] == "direct":
        saved = download_coco_direct(args.n, output_dir, args.seed, args.format, manifest_path)
    else:
        split = args.split or cfg["hf_split"]
        saved = download_huggingface(cfg, split, args.n, output_dir, args.seed,
                                     args.format, args.stream, manifest_path, args.dataset)

    print(f"\n{'='*50}")
    print(f"Done! Saved {saved}/{args.n} images to: {output_dir}/")
    print(f"\nNext step:")
    print(f"  python generate_concept_labels.py \\")
    print(f"      --images_dir {output_dir} \\")
    print(f"      --output concept_labels.json \\")
    print(f"      --layer 20")


if __name__ == "__main__":
    main()
