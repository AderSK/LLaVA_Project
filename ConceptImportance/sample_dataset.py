"""
sample_dataset.py

Downloads a random subset of images and saves them locally so
generate_concept_labels.py / collect_activations.py can process them.

Sources supported:
  - "direct"          : direct .zip download (used for COCO val2017)
  - "huggingface"     : load_dataset(...) — only works for datasets that ship
                        Parquet/Arrow/CSV directly (no Python loading script).
  - "hf_parquet_api"  : fetch the auto-converted Parquet files via the Hub API
                        ( /api/datasets/{name}/parquet/{config}/{split} ) and
                        load them with the parquet builder. This bypasses the
                        deprecated dataset-script loader, so it works for
                        script-based datasets like nlphuji/flickr30k under
                        datasets >= 4.0.

Usage:
    python sample_dataset.py --dataset coco        --n 5000 --output sampled_images
    python sample_dataset.py --dataset flickr30k   --n 2500 --output sampled_images
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
        # Switched from "huggingface" to "hf_parquet_api" because nlphuji/flickr30k
        # is script-based and datasets >= 4.0 dropped script support.
        "source": "hf_parquet_api",
        "hf_name": "nlphuji/flickr30k",
        "hf_config": "TEST",   # Discovered via the parquet API — not "default"
        "hf_split": "test",
        "image_key": "image",
        "approx_size": 31_783,
        "notes": "31K diverse real-world photos. Loaded via Parquet auto-conversion API.",
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
    url = DATASET_CONFIGS["coco"]["zip_url"]
    zip_cache = output_dir.parent / "val2017.zip"

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


# ── HF Parquet-API download (for script-based datasets) ────────────────────────

def _normalize_parquet_url(repo_id, config, split, raw):
    """
    Make sure the URL is a fetchable HTTP(S) URL. We download via urllib +
    pyarrow rather than load_dataset, so we don't need an HfFileSystem-
    parseable form — just a real URL.
    """
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith(("http://", "https://", "hf://")):
        return raw
    # Relative path returned by some Hub API responses → prepend host
    return "https://huggingface.co/" + raw.lstrip("/")


def _http_get_json(url, timeout=30):
    """GET a URL and return parsed JSON, or raise."""
    req = urllib.request.Request(url, headers={"User-Agent": "sample_dataset.py"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _flatten_parquet_response(data):
    """
    Normalize whatever the Hub parquet API returns into a list of
    (config, split, url) tuples.

    Handles every shape we've seen in the wild:
      - nested dict     {config: {split: [urls]}}
      - parquet_files   {"parquet_files": [{"config":..., "split":..., "url":...}]}
      - flat list-of-str    [url, url, ...]      (shape from /parquet/{c}/{s} endpoint)
      - flat list-of-dict   [{"url":..., ...}]
    """
    triples = []

    if isinstance(data, dict):
        if "parquet_files" in data:
            for entry in data["parquet_files"]:
                if "url" in entry:
                    triples.append((entry.get("config", "?"),
                                    entry.get("split", "?"),
                                    entry["url"]))
            return triples

        # Nested {config: {split: [urls]}}
        for cfg_name, splits in data.items():
            if isinstance(splits, dict):
                for split_name, urls in splits.items():
                    if isinstance(urls, list):
                        for u in urls:
                            if isinstance(u, str):
                                triples.append((cfg_name, split_name, u))
                            elif isinstance(u, dict) and "url" in u:
                                triples.append((cfg_name, split_name, u["url"]))
        return triples

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                triples.append(("?", "?", item))
            elif isinstance(item, dict) and "url" in item:
                triples.append((item.get("config", "?"),
                                item.get("split", "?"),
                                item["url"]))
    return triples


def fetch_parquet_urls(repo_id, config, split):
    """
    Discover the parquet shards for a dataset.

    Strategy:
      1. Hit the generic /parquet endpoint (lists every config/split).
      2. Show the user what's actually there.
      3. Match against the requested (config, split) — exact, or unique fallback.
      4. If still ambiguous and the user didn't specify, error with a helpful
         message instead of a silent wrong choice.
    """
    candidate_urls = [
        f"https://huggingface.co/api/datasets/{repo_id}/parquet",
        f"https://datasets-server.huggingface.co/parquet?dataset={repo_id}",
    ]

    triples = []
    last_error = None
    for api_url in candidate_urls:
        print(f"Fetching parquet URL list from: {api_url}")
        try:
            data = _http_get_json(api_url)
        except Exception as e:
            print(f"  [warn] {e}")
            last_error = e
            continue
        triples = _flatten_parquet_response(data)
        if triples:
            break
        print(f"  [warn] response had no recognizable parquet entries")

    if not triples:
        raise RuntimeError(
            f"Could not fetch any parquet URLs for {repo_id}. Last error: {last_error}"
        )

    # Group by (config, split) and report
    by_key = {}
    for c, s, u in triples:
        by_key.setdefault((c, s), []).append(u)

    print(f"  Available configs/splits on parquet branch:")
    for (c, s), urls in by_key.items():
        print(f"    config={c!r:<20} split={s!r:<12} shards={len(urls)}")

    def _norm(c, s, urls):
        out = [_normalize_parquet_url(repo_id, c, s, u) for u in urls]
        out = [u for u in out if u]
        # Show one for sanity
        if out:
            print(f"  example URL: {out[0]}")
        return out

    # 1) Exact match
    if (config, split) in by_key:
        urls = by_key[(config, split)]
        print(f"  → using requested config={config!r} split={split!r} ({len(urls)} shard(s))")
        return _norm(config, split, urls)

    # 2) Unique fallback — only one (config, split) pair available, use it
    if len(by_key) == 1:
        (c, s), urls = next(iter(by_key.items()))
        print(f"  [info] requested {config!r}/{split!r} not found, but only one "
              f"(config, split) is available — falling back to {c!r}/{s!r}")
        return _norm(c, s, urls)

    # 3) Match on split alone if only one config exists
    configs = {c for c, _ in by_key}
    if len(configs) == 1:
        only_config = next(iter(configs))
        for (c, s), urls in by_key.items():
            if s == split:
                print(f"  [info] using config={c!r} (only one available) with requested split={s!r}")
                return _norm(c, s, urls)
        # Or if only one split too
        splits_in_config = [s for c, s in by_key if c == only_config]
        if len(splits_in_config) == 1:
            s = splits_in_config[0]
            urls = by_key[(only_config, s)]
            print(f"  [info] requested split={split!r} not found; only split is {s!r} — using it")
            return _norm(only_config, s, urls)

    raise RuntimeError(
        f"Could not match requested config={config!r} split={split!r} "
        f"to anything on the parquet branch for {repo_id}.\n"
        f"Available: {list(by_key.keys())}\n"
        f"Tip: pass --split <name> to override, or update DATASET_CONFIGS"
        f"['{repo_id.split('/')[-1]}']['hf_config'] / ['hf_split']."
    )


def _download_parquet_shards(urls, cache_dir):
    """Download each URL into cache_dir if not already present. Returns local paths."""
    local_paths = []
    for i, url in enumerate(urls):
        local = cache_dir / f"shard_{i:04d}.parquet"
        if local.exists() and local.stat().st_size > 0:
            mb = local.stat().st_size / 1e6
            print(f"  [{i+1}/{len(urls)}] cached: {local.name}  ({mb:.1f} MB)")
        else:
            print(f"  [{i+1}/{len(urls)}] downloading...")

            def progress(count, block_size, total):
                if total > 0:
                    mb_done = count * block_size / 1e6
                    mb_total = total / 1e6
                    pct = min(100, 100 * mb_done / mb_total)
                    print(f"    {mb_done:.1f} / {mb_total:.1f} MB  ({pct:.1f}%)", end="\r")

            tmp = local.with_suffix(".tmp")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "sample_dataset.py"})
                # urlretrieve doesn't accept a Request object, so use urlopen + write
                with urllib.request.urlopen(req, timeout=600) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    bytes_done = 0
                    chunk_size = 1 << 20  # 1 MiB
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            out.write(chunk)
                            bytes_done += len(chunk)
                            if total:
                                pct = 100 * bytes_done / total
                                print(f"    {bytes_done/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.1f}%)", end="\r")
                            else:
                                print(f"    {bytes_done/1e6:.1f} MB", end="\r")
                tmp.rename(local)
                print(f"\n    done: {local.stat().st_size / 1e6:.1f} MB")
            except Exception:
                if tmp.exists():
                    tmp.unlink()
                raise
        local_paths.append(local)
    return local_paths


def download_hf_parquet_api(cfg, split, n, output_dir, seed, fmt, stream, manifest_path, dataset_name):
    """
    Download a script-based dataset by going around the script (and the
    datasets library) entirely. We:
        1. Fetch the parquet shard URL list from the Hub API.
        2. Download each shard with urllib into a local cache dir.
        3. Read row counts via pyarrow metadata (cheap).
        4. Globally sample n rows uniformly across shards.
        5. Read just the selected rows from each shard and save images.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Install pyarrow:  pip install pyarrow pillow")
        return 0

    config = cfg.get("hf_config", "default")
    urls = fetch_parquet_urls(cfg["hf_name"], config, split)

    # Cache shards next to the output dir (sibling so multiple runs reuse them)
    cache_name = f".parquet_cache_{cfg['hf_name'].replace('/', '_')}_{config}_{split}"
    cache_dir = output_dir.parent / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nParquet cache directory: {cache_dir}/")

    local_paths = _download_parquet_shards(urls, cache_dir)

    # Read metadata to count rows in each shard (no full file read)
    print("\nReading shard metadata...")
    parquet_files = [pq.ParquetFile(p) for p in local_paths]
    row_counts = [pf.metadata.num_rows for pf in parquet_files]
    n_total = sum(row_counts)
    for i, (p, c) in enumerate(zip(local_paths, row_counts)):
        print(f"  shard {i:2d}: {c:>6d} rows  ({p.name})")
    print(f"  total : {n_total:>6d} rows")

    # Globally sample n indices
    rng = random.Random(seed)
    if n >= n_total:
        print(f"\n  Requested {n} but dataset only has {n_total} — using all.")
        global_indices = list(range(n_total))
    else:
        global_indices = sorted(rng.sample(range(n_total), n))

    # Map each global index to (shard_idx, row_within_shard)
    shard_starts = []
    cum = 0
    for c in row_counts:
        shard_starts.append(cum)
        cum += c

    by_shard = {}  # shard_idx -> sorted list of row indices within that shard
    for gi in global_indices:
        # Walk shards from last to first, find the one whose start ≤ gi
        for si in range(len(shard_starts) - 1, -1, -1):
            if gi >= shard_starts[si]:
                by_shard.setdefault(si, []).append(gi - shard_starts[si])
                break

    # Extract selected rows from each shard
    print(f"\nExtracting {len(global_indices)} rows from {len(by_shard)} shard(s)...")
    rows = []
    for si in sorted(by_shard.keys()):
        in_shard_rows = sorted(by_shard[si])
        print(f"  shard {si:2d}: extracting {len(in_shard_rows)} rows")
        # Read the full shard once; cheaper than seeking row-by-row for our sizes
        table = parquet_files[si].read()
        for ri in in_shard_rows:
            row = {col: table.column(col)[ri].as_py() for col in table.column_names}
            rows.append(row)

    return _save_rows(rows, cfg, dataset_name, output_dir, fmt,
                      manifest_path, n, seed,
                      source_label=f"{cfg['hf_name']}@parquet/{config}/{split}",
                      split=split)


# ── HuggingFace download (for datasets without scripts) ────────────────────────

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

    return _save_rows(rows, cfg, dataset_name, output_dir, fmt,
                      manifest_path, n, seed, source_label=cfg["hf_name"], split=split)


def _save_rows(rows, cfg, dataset_name, output_dir, fmt, manifest_path, n, seed, source_label, split):
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
    _save_manifest(manifest_path, source_label, split, n, saved, seed, manifest)
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
    elif cfg["source"] == "hf_parquet_api":
        split = args.split or cfg["hf_split"]
        saved = download_hf_parquet_api(cfg, split, args.n, output_dir, args.seed,
                                        args.format, args.stream, manifest_path, args.dataset)
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
