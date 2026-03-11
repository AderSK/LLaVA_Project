"""
analyze_labels.py

Reads the JSON output of generate_concept_labels.py and prints a concept
frequency table, plus optional per-image breakdowns and co-occurrence stats.

Usage:
    python analyze_labels.py concept_labels.json
    python analyze_labels.py concept_labels.json --cooccurrence
    python analyze_labels.py concept_labels.json --list_images people
"""

import json
import argparse
from pathlib import Path


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_valid(results):
    """Return only entries that have a non-null labels dict."""
    return [r for r in results.values() if r.get("labels") is not None]


def print_frequency_table(valid, concepts=None):
    """Print the same bar-chart table that generate_concept_labels prints at the end."""
    if not valid:
        print("No valid labeled images found.")
        return

    # Infer concept list from data if not supplied
    if concepts is None:
        concepts = sorted({k for r in valid for k in r["labels"].keys()})

    n = len(valid)
    print(f"\nConcept frequency  ({n} images)\n")
    print(f"  {'Concept':<20} {'Count':>6}   Bar")
    print(f"  {'-'*20} {'-'*6}   {'-'*32}")

    for c in concepts:
        count = sum(1 for r in valid if r["labels"].get(c, 0) == 1)
        pct = count / n
        bar = "█" * int(pct * 30)
        print(f"  {c:<20} {count:4d}/{n}  {bar}  ({pct*100:.1f}%)")


def print_cooccurrence(valid, concepts):
    """Print a simple co-occurrence matrix (count of images where both A and B are 1)."""
    print(f"\nCo-occurrence matrix  (images where BOTH concepts are present)\n")
    col_w = 6
    header = f"  {'':18s}" + "".join(f"{c[:col_w]:>{col_w+1}}" for c in concepts)
    print(header)
    print(f"  {'-'*18}" + "-" * ((col_w + 1) * len(concepts)))

    for ca in concepts:
        row = f"  {ca:<18}"
        for cb in concepts:
            if cb == ca:
                count = sum(1 for r in valid if r["labels"].get(ca, 0) == 1)
                row += f"  {'--':>{col_w-1}}"
            else:
                count = sum(
                    1 for r in valid
                    if r["labels"].get(ca, 0) == 1 and r["labels"].get(cb, 0) == 1
                )
                row += f"  {count:>{col_w-1}}"
        print(row)


def list_images_for_concept(valid, concept):
    """Print all images where a given concept is labeled 1."""
    matches = [r for r in valid if r["labels"].get(concept, 0) == 1]
    print(f"\nImages labeled '{concept}': {len(matches)}\n")
    for r in matches:
        print(f"  {r['image_name']}")


def print_summary(results, valid):
    errors = [r for r in results.values() if r.get("labels") is None]
    print(f"\n{'='*60}")
    print(f"  Total entries : {len(results)}")
    print(f"  Labeled OK    : {len(valid)}")
    print(f"  Errors/skipped: {len(errors)}")
    if errors:
        print(f"\n  Failed images:")
        for r in errors:
            msg = r.get("error", "unknown error")
            print(f"    {r['image_name']}: {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse concept_labels.json from generate_concept_labels.py"
    )
    parser.add_argument("labels_json", help="Path to the concept_labels.json file")
    parser.add_argument("--cooccurrence", action="store_true",
                        help="Also print a concept co-occurrence matrix")
    parser.add_argument("--list_images", metavar="CONCEPT", default=None,
                        help="List all images labeled with this concept")
    args = parser.parse_args()

    json_path = Path(args.labels_json)
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    results = load_results(json_path)
    valid = get_valid(results)

    # Preserve concept order from first valid entry if possible
    if valid:
        concepts = list(valid[0]["labels"].keys())
    else:
        concepts = None

    print_summary(results, valid)
    print_frequency_table(valid, concepts)

    if args.cooccurrence and concepts:
        print_cooccurrence(valid, concepts)

    if args.list_images:
        list_images_for_concept(valid, args.list_images)


if __name__ == "__main__":
    main()
