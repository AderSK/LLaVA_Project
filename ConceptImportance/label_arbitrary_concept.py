"""
label_arbitrary_concept.py

Run LLaVA over a directory of images and label each with YES/NO for ONE
arbitrary user-specified concept. Output JSON is structured so it can be
joined with the output of apply_probes.py downstream.

Usage:
    python label_arbitrary_concept.py \
        --images_dir ./flickr_sample \
        --concept beach \
        --description "the scene is at a beach or coastal area with sand and ocean visible" \
        --output beach_labels.json

    # Resume after interruption
    python label_arbitrary_concept.py \
        --images_dir ./flickr_sample \
        --concept beach \
        --description "the scene is at a beach or coastal area" \
        --output beach_labels.json \
        --resume
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import json
import argparse
import re
from pathlib import Path


def load_model():
    print("Loading LLaVA model with 4-bit quantization...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Model loaded. Memory: ~{model.get_memory_footprint() / 1e9:.2f} GB")
    return model, processor


def build_prompt(concept, description):
    """Single-concept variant of the prompt in generate_concept_labels.py."""
    return (
        "Look at this image carefully. Answer YES or NO based strictly on what is visible.\n\n"
        f'Question: Does this image contain "{concept}"?\n'
        f"Definition: {description}\n\n"
        "Reply ONLY with a JSON object on one line, e.g.:\n"
        f'{{"{concept}": "YES"}}    or    {{"{concept}": "NO"}}\n\n'
        "Output only the JSON, no explanation."
    )


def parse_label(response_text, concept):
    """Parse YES/NO label for the concept. Returns 0 or 1."""
    # Try JSON first
    json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            val = str(raw.get(concept, "NO")).strip().upper()
            return 1 if val.startswith("Y") else 0
        except json.JSONDecodeError:
            pass

    # Fallback: regex for "concept": "YES"/"NO"
    pattern = rf'"{re.escape(concept)}"\s*:\s*"(YES|NO)"'
    m = re.search(pattern, response_text, re.IGNORECASE)
    if m:
        return 1 if m.group(1).upper() == "YES" else 0

    # Last resort: free-text scan
    upper = response_text.strip().upper()
    if upper.startswith("YES") or " YES" in upper:
        return 1
    if upper.startswith("NO") or " NO" in upper:
        return 0
    return 0  # default to absent if unparseable


def label_image(image_path, model, processor, concept, description):
    image = Image.open(image_path).convert("RGB")
    prompt_text = build_prompt(concept, description)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Short generation — we only need a YES/NO JSON object
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    label = parse_label(response, concept)
    return label, response


def get_image_paths(images_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    images_dir = Path(images_dir)
    paths = []
    for ext in extensions:
        paths.extend(images_dir.rglob(f"*{ext}"))
        paths.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Label images with one arbitrary concept using LLaVA"
    )
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing input images")
    parser.add_argument("--concept", required=True,
                        help="Concept name, e.g. 'beach'. Used as the JSON key.")
    parser.add_argument("--description", required=True,
                        help='Definition shown to LLaVA, e.g. '
                             '"the scene is at a beach with sand and ocean visible"')
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <concept>_labels.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already present in the output JSON")
    args = parser.parse_args()

    output_path = Path(args.output or f"{args.concept}_labels.json")

    image_paths = get_image_paths(args.images_dir)
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Concept    : {args.concept}")
    print(f"Description: {args.description}")
    print(f"Output     : {output_path}")

    results = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} entries already in output JSON")

    model, processor = load_model()

    for idx, img_path in enumerate(image_paths):
        img_key = str(img_path)
        if args.resume and img_key in results and results[img_key].get("labels") is not None:
            continue

        print(f"\n[{idx+1}/{len(image_paths)}] {img_path.name}")
        try:
            label, raw_response = label_image(
                img_path, model, processor, args.concept, args.description
            )
            print(f"  {args.concept} = {'YES' if label else 'NO'}  "
                  f"(raw: {raw_response[:80].strip()})")
            results[img_key] = {
                "image_name": img_path.name,
                "image_path": img_key,
                "labels": {args.concept: label},
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[img_key] = {
                "image_name": img_path.name,
                "image_path": img_key,
                "labels": None,
                "error": str(e),
            }

        # Incremental save so progress is never lost
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results.values() if r.get("labels") is not None]
    n_pos = sum(1 for r in valid if r["labels"][args.concept] == 1)
    n_neg = len(valid) - n_pos

    print(f"\n{'='*60}")
    print(f"Done! {len(valid)}/{len(image_paths)} images labeled.")
    print(f"  {args.concept} = YES: {n_pos}  ({100*n_pos/max(1,len(valid)):.1f}%)")
    print(f"  {args.concept} = NO : {n_neg}  ({100*n_neg/max(1,len(valid)):.1f}%)")
    print(f"\nLabels saved to: {output_path}")

    if n_pos < 30 or n_neg < 30:
        print(f"\n[heads-up] One group has < 30 examples. The importance metric "
              f"will have wide confidence intervals. Consider a larger sample "
              f"or a more common concept.")


if __name__ == "__main__":
    main()
