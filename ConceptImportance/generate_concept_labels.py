"""
generate_concept_labels.py

Runs LLaVA over a dataset directory and generates multi-label concept annotations
for each image. Labels are saved to a JSON file for use in classifier training.

Usage:
    python generate_concept_labels.py --images_dir ./dataset --output labels.json
    python generate_concept_labels.py --images_dir ./dataset --output labels.json --layer 20
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import json
import argparse
import pickle
import re
from pathlib import Path

# ==============================================================================
# CONCEPT DEFINITIONS
# Edit this list to change which concepts are detected.
# Aim for concepts that are:
#   - Visually distinct and unambiguous
#   - Not overly correlated with each other
#   - Useful for downstream tasks / representation analysis
# ==============================================================================
CONCEPTS = [
    "people",        # Contains one or more humans / people
    "animals",       # Contains animals (pets, wildlife, etc.)
    "outdoor",       # Scene takes place outdoors
    "indoor",        # Scene takes place indoors
    "urban",         # City, buildings, streets, architecture
    "nature",        # Natural landscape (forest, mountains, water, sky)
    "food",          # Food or drink is present
    "vehicles",      # Cars, bikes, trains, aircraft, etc.
    "text_or_signs", # Visible text, signage, or writing
    "nighttime",     # Scene is dark / nighttime / low-light
]

CONCEPT_DESCRIPTIONS = {
    "people":        "one or more humans or people are visible",
    "animals":       "animals (pets, wildlife, birds, etc.) are present",
    "outdoor":       "the scene is outdoors or in an open environment",
    "indoor":        "the scene is indoors (room, building interior, etc.)",
    "urban":         "city elements like buildings, streets, or architecture are visible",
    "nature":        "natural landscape elements like trees, sky, water, or mountains are present",
    "food":          "food or drink items are visible",
    "vehicles":      "vehicles such as cars, bicycles, trains, or aircraft are present",
    "text_or_signs": "readable text, signs, labels, or written words are visible",
    "nighttime":     "the scene is dark, at night, or in very low light",
}


def load_model():
    """Load LLaVA model with 4-bit quantization."""
    print("Loading LLaVA model with 4-bit quantization...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"Model loaded. Memory: ~{model.get_memory_footprint() / 1e9:.2f} GB")
    return model, processor


def register_activation_hook(model, layer_idx):
    """Register a hook on a single layer. Returns (store_dict, handle)."""
    store = {"hidden": None}

    def hook(module, input, output):
        # output[0] is the hidden state: [batch, seq_len, hidden_dim]
        store["hidden"] = output[0].detach().cpu().to(torch.float16)

    handle = model.language_model.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def build_concept_prompt(concepts, concept_descriptions):
    """Build a structured prompt asking LLaVA to label concepts."""
    concept_lines = "\n".join(
        f"  - {c}: {concept_descriptions[c]}" for c in concepts
    )
    prompt = (
        "Look at this image carefully. For each concept below, answer YES or NO "
        "based strictly on what is visible in the image.\n\n"
        f"Concepts:\n{concept_lines}\n\n"
        "Reply ONLY with a JSON object like:\n"
        '{"people": "YES", "animals": "NO", ...}\n\n'
        "Do not add any explanation. Output only the JSON."
    )
    return prompt


def parse_concept_labels(response_text, concepts):
    """
    Parse the JSON response from LLaVA into a binary label dict.
    Falls back to keyword scanning if JSON parsing fails.
    """
    # Try to extract a JSON block from the response
    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            labels = {}
            for c in concepts:
                val = str(raw.get(c, "NO")).strip().upper()
                labels[c] = 1 if val.startswith("Y") else 0
            return labels
        except json.JSONDecodeError:
            pass

    # Fallback: scan for "concept_name": "YES/NO" patterns
    labels = {}
    for c in concepts:
        pattern = rf'"{c}"\s*:\s*"(YES|NO)"'
        m = re.search(pattern, response_text, re.IGNORECASE)
        if m:
            labels[c] = 1 if m.group(1).upper() == "YES" else 0
        else:
            labels[c] = 0  # default to absent if unparseable
            print(f"  [warn] Could not parse concept '{c}', defaulting to 0")

    return labels


def label_image(image_path, model, processor, concepts, concept_descriptions,
                activation_layer=None):
    """
    Run LLaVA on one image, return (label_dict, activation_tensor_or_None).
    activation_tensor shape: [seq_len, hidden_dim]  (float16)
    """
    image = Image.open(image_path).convert("RGB")
    prompt_text = build_concept_prompt(concepts, concept_descriptions)

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

    activation_store = None
    hook_handle = None
    if activation_layer is not None:
        activation_store, hook_handle = register_activation_hook(model, activation_layer)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    if hook_handle is not None:
        hook_handle.remove()

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    labels = parse_concept_labels(response, concepts)
    activation = activation_store["hidden"][0] if activation_store is not None else None  # [seq, hidden]

    return labels, activation, response


def get_image_paths(images_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    images_dir = Path(images_dir)
    paths = []
    for ext in extensions:
        paths.extend(images_dir.rglob(f"*{ext}"))
        paths.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="Generate concept labels for images using LLaVA")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default="concept_labels.json", help="Output JSON file for labels")
    parser.add_argument("--layer", type=int, default=None,
                        help="If set, also save activations from this layer index to .pkl files in activations/")
    parser.add_argument("--activations_dir", default="activations",
                        help="Directory to save per-image activation .pkl files")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already present in the output JSON")
    args = parser.parse_args()

    image_paths = get_image_paths(args.images_dir)
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"\nConcepts to label ({len(CONCEPTS)}):")
    for c, desc in CONCEPT_DESCRIPTIONS.items():
        print(f"  [{c}]  {desc}")

    # Load existing results if resuming
    output_path = Path(args.output)
    results = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"\nResuming — {len(results)} images already labeled")

    model, processor = load_model()

    if args.layer is not None:
        total_layers = len(model.language_model.layers)
        assert 0 <= args.layer < total_layers, \
            f"Layer {args.layer} out of range (model has {total_layers} layers)"
        Path(args.activations_dir).mkdir(exist_ok=True)
        print(f"\nWill also save layer-{args.layer} activations to '{args.activations_dir}/'")

    for idx, img_path in enumerate(image_paths):
        img_key = str(img_path)

        if args.resume and img_key in results:
            print(f"[{idx+1}/{len(image_paths)}] Skipping (already done): {img_path.name}")
            continue

        print(f"\n[{idx+1}/{len(image_paths)}] {img_path.name}")

        try:
            labels, activation, raw_response = label_image(
                img_path, model, processor,
                CONCEPTS, CONCEPT_DESCRIPTIONS,
                activation_layer=args.layer
            )

            # Show which concepts were detected
            detected = [c for c, v in labels.items() if v == 1]
            print(f"  Detected: {detected if detected else '(none)'}")
            print(f"  Raw response snippet: {raw_response[:120].strip()}")

            results[img_key] = {
                "image_name": img_path.name,
                "image_path": img_key,
                "labels": labels,
            }

            # Save activation alongside labels reference
            if activation is not None:
                act_file = Path(args.activations_dir) / f"{img_path.stem}_layer{args.layer}.pkl"
                with open(act_file, "wb") as f:
                    pickle.dump({
                        "image_path": img_key,
                        "layer": args.layer,
                        "activation": activation,   # float16 [seq_len, hidden_dim]
                        "labels": labels,
                    }, f)
                results[img_key]["activation_file"] = str(act_file)

        except Exception as e:
            print(f"  ERROR: {e}")
            results[img_key] = {
                "image_name": img_path.name,
                "image_path": img_key,
                "labels": None,
                "error": str(e),
            }

        # Save incrementally so progress isn't lost
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results.values() if r.get("labels") is not None]
    print(f"\n{'='*60}")
    print(f"Done! {len(valid)}/{len(image_paths)} images labeled.")
    print(f"Labels saved to: {output_path}")

    if valid:
        print("\nConcept frequency:")
        for c in CONCEPTS:
            count = sum(1 for r in valid if r["labels"].get(c, 0) == 1)
            bar = "█" * int(count / len(valid) * 30)
            print(f"  {c:18s} {count:4d}/{len(valid)}  {bar}")


if __name__ == "__main__":
    main()
