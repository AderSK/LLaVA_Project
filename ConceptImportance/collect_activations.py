"""
collect_activations.py

Runs LLaVA over a folder of images and saves per-image layer activations
to .pkl files. Optionally also saves a manifest JSON.

Usage:
    # Collect layer 20 activations with default prompt
    python collect_activations.py --images_dir ./sampled_images --layers 20

    # Collect multiple layers with a custom prompt
    python collect_activations.py --images_dir ./sampled_images --layers 10,20,30 \\
        --prompt "What objects are visible in this image?"

    # Collect all layers (slow, large output)
    python collect_activations.py --images_dir ./sampled_images --all_layers

    # Resume an interrupted run
    python collect_activations.py --images_dir ./sampled_images --layers 20 --resume
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import json
import argparse
import pickle
from pathlib import Path

# ==============================================================================
# DEFAULT PROMPT
# Same as the interactive default in run_llava_4bit_activations.py
# ==============================================================================
DEFAULT_PROMPT = "Describe this image in detail."


def load_model():
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


def register_activation_hooks(model, layers):
    """
    Register forward hooks on specified layers.
    Returns (activations_store dict, list of handles).
    """
    total_layers = len(model.language_model.layers)
    invalid = [l for l in layers if not (0 <= l < total_layers)]
    if invalid:
        raise ValueError(f"Layer(s) {invalid} out of range — model has {total_layers} layers (0–{total_layers-1})")

    store = {l: None for l in layers}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[0]: [batch, seq_len, hidden_dim]
            store[layer_idx] = output[0].detach().cpu().to(torch.float16)
        return hook

    for l in layers:
        handle = model.language_model.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    return store, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def run_image(image_path, prompt, model, processor, activation_store):
    """
    Run LLaVA on one image. Clears activation_store in-place before the
    forward pass so each call contains only that image's activations.
    Returns the decoded text response.
    """
    # Clear previous activations
    for k in activation_store:
        activation_store[k] = None

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_str = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


def get_image_paths(images_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    images_dir = Path(images_dir)
    paths = []
    for ext in extensions:
        paths.extend(images_dir.rglob(f"*{ext}"))
        paths.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def already_done(img_stem, layers, output_dir):
    """Check if all requested layer pkl files exist for this image."""
    return all(
        (Path(output_dir) / f"{img_stem}_layer{l}.pkl").exists()
        for l in layers
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect LLaVA layer activations for all images in a folder"
    )
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", default="activations",
                        help="Directory to write .pkl activation files (default: activations/)")
    parser.add_argument("--manifest", default="activation_manifest.json",
                        help="Path to manifest JSON (default: activation_manifest.json)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help=f'Text prompt sent alongside each image (default: "{DEFAULT_PROMPT}")')
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to collect, e.g. 10,20,30")
    parser.add_argument("--all_layers", action="store_true",
                        help="Collect activations from ALL layers (overrides --layers; slow & large)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images whose .pkl files already exist in output_dir")
    args = parser.parse_args()

    # Resolve image list
    image_paths = get_image_paths(args.images_dir)
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return
    print(f"Found {len(image_paths)} images in '{args.images_dir}'")

    # Load model first so we know total layer count
    model, processor = load_model()
    total_layers = len(model.language_model.layers)
    print(f"Model has {total_layers} layers")

    # Resolve which layers to collect
    if args.all_layers:
        layers = list(range(total_layers))
        print(f"Collecting ALL {total_layers} layers")
    elif args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"Collecting layers: {layers}")
    else:
        parser.error("Specify --layers <indices> or --all_layers")

    print(f"\nPrompt: \"{args.prompt}\"")

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_store, handles = register_activation_hooks(model, layers)

    manifest = []
    saved = 0
    skipped = 0
    errors = 0

    for idx, img_path in enumerate(image_paths):
        prefix = f"[{idx+1}/{len(image_paths)}]"

        if args.resume and already_done(img_path.stem, layers, output_dir):
            print(f"{prefix} Skipping (already done): {img_path.name}")
            skipped += 1
            manifest.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "status": "skipped",
                "pkl_files": [
                    str(output_dir / f"{img_path.stem}_layer{l}.pkl") for l in layers
                ],
            })
            continue

        print(f"\n{prefix} {img_path.name}")

        try:
            response = run_image(img_path, args.prompt, model, processor, activation_store)
            print(f"  Response snippet: {response[:100].strip()}")

            # Save one pkl per layer
            pkl_files = []
            for l in layers:
                act = activation_store[l]
                if act is None:
                    print(f"  [warn] Layer {l} activation is None — skipping")
                    continue
                pkl_path = output_dir / f"{img_path.stem}_layer{l}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump({
                        "image_name": img_path.name,
                        "image_path": str(img_path),
                        "prompt": args.prompt,
                        "layer": l,
                        "activation": act,   # float16 [seq_len, hidden_dim]
                        "response": response,
                    }, f)
                pkl_files.append(str(pkl_path))
                # Print shape once per image (same for all layers)
                if l == layers[0]:
                    print(f"  Activation shape: {act.shape}  (seq_len × hidden_dim)")

            manifest.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "status": "ok",
                "response_snippet": response[:200],
                "pkl_files": pkl_files,
            })
            saved += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            manifest.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "status": "error",
                "error": str(e),
            })
            errors += 1

        # Incremental manifest save
        manifest_path = Path(args.manifest)
        with open(manifest_path, "w") as f:
            json.dump({
                "prompt": args.prompt,
                "layers": layers,
                "images_dir": str(args.images_dir),
                "entries": manifest,
            }, f, indent=2)

    remove_hooks(handles)

    print(f"\n{'='*60}")
    print(f"Done!  saved={saved}  skipped={skipped}  errors={errors}")
    print(f"Activations in : {output_dir}/")
    print(f"Manifest       : {args.manifest}")
    print(f"\nPkl file layout:")
    print(f"  {{'image_name', 'image_path', 'prompt', 'layer', 'activation', 'response'}}")
    print(f"  activation shape: [seq_len, hidden_dim]  dtype=float16")


if __name__ == "__main__":
    main()
