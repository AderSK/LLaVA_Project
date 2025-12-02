import torch
import os
import pickle
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


DATASET_TYPE = "val2014" 
BASE_DATA_DIR = Path("/tmp/llava_data")
IMAGE_DIR = BASE_DATA_DIR / DATASET_TYPE
ANNOTATION_FILE = BASE_DATA_DIR / "annotations" / f"instances_{DATASET_TYPE}.json"
OUTPUT_DIR = Path("activations_2014")
MAX_SAMPLES = None
LAYERS_TO_HOOK = [10, 15, 20, 25] 
PROMPT_TEXT = "Describe this image in detail."


def load_model_float16():
    print("Loading LLaVA model...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, processor


def register_hooks(model, layers):
    activations = {layer: [] for layer in layers}

    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            hidden_state = output[0].detach().cpu().to(torch.float16)
            activations[layer_idx].append(hidden_state)
        return hook

    handles = []
    for layer_idx in layers:
        layer_module = model.language_model.layers[layer_idx]
        handle = layer_module.register_forward_hook(get_activation_hook(layer_idx))
        handles.append(handle)
        
    return activations, handles


def get_image_list_from_json(json_path):
    print(f"Loading annotations from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_files = [img['file_name'] for img in data['images']]
    print(f"Found {len(image_files)} images in dataset manifest.")
    return image_files


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_filenames = get_image_list_from_json(ANNOTATION_FILE)
    
    if MAX_SAMPLES is not None:
        image_filenames = image_filenames[:MAX_SAMPLES]
        print(f"Limiting to {MAX_SAMPLES} images")

    model, processor = load_model_float16()
    activations_store, handles = register_hooks(model, LAYERS_TO_HOOK)

    print(f"\nStarting for {len(image_filenames)} images")
    
    for filename in tqdm(image_filenames):
        img_path = IMAGE_DIR / filename
        
        save_path = OUTPUT_DIR / f"{Path(filename).stem}.pkl"
        if save_path.exists():
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]}]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            for layer in LAYERS_TO_HOOK:
                activations_store[layer].clear()

            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)

            save_data = {
                "image_name": filename,
                "activations": {l: activations_store[l][0] for l in LAYERS_TO_HOOK} 
            }
            
            with open(save_path, "wb") as f:
                pickle.dump(save_data, f)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    for handle in handles:
        handle.remove()
    print(f"\nActivations saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
