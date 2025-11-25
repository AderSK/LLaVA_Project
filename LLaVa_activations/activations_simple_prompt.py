import torch
import os
import pickle
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

IMAGE_DIR = Path("../large-data/val2017") 
OUTPUT_DIR = Path("activations")
MAX_SAMPLES = 200 
LAYERS_TO_HOOK = [10, 15, 20, 25] 
PROMPT_TEXT = "Describe this image in detail."

def load_model_4bit():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
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

def main():
    if not IMAGE_DIR.exists():
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    image_files = sorted(list(IMAGE_DIR.glob("*.jpg")))
    image_files = image_files[:MAX_SAMPLES]

    model, processor = load_model_4bit()
    activations_store, handles = register_hooks(model, LAYERS_TO_HOOK)
    
    for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
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
                "image_name": img_path.name,
                "activations": {l: activations_store[l][0] for l in LAYERS_TO_HOOK} 
            }
            
            save_path = OUTPUT_DIR / f"{img_path.stem}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(save_data, f)

        except Exception as e:
            print(f"Error {img_path.name}: {e}")
            continue

    for handle in handles:
        handle.remove()
    print(f"\nActivations saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
