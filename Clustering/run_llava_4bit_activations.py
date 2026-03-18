from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import sys
import os
import pickle
import numpy as np
import gc
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def load_model():
    """Load LLaVA model with 4-bit quantization for 8GB VRAM"""
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

    print("Model loaded successfully!")
    print(f"Model memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")
    return model, processor

def register_activation_hooks(model, layers=None):
    total_layers = len(model.language_model.layers)
    if layers is None:
        layers = list(range(total_layers))
    
    print(f"Registering hooks for {len(layers)} layers (out of {total_layers} total)")
    activations_store = {layer: [] for layer in layers}
    
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            hidden_state = output[0].detach().cpu().numpy().astype(np.float16)
            activations_store[layer_idx].append(hidden_state)
        return hook
    
    handles = []
    for layer_idx in layers:
        if layer_idx >= total_layers:
            continue
        layer_module = model.language_model.layers[layer_idx]
        handle = layer_module.register_forward_hook(get_activation_hook(layer_idx))
        handles.append(handle)
    
    return activations_store, handles

def clear_activations(activations_store):
    for layer in activations_store:
        activations_store[layer].clear()

def remove_hooks(handles):
    for handle in handles:
        handle.remove()
    print("All hooks removed")

# --- UPDATED: Hardcoded target path and added parents=True ---
def save_activations(activations_store, image_path, output_dir=r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Activations"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "image_name": Path(image_path).name,
        "activations": {layer: activations_store[layer][0] if activations_store[layer] else None 
                       for layer in activations_store}
    }
    
    save_file = output_path / f"{Path(image_path).stem}_activations.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"Activations saved to: {save_file}")
    return save_file

def resolve_image_path(image_path):
    image_path = image_path.strip('"').strip("'")
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    script_relative = SCRIPT_DIR / image_path
    if script_relative.exists():
        return str(script_relative)
    if os.path.exists(image_path):
        return image_path
    return None

def analyze_image(image_path, question, model, processor, max_tokens=512, 
                  collect_activations=False, activations_store=None, save_activations_to_file=False):
    try:
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            return f"Error: Could not find image at '{image_path}'"

        print(f"Loading image from: {resolved_path}")
        image = Image.open(resolved_path).convert('RGB')

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        if collect_activations and activations_store is not None:
            clear_activations(activations_store)

        print("Generating response...")
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        if collect_activations and save_activations_to_file and activations_store is not None:
            save_activations(activations_store, resolved_path)

        response = processor.decode(output[0], skip_special_tokens=True)

        if "assistant\n" in response.lower():
            parts = response.split("assistant\n")
            if len(parts) > 1:
                response = parts[-1].strip()
        elif "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    except Exception as e:
        return f"Error: {str(e)}"

def process_folder(folder_path, question, model, processor, collect_activations, activations_store, save_activations_flag):
    folder_path = Path(folder_path.strip('"').strip("'"))
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Directory not found at: {folder_path}")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in valid_extensions]
    image_files.sort() 
    
    if not image_files:
        print(f"No valid images found in '{folder_path}'")
        return

    print(f"\nFound {len(image_files)} images. Starting sequential processing...")
    
    output_log_path = folder_path / "folder_results.txt"
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Prompt used: {question}\n")
        log_file.write("="*60 + "\n\n")

    for i, img_path in enumerate(image_files, 1):
        print("\n" + "="*60)
        print(f"Processing Image {i}/{len(image_files)}: {img_path.name}")
        print("="*60)
        
        response = analyze_image(
            str(img_path), 
            question, 
            model, 
            processor,
            collect_activations=collect_activations,
            activations_store=activations_store,
            save_activations_to_file=save_activations_flag
        )
        
        print(f"\nResponse:\n{response}")
        
        with open(output_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"--- {img_path.name} ---\n{response}\n\n")
        
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFolder processing complete! All answers saved to: {output_log_path}")

def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Current working directory: {os.getcwd()}\n")

    model, processor = load_model()

    print("\n" + "="*60)
    print("LLaVA is ready!")
    print("="*60)
    
    print("\nActivation Collection Mode:")
    print("  1. Normal mode (no activation collection)")
    print("  2. Collect activations (specify layers)")
    print("  3. Collect activations (all layers)")
    
    mode = input("\nSelect mode (1-3): ").strip()
    
    activations_store = None
    handles = []
    collect_activations = False
    save_activations_flag = False
    
    if mode in ['2', '3']:
        collect_activations = True
        if mode == '2':
            total_layers = len(model.language_model.layers)
            print(f"\nModel has {total_layers} layers (0-{total_layers-1})")
            layer_input = input("Enter layer indices (comma-separated, e.g., 10,15,20,25): ").strip()
            try:
                layers = [int(x.strip()) for x in layer_input.split(',')]
            except:
                print("Invalid input, using default layers [10, 15, 20, 25]")
                layers = [10, 15, 20, 25]
        else:
            layers = None
        
        activations_store, handles = register_activation_hooks(model, layers)
        save_choice = input("\nSave activations to files? (y/n): ").strip().lower()
        save_activations_flag = save_choice == 'y'

    print("\nProcessing Mode:")
    print("  1. Single Image Mode (Interactive)")
    print("  2. Folder Mode (Process all images sequentially)")
    proc_mode = input("\nSelect mode (1-2): ").strip()

    if proc_mode == '2':
        folder_path = input("\nEnter the folder path containing the images: ").strip()
        question = input("Enter the single prompt to apply to ALL images: ").strip()
        
        if not question:
            question = "Describe this image in detail."
            
        process_folder(
            folder_path, question, model, processor,
            collect_activations, activations_store, save_activations_flag
        )
    else:
        while True:
            print("\n" + "-"*60)
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()

            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            question = input("Enter your question about the image: ").strip()
            if not question:
                question = "Describe this image in detail."

            response = analyze_image(
                image_path, question, model, processor,
                collect_activations=collect_activations,
                activations_store=activations_store,
                save_activations_to_file=save_activations_flag
            )

            print("\n" + "="*60)
            print("Response:")
            print("="*60)
            print(response)
            print("="*60)
            
            if collect_activations:
                print(f"\nActivations collected for {len(activations_store)} layers")
                for layer_idx in sorted(activations_store.keys()):
                    if activations_store[layer_idx]:
                        act_shape = activations_store[layer_idx][0].shape
                        print(f"  Layer {layer_idx}: {act_shape}")
    
    if handles:
        remove_hooks(handles)

if __name__ == "__main__":
    main()
