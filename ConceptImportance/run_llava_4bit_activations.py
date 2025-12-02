from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import sys
import os
import pickle
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def load_model():
    """Load LLaVA model with 4-bit quantization for 8GB VRAM"""
    print("Loading LLaVA model with 4-bit quantization...")

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    # Configure 4-bit quantization (uses ~4-5GB VRAM instead of ~7GB)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_id)

    # Load model with quantization
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
    """
    Register hooks to collect activations from specified layers.
    
    Args:
        model: The LLaVA model
        layers: List of layer indices to hook, or None for all layers
    
    Returns:
        activations_store: Dictionary to store activations
        handles: List of hook handles (to remove later)
    """
    # Get total number of layers
    total_layers = len(model.language_model.layers)
    
    # If no layers specified, hook all layers
    if layers is None:
        layers = list(range(total_layers))
    
    print(f"Registering hooks for {len(layers)} layers (out of {total_layers} total)")
    
    activations_store = {layer: [] for layer in layers}
    
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            # Extract hidden state and convert to float16 to save memory
            hidden_state = output[0].detach().cpu().to(torch.float16)
            activations_store[layer_idx].append(hidden_state)
        return hook
    
    handles = []
    for layer_idx in layers:
        if layer_idx >= total_layers:
            print(f"Warning: Layer {layer_idx} doesn't exist (max: {total_layers-1})")
            continue
        layer_module = model.language_model.layers[layer_idx]
        handle = layer_module.register_forward_hook(get_activation_hook(layer_idx))
        handles.append(handle)
    
    return activations_store, handles

def clear_activations(activations_store):
    """Clear all stored activations"""
    for layer in activations_store:
        activations_store[layer].clear()

def remove_hooks(handles):
    """Remove all registered hooks"""
    for handle in handles:
        handle.remove()
    print("All hooks removed")

def save_activations(activations_store, image_path, output_dir="activations"):
    """
    Save collected activations to a pickle file.
    
    Args:
        activations_store: Dictionary of layer activations
        image_path: Path to the original image
        output_dir: Directory to save activations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create save data - take the first (and usually only) activation from each layer
    save_data = {
        "image_name": Path(image_path).name,
        "activations": {layer: activations_store[layer][0] if activations_store[layer] else None 
                       for layer in activations_store}
    }
    
    # Save with image name
    save_file = output_path / f"{Path(image_path).stem}_activations.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"Activations saved to: {save_file}")
    return save_file

def resolve_image_path(image_path):
    """Resolve image path - handles both absolute and relative paths"""
    # Remove quotes if present
    image_path = image_path.strip('"').strip("'")

    # If it's already an absolute path and exists, use it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    # Try relative to script directory
    script_relative = SCRIPT_DIR / image_path
    if script_relative.exists():
        return str(script_relative)

    # Try relative to current working directory
    if os.path.exists(image_path):
        return image_path

    # Path not found
    return None

def analyze_image(image_path, question, model, processor, max_tokens=512, 
                  collect_activations=False, activations_store=None, save_activations_to_file=False):
    """
    Analyze an image with a question
    
    Args:
        image_path: Path to the image
        question: Question to ask about the image
        model: The LLaVA model
        processor: The LLaVA processor
        max_tokens: Maximum tokens to generate
        collect_activations: Whether to collect activations during generation
        activations_store: Dictionary to store activations (required if collect_activations=True)
        save_activations_to_file: Whether to save activations to a pickle file
    
    Returns:
        response: Model's text response
    """
    try:
        # Resolve the path
        resolved_path = resolve_image_path(image_path)

        if resolved_path is None:
            return f"Error: Could not find image at '{image_path}'\nTried:\n  - {image_path}\n  - {SCRIPT_DIR / image_path}\n  - Current directory: {os.getcwd()}"

        print(f"Loading image from: {resolved_path}")

        # Load image
        image = Image.open(resolved_path).convert('RGB')

        # Prepare prompt (proper format for LLaVA 1.6)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Clear activations before generation if collecting
        if collect_activations and activations_store is not None:
            clear_activations(activations_store)

        # Generate response
        print("Generating response...")
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        # Save activations if requested
        if collect_activations and save_activations_to_file and activations_store is not None:
            save_activations(activations_store, resolved_path)

        # Decode and return
        response = processor.decode(output[0], skip_special_tokens=True)

        # Extract just the answer (remove the prompt part)
        # The response format varies, so we try multiple extraction methods
        if "assistant\n" in response.lower():
            parts = response.split("assistant\n")
            if len(parts) > 1:
                response = parts[-1].strip()
        elif "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Current working directory: {os.getcwd()}\n")

    # Load model once
    model, processor = load_model()

    print("\n" + "="*60)
    print("LLaVA is ready! You can now analyze images.")
    print("="*60)
    print("\nPath tips:")
    print("  - Relative paths are relative to the script location")
    print("  - Example: test_images\\image.png")
    print("  - Or use full path: C:\\Users\\...\\image.jpg")
    print("="*60)
    
    # Ask if user wants to collect activations
    print("\nActivation Collection Mode:")
    print("  1. Normal mode (no activation collection)")
    print("  2. Collect activations (specify layers)")
    print("  3. Collect activations (all layers)")
    
    mode = input("\nSelect mode (1-3): ").strip()
    
    activations_store = None
    handles = []
    collect_activations = False
    
    if mode in ['2', '3']:
        collect_activations = True
        
        if mode == '2':
            # Get specific layers
            total_layers = len(model.language_model.layers)
            print(f"\nModel has {total_layers} layers (0-{total_layers-1})")
            layer_input = input("Enter layer indices (comma-separated, e.g., 10,15,20,25): ").strip()
            try:
                layers = [int(x.strip()) for x in layer_input.split(',')]
            except:
                print("Invalid input, using default layers [10, 15, 20, 25]")
                layers = [10, 15, 20, 25]
        else:
            # All layers
            layers = None
        
        activations_store, handles = register_activation_hooks(model, layers)
        
        # Ask if user wants to save activations
        save_choice = input("\nSave activations to files? (y/n): ").strip().lower()
        save_activations_flag = save_choice == 'y'
    else:
        save_activations_flag = False

    # Interactive mode
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
            image_path, 
            question, 
            model, 
            processor,
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
    
    # Clean up hooks
    if handles:
        remove_hooks(handles)

if __name__ == "__main__":
    main()
