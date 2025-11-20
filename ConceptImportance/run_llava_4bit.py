from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import sys
import os
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

def analyze_image(image_path, question, model, processor, max_tokens=512):
    """Analyze an image with a question"""
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

        # Generate response
        print("Generating response...")
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

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

        response = analyze_image(image_path, question, model, processor)

        print("\n" + "="*60)
        print("Response:")
        print("="*60)
        print(response)
        print("="*60)

if __name__ == "__main__":
    main()
