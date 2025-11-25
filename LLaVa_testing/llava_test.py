import torch
import os
from pathlib import Path
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

IMAGE_DIR = Path("../large-data/val2017")
PROMPT = "Describe this image in detail."
MAX_IMAGES = 5

def load_model():
    print("Loading LLaVA (Float16)...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, processor

def main():
    images = sorted(list(IMAGE_DIR.glob("*.jpg")))[:MAX_IMAGES]

    model, processor = load_model()

    for img_path in images:
        try:
            print(f"Image: {img_path.name}")
            image = Image.open(img_path).convert('RGB')
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=100)
            response = processor.decode(output[0], skip_special_tokens=True)
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            print(f"Description: {response}\n" + "-"*50)

        except Exception as e:
            print(f"Error {img_path.name}: {e}")

if __name__ == "__main__":
    main()
