import torch
import os
import random
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

DEVICE = "cuda"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = LlavaProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0})

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
random_img_path = os.path.join(IMAGE_DIR, random.choice(all_images))
image = Image.open(random_img_path).convert("RGB")

print(f"Testing with image: {random_img_path}")

prompt = "USER: <image>\nWhat is in this picture? Be concise. ASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1])