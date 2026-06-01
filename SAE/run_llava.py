import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

IMAGE_PATH = "/home/adam/Projects/street.jpg"
DEVICE     = "cuda:0"

print("Loading LLaVA-1.5-7B (from local cache)...")
model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)

llava_model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(DEVICE)

img = Image.open(IMAGE_PATH).convert("RGB")

prompt = "USER: <image>\nDescribe what you see in detail. ASSISTANT:"

inputs = processor(text=prompt, images=img, return_tensors="pt").to(DEVICE, torch.float16)

with torch.no_grad():
    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=True
    )

output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
final_description = output_text.split("ASSISTANT:")[-1].strip()

print("\n" + "="*60 + "\n")
print(prompt + "\n")
print("="*60)
print(f"\n{final_description}\n")
print("="*60)
