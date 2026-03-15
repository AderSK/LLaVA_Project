import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae_topk/trainer_0/ae.pt"
IMG_PATH = "/home/cervenka25/large-data/test2014/COCO_test2014_000000282536.jpg"
TARGET_FID = 2152 
DEVICE = "cuda:0"

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", 
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map={"": 0}
)
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

image = Image.open(IMG_PATH).convert("RGB")
inputs = processor(images=image, text="USER: <image>\nWhat is this? ASSISTANT:", return_tensors="pt").to(DEVICE)

with torch.no_grad():
    h = model.model.vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
    f = sae.encode(h)
    acts = f[0, 1:, TARGET_FID].cpu().float().numpy()
    heatmap = acts.reshape(24, 24)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(image)
im = axes[1].imshow(heatmap, cmap='jet', alpha=0.6, 
                    extent=(0, image.size[0], image.size[1], 0), 
                    interpolation='nearest')

axes[1].set_title(f"Raw 24x24 Patch Activations (Feature {TARGET_FID})")
axes[1].axis('off')

patch_w, patch_h = image.size[0] / 24, image.size[1] / 24
for i in range(25):
    axes[1].axhline(i * patch_h, color='white', alpha=0.2, linewidth=0.5)
    axes[1].axvline(i * patch_w, color='white', alpha=0.2, linewidth=0.5)

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.savefig("raw_patch_heatmap.png", bbox_inches='tight')