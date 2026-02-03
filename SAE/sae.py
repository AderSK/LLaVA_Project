import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

LAYER = 11
BASE_PATH = "/home/cervenka25/large-data"
IMAGE_DIR = os.path.join(BASE_PATH, "test2014")
SAE_FILE = os.path.join(BASE_PATH, "CLIP-ViT-L-14-SAE-L11/11_resid/1000104192.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_FEATURES = 5

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", 
    quantization_config=bnb_config,
    device_map="auto"
)

checkpoint = torch.load(SAE_FILE, map_location=device)
state_dict = checkpoint['model_state_dict']

pre_b = state_dict['pre_b'].to(device).half()
W_enc = state_dict['enc'].to(device).half()
W_dec = state_dict['dec'].to(device).half()

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

random_image_path = os.path.join(IMAGE_DIR, random.choice(images))

raw_image = Image.open(random_image_path).convert("RGB")

prompt = "Describe the image"
inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(device)

with torch.no_grad():
    vision_tower = model.model.vision_tower
    v_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
    h = v_outputs.hidden_states[LAYER]
    latents = torch.relu((h - pre_b) @ W_enc)

spatial_acts = latents[0, 1:, :]
feature_sums = spatial_acts.sum(dim=0)
top_k_values, top_k_indices = torch.topk(feature_sums, TOP_K_FEATURES)

active_features = (feature_sums > 0).sum().item()
for i, (idx, val) in enumerate(zip(top_k_indices, top_k_values)):
    print(f"  {i+1}. Feature ID {idx.item()}: aktivácia {val.item():.2f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(raw_image)
axes[0].set_title(f"Pôvodný obrázok\n{os.path.basename(random_image_path)}")
axes[0].axis('off')

for i in range(TOP_K_FEATURES):
    feature_id = top_k_indices[i].item()
    heatmap = spatial_acts[:, feature_id].cpu().numpy().reshape(24, 24)
    
    im = axes[i+1].imshow(heatmap, cmap='viridis', interpolation='bilinear')
    axes[i+1].set_title(f"Feature {feature_id}\n(aktivácia: {top_k_values[i].item():.1f})")
    axes[i+1].axis('off')
    plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)

plt.tight_layout()

output_plot = "vysledok_sae_top5.png"
plt.savefig(output_plot, dpi=150)
