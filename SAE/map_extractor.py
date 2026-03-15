import os, random, torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"
TARGET_FID = 2152
MAX_SEARCH = 3000

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

patch_crops = []

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.shuffle(all_images)

for img_name in tqdm(all_images[:MAX_SEARCH]):
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    img_resized = img.resize((336, 336)) 
    
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
        f = sae.encode(h)[0, 1:, TARGET_FID]
        
    max_act, patch_idx = f.max(dim=0)
    
    if max_act.item() > 0:
        row = (patch_idx.item() // 24) * 14
        col = (patch_idx.item() % 24) * 14
        crop = img_resized.crop((max(0, col-7), max(0, row-7), min(336, col+21), min(336, row+21)))
        patch_crops.append((max_act.item(), crop))

patch_crops.sort(key=lambda x: x[0], reverse=True)
top_crops = patch_crops[:25]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, (act, crop) in enumerate(top_crops):
    ax = axes[i // 5, i % 5]
    ax.imshow(crop)
    ax.set_title(f"Act: {act:.1f}", fontsize=8)
    ax.axis('off')

plt.suptitle(f"Max Activating Patches for Feature {TARGET_FID}", fontsize=14)
plt.tight_layout()
plt.savefig("map_extractor.png")