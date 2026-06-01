import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR  = "/home/adam/Projects/data/test2014"
SAE_PATH   = "/home/adam/Documents/sae_backup_lol/ae_layer18_topk64.pt"
OUTPUT_PLOT = "/home/adam/Projects/feature_41045_distribution.png"

DEVICE      = "cuda:0"
LAYER       = 18
FEATURE_IDX = 41045
BATCH_SIZE  = 16
SCAN_LIMIT  = None

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

ae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
ae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
ae.eval()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
if SCAN_LIMIT:
    all_images = all_images[:SCAN_LIMIT]

activations_list = []


with torch.no_grad():
    for i in tqdm(range(0, len(all_images), BATCH_SIZE)):
        batch_paths = all_images[i:i+BATCH_SIZE]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB"))
            except: continue
        
        if not images: continue
        
        inputs = proc(images=images, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        h_spatial = h[:, 1:, :].float()
        
        acts = ae.encode(h_spatial)
        max_acts = acts[:, :, FEATURE_IDX].max(dim=1).values.cpu().numpy()
        
        activations_list.extend(max_acts.tolist())

activations_np = np.array(activations_list)

zeros_count = np.sum(activations_np == 0)
non_zeros = activations_np[activations_np > 0]

plt.figure(figsize=(10, 6))

plt.hist(non_zeros, bins=50, color='orange', edgecolor='black', alpha=0.7)

plt.title(f"Activation Distribution for Food Feature ({FEATURE_IDX})\nDataset: test2014 | Total Images: {len(activations_np)}", fontsize=14)
plt.xlabel("Max Activation Value", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)

info_text = f"Images where feature fired: {len(non_zeros)}\nImages where feature was dead: {zeros_count}"
plt.gca().text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()