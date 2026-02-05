import torch
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

LAYER = 11
BASE_PATH = "/home/cervenka25/large-data"
IMAGE_DIR = os.path.join(BASE_PATH, "test2014")
SAE_FILE = os.path.join(BASE_PATH, "CLIP-ViT-L-14-SAE-L11/11_resid/1000104192.pt")
OUTPUT_DIR = "/home/cervenka25/LLaVA_Project/SAE/multi_seed"
device = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", 
    quantization_config=bnb_config, device_map="auto")

checkpoint = torch.load(SAE_FILE, map_location=device)
pre_b = checkpoint['model_state_dict']['pre_b'].to(device).half()
W_enc = checkpoint['model_state_dict']['enc'].to(device).half()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
seed_images = random.sample(all_images, 5)
seed_image_set = set(seed_images)

vision_tower = model.model.vision_tower
seeds = []
all_features = {}

# Process seed images
for seed_img in seed_images:
    img = Image.open(os.path.join(IMAGE_DIR, seed_img)).convert("RGB")
    inputs = processor(text="Describe the image", images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[LAYER]
        latents = torch.relu((h - pre_b) @ W_enc)
    
    acts = latents[0, 1:, :].sum(dim=0)
    top_vals, top_ids = torch.topk(acts, 5)
    
    heatmaps = {}
    for fid in top_ids.cpu().numpy():
        fid_int = int(fid)
        heatmaps[fid_int] = latents[0, 1:, fid].cpu().numpy().reshape(24, 24)
        all_features[fid_int] = []  # FIXED: No conditional check
    
    seeds.append({'img': img, 'name': seed_img, 'features': top_ids.cpu().numpy(), 
                  'vals': top_vals.cpu().numpy(), 'heatmaps': heatmaps})

# Process all images - store only activation values, not heatmaps
for img_name in tqdm(all_images):
    if img_name in seed_image_set:
        continue
    
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        inputs = processor(text="Describe the image", images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[LAYER]
            latents = torch.relu((h - pre_b) @ W_enc)
        
        acts = latents[0, 1:, :]
        
        for fid in all_features.keys():
            activation = acts[:, fid].sum().item()
            # FIXED: Only store activation and path, compute heatmap later
            all_features[fid].append({'act': activation, 
                                      'path': os.path.join(IMAGE_DIR, img_name), 
                                      'name': img_name})
    except:
        pass

# Sort and keep top 5 for each feature
for fid in all_features.keys():
    all_features[fid] = sorted(all_features[fid], key=lambda x: x['act'], reverse=True)[:5]

# NOW compute heatmaps only for top 5 images per feature
print("Computing heatmaps for top activating images...")
for fid in tqdm(all_features.keys()):
    for ex in all_features[fid]:
        img = Image.open(ex['path']).convert("RGB")
        inputs = processor(text="Describe the image", images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[LAYER]
            latents = torch.relu((h - pre_b) @ W_enc)
        
        ex['heatmap'] = latents[0, 1:, fid].cpu().numpy().reshape(24, 24)

# Visualization (unchanged)
for i, seed in enumerate(seeds):
    fig = plt.figure(figsize=(22, 14))
    
    ax_seed = plt.subplot(6, 11, 1)
    ax_seed.imshow(seed['img'])
    ax_seed.set_title(f"SEED {i+1}\n{seed['name']}", fontsize=9, fontweight='bold')
    ax_seed.axis('off')
    
    for j, fid in enumerate(seed['features']):
        ax_heat = plt.subplot(6, 11, j*2 + 2)
        im = ax_heat.imshow(seed['heatmaps'][fid], cmap='viridis')
        ax_heat.set_title(f"Feature {fid}\nAct: {seed['vals'][j]:.1f}", fontsize=8, fontweight='bold')
        ax_heat.axis('off')
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        
        for k, ex in enumerate(all_features[fid]):
            ax_img = plt.subplot(6, 11, (j+1)*11 + k*2 + 1)
            ax_img.imshow(Image.open(ex['path']))
            ax_img.set_title(f"Act: {ex['act']:.1f}", fontsize=7)
            ax_img.axis('off')
            
            ax_ex_heat = plt.subplot(6, 11, (j+1)*11 + k*2 + 2)
            ax_ex_heat.imshow(ex['heatmap'], cmap='viridis')
            ax_ex_heat.axis('off')
    
    plt.suptitle(f"Seed {i+1}: {seed['name']} - Top 5 Features & Their Maximum Activating Images", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"seed{i+1}_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()