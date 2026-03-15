import os
import transformers.utils.import_utils as import_utils
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"
NUM_SEEDS = 5
TOP_F = 5
TOP_K_IMAGES = 5

OUTPUT_DIR = "resonance_outputs"
ORIGINALS_DIR = os.path.join(OUTPUT_DIR, "original_seeds")
os.makedirs(ORIGINALS_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.shuffle(all_images)
search_pool = all_images 

seeds = random.sample(search_pool, NUM_SEEDS)
target_features = {}
feature_tracker = {}
seed_heatmaps = {}

for seed in seeds:
    img = Image.open(os.path.join(IMAGE_DIR, seed)).convert("RGB")
    img.save(os.path.join(ORIGINALS_DIR, f"original_{seed}"))
    
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    seed_heatmaps[seed] = {}
    
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
        f = sae.encode(h)[0, 1:, :]
        
        top_fids = f.sum(dim=0).topk(TOP_F).indices.tolist()
        target_features[seed] = top_fids
        
        for fid in top_fids:
            feature_tracker[fid] = []
            seed_heatmaps[seed][fid] = f[:, fid].cpu().float().numpy().reshape(24, 24)

all_target_fids = list(feature_tracker.keys())

for img_name in tqdm(search_pool):
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
            f = sae.encode(h)[0, 1:, :]
        
        for fid in all_target_fids:
            max_act = f[:, fid].max().item()
            if max_act > 0:
                heatmap = f[:, fid].cpu().float().numpy().reshape(24, 24)
                feature_tracker[fid].append((max_act, img_name, heatmap))
    except:
        continue

for fid in all_target_fids:
    feature_tracker[fid].sort(key=lambda x: x[0], reverse=True)
    feature_tracker[fid] = feature_tracker[fid][:TOP_K_IMAGES]

for seed in seeds:
    fig, axes = plt.subplots(TOP_F, TOP_K_IMAGES + 1, figsize=(4 * (TOP_K_IMAGES + 1), 4 * TOP_F))
    seed_img = Image.open(os.path.join(IMAGE_DIR, seed)).convert("RGB")
    
    for row, fid in enumerate(target_features[seed]):
        ax_seed = axes[row, 0]
        ax_seed.imshow(seed_img)
        s_heatmap = seed_heatmaps[seed][fid]
        ax_seed.imshow(s_heatmap, cmap='jet', alpha=0.5, 
                       extent=(0, seed_img.size[0], seed_img.size[1], 0), interpolation='nearest')
        ax_seed.set_title(f"SEED: Feat {fid}", fontsize=12, weight='bold')
        ax_seed.axis('off')
        
        matches = feature_tracker[fid]
        for col in range(TOP_K_IMAGES):
            ax_match = axes[row, col + 1]
            if col < len(matches):
                act_val, match_name, m_heatmap = matches[col]
                match_img = Image.open(os.path.join(IMAGE_DIR, match_name)).convert("RGB")
                ax_match.imshow(match_img)
                ax_match.imshow(m_heatmap, cmap='jet', alpha=0.5, 
                                extent=(0, match_img.size[0], match_img.size[1], 0), interpolation='nearest')
                ax_match.set_title(f"Act: {act_val:.2f}", fontsize=10)
            ax_match.axis('off')
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resonance_seed_{seed.split('.')[0]}.png")
    plt.close()