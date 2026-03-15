import os
import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda: True 
os.environ["HF_HUB_DISABLE_TORCH_LOAD_SAFE_CHECK"] = "1"

import torch
original_load = torch.load
def trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = trusted_load

import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"
NUM_FEATURES = 50
GRID_SIZE = 10

OUTPUT_DIR = "massive_kolaze"
os.makedirs(OUTPUT_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True).to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

D_SAE = sae.dict_size
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
target_fids = random.sample(range(D_SAE), NUM_FEATURES)
feature_tracker = {fid: [] for fid in target_fids}

for img_name in tqdm(all_images):
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
            f = sae.encode(h)[0, 1:, :]
            
        for fid in target_fids:
            max_act = f[:, fid].max().item()
            
            if max_act > 0:
                feature_tracker[fid].append((max_act, img_name))
                
                if len(feature_tracker[fid]) > GRID_SIZE * GRID_SIZE * 2:
                    feature_tracker[fid].sort(key=lambda x: x[0], reverse=True)
                    feature_tracker[fid] = feature_tracker[fid][:GRID_SIZE * GRID_SIZE]
    except Exception:
        continue

for fid in tqdm(target_fids):
    matches = sorted(feature_tracker[fid], key=lambda x: x[0], reverse=True)[:GRID_SIZE * GRID_SIZE]
    
    if len(matches) < 10:
        continue

    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(20, 20))
    fig.subplots_adjust(wspace=0.2, hspace=0.02)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(matches):
            act_val, match_name = matches[i]
            match_img = Image.open(os.path.join(IMAGE_DIR, match_name)).convert("RGB")
            
            ax.imshow(match_img)
            ax.text(0.05, 0.95, f"{act_val:.1f}", transform=ax.transAxes, color='lime', fontsize=12, weight='bold', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, pad=1))
        ax.axis('off')
        
    plt.suptitle(f"Feature {fid} | Top {len(matches)} Images", fontsize=24, weight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"velka_kolaz_feat_{fid}.jpg"), dpi=150, bbox_inches='tight')
    plt.close()