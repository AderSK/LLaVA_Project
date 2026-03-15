import os, random, torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"

NUM_FEATURES = 10
TOP_X = 8 
MAX_SEARCH = 3000

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

target_fids = random.sample(range(sae.dict_size), NUM_FEATURES)
top_activations = {fid: [] for fid in target_fids}

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
MAX_SEARCH = len(all_images)
random.shuffle(all_images)
search_pool = all_images[:MAX_SEARCH]

for img_name in tqdm(search_pool):
    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
    except: continue
        
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
        f = sae.encode(h)[0, 1:, :]
        
    for fid in target_fids:
        max_act, patch_idx = f[:, fid].max(dim=0)
        
        if max_act.item() > 0:
            top_activations[fid].append((max_act.item(), img_path, patch_idx.item()))
            
            top_activations[fid].sort(key=lambda x: x[0], reverse=True)
            top_activations[fid] = top_activations[fid][:TOP_X]

fig, axes = plt.subplots(NUM_FEATURES, TOP_X, figsize=(3 * TOP_X, 3 * NUM_FEATURES))

for row_idx, fid in enumerate(target_fids):
    matches = top_activations[fid]
    
    for col_idx in range(TOP_X):
        ax = axes[row_idx, col_idx]
        ax.axis('off')
        
        if col_idx < len(matches):
            act_val, img_path, patch_idx = matches[col_idx]
            
            img = Image.open(img_path).convert("RGB").resize((336, 336))
            
            p_row = (patch_idx // 24) * 14
            p_col = (patch_idx % 24) * 14
            
            crop_box = (max(0, p_col-21), max(0, p_row-21), min(336, p_col+35), min(336, p_row+35))
            crop = img.crop(crop_box)
            
            ax.imshow(crop)
            
            if col_idx == 0:
                ax.set_title(f"FEATURE {fid}\nAct: {act_val:.1f}", fontsize=14, weight='bold', loc='left')
            else:
                ax.set_title(f"Act: {act_val:.1f}", fontsize=10)
        else:
            if col_idx == 0:
                ax.set_title(f"FEATURE {fid}\n(Dead/Rare)", fontsize=14, color='red')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("feature_kolaz.png", bbox_inches='tight', dpi=150)