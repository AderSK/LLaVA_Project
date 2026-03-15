import os, random, torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"

NUM_FEATURES = 10
TOP_X = 6 
BATCH_SIZE = 32 
MAX_SEARCH = 5000

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

target_fids = random.sample(range(sae.dict_size), NUM_FEATURES)
feature_matches = {fid: [] for fid in target_fids}

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.shuffle(all_images)
search_pool = all_images[:MAX_SEARCH]

for i in tqdm(range(0, len(search_pool), BATCH_SIZE)):
    batch_paths = search_pool[i:i+BATCH_SIZE]
    valid_imgs, valid_full_paths = [], []
    
    for p in batch_paths:
        full_p = os.path.join(IMAGE_DIR, p)
        try:
            valid_imgs.append(Image.open(full_p).convert("RGB"))
            valid_full_paths.append(full_p)
        except: continue
    
    if not valid_imgs: continue

    inputs = proc(images=valid_imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
        f = sae.encode(h)[:, 1:, :] 
        
    for fid in target_fids:
        max_acts, _ = f[:, :, fid].max(dim=1) 
        
        for b_idx, act in enumerate(max_acts):
            if act.item() > 0:
                feature_matches[fid].append((act.item(), valid_full_paths[b_idx]))
                feature_matches[fid].sort(key=lambda x: x[0], reverse=True)
                feature_matches[fid] = feature_matches[fid][:TOP_X]

fig, axes = plt.subplots(NUM_FEATURES, TOP_X, figsize=(4 * TOP_X, 4 * NUM_FEATURES))

for row_idx, fid in enumerate(target_fids):
    matches = feature_matches[fid]
    
    for col_idx in range(TOP_X):
        ax = axes[row_idx, col_idx]
        ax.axis('off')
        
        if col_idx < len(matches):
            act_val, img_path = matches[col_idx]
            img = Image.open(img_path).convert("RGB")
            
            ax.imshow(img)
            ax.text(5, 5, f"Act: {act_val:.1f}", color='white', backgroundcolor='black', 
                    fontsize=10, verticalalignment='top')
            
            if col_idx == 0:
                ax.set_title(f"FEATURE {fid}", fontsize=16, weight='bold', loc='left', color='blue')
        else:
            if col_idx == 0:
                ax.set_title(f"FEATURE {fid} (No Activations)", fontsize=12, color='red')

plt.tight_layout()
plt.savefig("kolaz_full_images.png", dpi=100)