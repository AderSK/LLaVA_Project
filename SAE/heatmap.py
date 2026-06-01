import os, sys, torch, heapq, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR  = "/home/adam/Projects/data/train2014"
SAE_PATH   = "/home/adam/Projects/trained_sae/ae_layer4_topk128.pt"
OUTPUT_DIR = "/home/adam/Projects/heatmaps"

DEVICE      = "cuda:0"
LAYER       = 4
FEATURE_IDX = 20551
BATCH_SIZE  = 32
TOP_K       = 5       
SCAN_LIMIT  = 5000    

os.makedirs(OUTPUT_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

ae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
ae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
ae.eval()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)
search_images = all_images # [:SCAN_LIMIT]

top_images_heap = []

with torch.no_grad():
    for i in tqdm(range(0, len(search_images), BATCH_SIZE)):
        batch_paths = search_images[i:i+BATCH_SIZE]
        images, valid_paths = [], []
        
        for p in batch_paths:
            try:
                images.append(Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB"))
                valid_paths.append(p)
            except: continue
                
        if not images: continue
        
        inputs = proc(images=images, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        h_spatial = h[:, 1:, :].float() 
        
        for b_idx in range(len(valid_paths)):
            acts = ae.encode(h_spatial[b_idx]) 
            max_act = acts[:, FEATURE_IDX].max().item()
            
            if max_act > 0:
                if len(top_images_heap) < TOP_K:
                    heapq.heappush(top_images_heap, (max_act, valid_paths[b_idx]))
                else:
                    heapq.heappushpop(top_images_heap, (max_act, valid_paths[b_idx]))

best_images = sorted(top_images_heap, key=lambda x: x[0], reverse=True)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

for rank, (act_val, img_name) in enumerate(best_images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    inputs['pixel_values'] = inputs['pixel_values'].half()
    
    with torch.no_grad():
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        spatial_h = h[:, 1:, :].float()
        acts = ae.encode(spatial_h.squeeze(0)) 
        feature_acts = acts[:, FEATURE_IDX].cpu().numpy()
        
    seen_image = (inputs['pixel_values'][0].cpu().float() * CLIP_STD + CLIP_MEAN)
    seen_image = seen_image.clamp(0, 1).permute(1, 2, 0).numpy()
        
    heatmap_24x24 = feature_acts.reshape(24, 24)
    
    if heatmap_24x24.max() > 0:
        heatmap_norm = heatmap_24x24 / heatmap_24x24.max()
    else:
        heatmap_norm = heatmap_24x24

    plt.figure(figsize=(8, 8))
    
    extent_bounds = (0, 24, 24, 0)
    plt.imshow(seen_image, extent=extent_bounds)
    plt.imshow(heatmap_norm, cmap='inferno', alpha=0.6, extent=extent_bounds)
    plt.axis('off')
    
    plt.title(f"Layer 18 | Feature {FEATURE_IDX} | Max Act: {act_val:.2f} | {img_name}", color='white', backgroundcolor='black', fontsize=14)

    save_path = os.path.join(OUTPUT_DIR, f"rank_{rank+1}_heatmap.jpg")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
