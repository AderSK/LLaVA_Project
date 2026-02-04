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
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", quantization_config=bnb_config, device_map="auto")

checkpoint = torch.load(SAE_FILE, map_location=device)
pre_b = checkpoint['model_state_dict']['pre_b'].to(device).half()
W_enc = checkpoint['model_state_dict']['enc'].to(device).half()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
seed_images = random.sample(all_images, 5)

vision_tower = model.model.vision_tower
seeds = []
all_features = {}

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
        heatmaps[int(fid)] = latents[0, 1:, fid].cpu().numpy().reshape(24, 24)
        if fid.item() not in all_features:
            all_features[fid.item()] = []
    
    seeds.append({'img': img, 'name': seed_img, 'features': top_ids.cpu().numpy(), 'vals': top_vals.cpu().numpy(), 'heatmaps': heatmaps})

for img_name in tqdm(all_images):
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        inputs = processor(text="Describe the image", images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[LAYER]
            latents = torch.relu((h - pre_b) @ W_enc)
        
        acts = latents[0, 1:, :]
        
        for fid in all_features.keys():
            activation = acts[:, fid].sum().item()
            heatmap = acts[:, fid].cpu().numpy().reshape(24, 24)
            all_features[fid].append({'act': activation, 'path': os.path.join(IMAGE_DIR, img_name), 'name': img_name, 'heatmap': heatmap})
    except:
        pass

for fid in all_features.keys():
    all_features[fid] = sorted(all_features[fid], key=lambda x: x['act'], reverse=True)[:5]

for i, seed in enumerate(seeds):
    fig = plt.figure(figsize=(20, 12))
    
    plt.subplot(6, 11, 1)
    plt.imshow(seed['img'])
    plt.title(seed['name'], fontsize=8)
    plt.axis('off')
    
    for j, fid in enumerate(seed['features']):
        plt.subplot(6, 11, j*2 + 2)
        plt.imshow(seed['heatmaps'][fid], cmap='viridis')
        plt.title(f"F{fid}", fontsize=7)
        plt.axis('off')
        
        for k, ex in enumerate(all_features[fid]):
            plt.subplot(6, 11, (j+1)*11 + k*2 + 1)
            plt.imshow(Image.open(ex['path']))
            plt.title(f"{ex['act']:.0f}", fontsize=6)
            plt.axis('off')
            
            plt.subplot(6, 11, (j+1)*11 + k*2 + 2)
            plt.imshow(ex['heatmap'], cmap='viridis')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"seed{i+1}.png"), dpi=120)
    plt.close()