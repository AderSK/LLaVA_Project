import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
import json
from datetime import datetime
import shutil

LAYER = 11
BASE_PATH = "/home/cervenka25/large-data"
IMAGE_DIR = os.path.join(BASE_PATH, "test2014")
SAE_FILE = os.path.join(BASE_PATH, "CLIP-ViT-L-14-SAE-L11/11_resid/1000104192.pt")
OUTPUT_DIR = "/home/cervenka25/LLaVA_Project/SAE/max_activations"
device = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_FEATURES = 6
N_SEARCH_IMAGES = 1000
N_MAX_EXAMPLES = 6 

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
seed_image_name = random.choice(all_images)
seed_image_path = os.path.join(IMAGE_DIR, seed_image_name)

seed_image = Image.open(seed_image_path).convert("RGB")

seed_output_path = os.path.join(OUTPUT_DIR, f"seed_{seed_image_name}")
shutil.copy(seed_image_path, seed_output_path)

prompt = "Describe the image"
inputs = processor(text=prompt, images=seed_image, return_tensors="pt").to(device)

with torch.no_grad():
    vision_tower = model.model.vision_tower
    v_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
    h = v_outputs.hidden_states[LAYER]
    latents = torch.relu((h - pre_b) @ W_enc)

spatial_acts = latents[0, 1:, :]
feature_sums = spatial_acts.sum(dim=0)
top_k_values, top_k_indices = torch.topk(feature_sums, TOP_K_FEATURES)

print(f"Top {TOP_K_FEATURES} features v seed obrázku:")
for i, (idx, val) in enumerate(zip(top_k_indices, top_k_values)):
    print(f"    {i+1}. Feature {idx.item()}: aktivácia {val.item():.2f}")

seed_heatmaps = {}
for feature_id in top_k_indices.cpu().numpy():
    heatmap = spatial_acts[:, feature_id].cpu().numpy().reshape(24, 24)
    seed_heatmaps[int(feature_id)] = heatmap

fig, axes = plt.subplots(2, TOP_K_FEATURES + 1, figsize=(20, 8))

axes[0, 0].imshow(seed_image)
axes[0, 0].set_title(f"SEED IMAGE\n{seed_image_name}", fontsize=10, fontweight='bold')
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for i, feature_id in enumerate(top_k_indices.cpu().numpy()):
    axes[0, i+1].text(0.5, 0.5, f"Feature {feature_id}\nAct: {top_k_values[i].item():.1f}", 
    ha='center', va='center', fontsize=9, fontweight='bold')
    axes[0, i+1].axis('off')
    
    im = axes[1, i+1].imshow(seed_heatmaps[feature_id], cmap='viridis', interpolation='bilinear')
    axes[1, i+1].axis('off')
    plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)

plt.suptitle(f"Seed Image Analysis - Top {TOP_K_FEATURES} Features", fontsize=14, fontweight='bold')
plt.tight_layout()
seed_viz_path = os.path.join(OUTPUT_DIR, "seed_analysis.png")
plt.savefig(seed_viz_path, dpi=150, bbox_inches='tight')
plt.close()

max_examples = {idx.item(): [] for idx in top_k_indices}
search_images = random.sample(all_images, min(N_SEARCH_IMAGES, len(all_images)))

processed = 0
errors = 0

for img_name in tqdm(search_images, desc="Spracovávam obrázky"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            v_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
            h = v_outputs.hidden_states[LAYER]
            latents = torch.relu((h - pre_b) @ W_enc)
            
            spatial_acts = latents[0, 1:, :]
            
            for feature_id in max_examples.keys():
                activation = spatial_acts[:, feature_id].sum().item()
                heatmap = spatial_acts[:, feature_id].cpu().numpy().reshape(24, 24)
                
                max_examples[feature_id].append({
                    'activation': activation,
                    'image_path': img_path,
                    'image_name': img_name,
                    'heatmap': heatmap
                })
        
        processed += 1
    
    except Exception as e:
        errors += 1

print(f"\n  Úspešne spracovaných: {processed}/{N_SEARCH_IMAGES}")
if errors > 0:
    print(f"  Chyby: {errors}")

for feature_id in max_examples.keys():
    max_examples[feature_id] = sorted(
        max_examples[feature_id], 
        key=lambda x: x['activation'], 
        reverse=True
    )[:N_MAX_EXAMPLES]

metadata = {
    'seed_image': seed_image_name,
    'seed_image_path': f"seed_{seed_image_name}",
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_images_searched': processed,
    'layer': LAYER,
    'features': {}
}

for feat_idx, feature_id in enumerate(top_k_indices.cpu().numpy()):
    metadata['features'][int(feature_id)] = {
        'rank': feat_idx + 1,
        'seed_activation': float(top_k_values[feat_idx]),
        'max_examples': [
            {
                'image': ex['image_name'],
                'activation': float(ex['activation'])
            }
            for ex in max_examples[feature_id]
        ]
    }

metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

fig = plt.figure(figsize=(24, 4 * TOP_K_FEATURES))

for feat_idx, feature_id in enumerate(top_k_indices.cpu().numpy()):
    examples = max_examples[feature_id]
    
    for ex_idx, example in enumerate(examples):
        img = Image.open(example['image_path']).convert("RGB")
        
        ax_img = plt.subplot(TOP_K_FEATURES, N_MAX_EXAMPLES * 2, 
                             feat_idx * N_MAX_EXAMPLES * 2 + ex_idx * 2 + 1)
        ax_img.imshow(img)
        ax_img.axis('off')
        
        if ex_idx == 0:
            ax_img.set_title(f"Feature {feature_id} (seed: {top_k_values[feat_idx].item():.1f})\n{example['image_name']}\nAct: {example['activation']:.1f}", 
                           fontsize=9, fontweight='bold')
        else:
            ax_img.set_title(f"{example['image_name']}\nAct: {example['activation']:.1f}", 
                           fontsize=8)
        
        ax_heat = plt.subplot(TOP_K_FEATURES, N_MAX_EXAMPLES * 2, 
                             feat_idx * N_MAX_EXAMPLES * 2 + ex_idx * 2 + 2)
        im = ax_heat.imshow(example['heatmap'], cmap='viridis', interpolation='bilinear')
        ax_heat.axis('off')

plt.suptitle(f"Maximum Activating Examples for Top {TOP_K_FEATURES} Features\n(Seed: {seed_image_name}, Searched: {processed} images)", 
             fontsize=14, y=0.998)
plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "feature_max_activations.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()


for feat_idx, feature_id in enumerate(top_k_indices.cpu().numpy()):
    fig, axes = plt.subplots(2, N_MAX_EXAMPLES, figsize=(18, 6))
    
    examples = max_examples[feature_id]
    
    for ex_idx, example in enumerate(examples):
        img = Image.open(example['image_path']).convert("RGB")
        
        axes[0, ex_idx].imshow(img)
        axes[0, ex_idx].axis('off')
        axes[0, ex_idx].set_title(f"Act: {example['activation']:.1f}\n{example['image_name']}", 
                                  fontsize=8)
        
        axes[1, ex_idx].imshow(example['heatmap'], cmap='viridis', interpolation='bilinear')
        axes[1, ex_idx].axis('off')
    
    plt.suptitle(f"Feature {feature_id} - Top {N_MAX_EXAMPLES} Activating Images (Seed act: {top_k_values[feat_idx].item():.1f})", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    feature_file = os.path.join(OUTPUT_DIR, f"feature_{feature_id}.png")
    plt.savefig(feature_file, dpi=120, bbox_inches='tight')
    plt.close()
