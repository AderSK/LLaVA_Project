import os, sys, torch, random, gc
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.expanduser("~/LLaVA_Project/SAE"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR  = "/home/cervenka25/large-data/test2014"
SAE_DIR    = "/home/cervenka25/LLaVA_Project/SAE/trained_sae"
OUTPUT_CSV = "/home/cervenka25/LLaVA_Project/SAE/feature_paths_all_layers.csv"

DEVICE      = "cuda:0"
LAYERS      = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] 
BATCH_SIZE  = 16 
SAMPLE_SIZE = 1280

print("Loading CLIP Vision Tower...")
vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.seed(42) 
sampled_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))

master_results = {img: {"image": img} for img in sampled_images}

for L in LAYERS:
    
    sae_path = os.path.join(SAE_DIR, f"ae_layer{L}_topk64.pt")
    sae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
    sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    sae.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sampled_images), BATCH_SIZE)):
            batch_paths = sampled_images[i:i+BATCH_SIZE]
            raw_images, valid_paths = [], []
            for p in batch_paths:
                try:
                    raw_images.append(Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB"))
                    valid_paths.append(p)
                except: continue
                    
            if not raw_images: continue
            
            inputs = proc(images=raw_images, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[L]
            acts = sae.encode(h[:, 1:, :].float()) 
            
            max_acts_per_image = acts.max(dim=1).values
            top_feats = torch.max(max_acts_per_image, dim=1).indices
            
            for b_idx, img_name in enumerate(valid_paths):
                master_results[img_name][f"L{L}_feature"] = top_feats[b_idx].item()
    
    del sae
    torch.cuda.empty_cache()
    gc.collect()

final_data = [data for data in master_results.values() if f"L{LAYERS[0]}_feature" in data]
df = pd.DataFrame(final_data)
df.to_csv(OUTPUT_CSV, index=False)