import os, random, sys
sys.path.append(os.path.abspath("dictionary_learning"))
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR = "/home/cervenka25/large-data/test2014"
SAE_DIR   = "/home/cervenka25/large-data/trained_sae"
DEVICE    = "cuda:0"

D_MODEL  = 1024
D_SAE    = 65536
K_VALUES = [8, 16, 32, 64, 128]
BATCH    = 16
LAYER    = 18 

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half().eval()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
print(f"Found {len(all_images)} test images.")

def get_batches(image_list, batch_size):
    for i in range(0, len(image_list), batch_size):
        yield image_list[i:i+batch_size]

def create_collage(image_paths_grid, save_path):
    w, h = 224, 224
    collage = Image.new('RGB', (w * 5, h * 5), color=(0, 0, 0))
    for row, paths in enumerate(image_paths_grid):
        for col, path in enumerate(paths):
            if path:
                try:
                    full_path = os.path.join(IMAGE_DIR, path)
                    img = Image.open(full_path).convert('RGB').resize((w, h))
                    collage.paste(img, (col * w, row * h))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    collage.save(save_path)
    print(f"Saved collage to {save_path}")

for K in K_VALUES:
    print(f"\nEvaluating SAE with K={K}...")
    sae_path = os.path.join(SAE_DIR, f"ae_layer{LAYER}_topk{K}.pt")
    
    if not os.path.exists(sae_path):
        print(f"Model file missing: {sae_path}")
        continue
        
    sae = AutoEncoderTopK(D_MODEL, D_SAE, K).to(DEVICE)
    sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    sae.eval()

    top_scores = torch.full((D_SAE, 5), -1.0, device=DEVICE)
    top_img_idx = torch.full((D_SAE, 5), -1, dtype=torch.long, device=DEVICE)
    
    processed_paths = []
    global_idx_counter = 0

    with torch.no_grad():
        for batch_paths in get_batches(all_images, BATCH):
            images = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB")
                    images.append(img)
                    valid_paths.append(p)
                except: continue
                
            if not images: continue
            
            inputs = proc(images=images, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
            acts = h[:, 1:, :].reshape(len(valid_paths), -1, D_MODEL) 
            
            acts_flat = acts.reshape(-1, D_MODEL).float()
            
            z_flat = sae.encode(acts_flat)
            z = z_flat.reshape(len(valid_paths), -1, D_SAE)
            
            z_max_per_img, _ = z.max(dim=1)

            for b in range(len(valid_paths)):
                processed_paths.append(valid_paths[b])
                curr_idx = global_idx_counter
                global_idx_counter += 1
                
                curr_scores = z_max_per_img[b]
                active_feats = curr_scores.nonzero(as_tuple=True)[0]
                
                for f in active_feats:
                    s = curr_scores[f].item()
                    if s > top_scores[f, -1]:
                        top_scores[f, -1] = s
                        top_img_idx[f, -1] = curr_idx
                        
                        sorted_idx = torch.argsort(top_scores[f], descending=True)
                        top_scores[f] = top_scores[f][sorted_idx]
                        top_img_idx[f] = top_img_idx[f][sorted_idx]

    max_scores_per_feat = top_scores[:, 0]
    top_5_features = torch.topk(max_scores_per_feat, 5).indices.tolist()

    print(f"Top 5 activating features for K={K}: {top_5_features}")

    collage_paths = []
    for feat in top_5_features:
        feat_img_paths = []
        for rank in range(5):
            idx = top_img_idx[feat, rank].item()
            if idx != -1:
                feat_img_paths.append(processed_paths[idx])
            else:
                feat_img_paths.append(None)
        collage_paths.append(feat_img_paths)

    collage_filename = os.path.join(SAE_DIR, f"collage_layer{LAYER}_topk{K}.png")
    create_collage(collage_paths, collage_filename)