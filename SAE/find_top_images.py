import shutil
import torch
import os
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

DEVICE = "cuda:0"
LAYER = 4
FEATURE_IDX = 20551
K = 128
SAVE_DIR = "top_activating_images"
IMAGE_DIR = "/home/cervenka25/large-data/train2014"
SAE_PATH = f"/home/cervenka25/LLaVA_Project/SAE/trained_sae/ae_layer{LAYER}_topk{K}.pt"

os.makedirs(SAVE_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
ae = AutoEncoderTopK(1024, 65536, K)
ae.load_state_dict(torch.load(SAE_PATH))
ae.to(DEVICE)

top_activations = []
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

i = 0
for img_name in all_images:
    i += 1
    if i % 1000 == 0:
        print(f"Processed {i}/{len(all_images)} images...")
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        h_spatial = h[:, 1:, :] 
        
        features = ae.encode(h_spatial.float())
        
        feat_acts = features[0, :, FEATURE_IDX] 
        max_act = feat_acts.max().item()
        
        top_activations.append((max_act, img_path))

top_activations.sort(key=lambda x: x[0], reverse=True)
for i, (score, path) in enumerate(top_activations[:20]):
    print(f"Saving top {i+1} image with score {score:.4f}: {path}")
    shutil.copy(path, os.path.join(SAVE_DIR, f"top_{i}_{score:.2f}.jpg"))