import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from transformers import CLIPVisionModel, CLIPImageProcessor

SAE_FILE = "/home/cervenka25/large-data/CLIP-ViT-L-14-SAE-L11/11_resid/1000104192.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

checkpoint = torch.load(SAE_FILE, map_location=DEVICE)
sd = checkpoint['model_state_dict']
pre_b = sd['pre_b'].to(DEVICE).half()
W_enc = sd['enc'].to(DEVICE).half()

img_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:5]
all_acts = []

for name in img_names:
    img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
    
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    inputs['pixel_values'] = inputs['pixel_values'].half() 
    
    with torch.no_grad():
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[11]
        f = torch.relu((h - pre_b) @ W_enc)[0, 1:, :] 
        all_acts.append(f.cpu().float())

avg_acts = torch.stack(all_acts).mean(dim=[0, 1])
top_fid = int(avg_acts.argmax())

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, name in enumerate(img_names):
    img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
    heatmap = all_acts[i][:, top_fid].reshape(24, 24).numpy()
    
    axes[i].imshow(img)
    axes[i].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[i].axis('off')
    axes[i].set_title(f"Img {i+1}")

plt.suptitle(f"Legacy SAE Feature {top_fid} (Most Active in Sample)")
plt.savefig("mini_analysis.png")
print(f"Success! Saved mini_analysis.png. Feature {top_fid} is being visualized.")