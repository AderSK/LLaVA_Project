import os, random, torch

original_load = torch.load
def trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = trusted_load

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"
NUM_FEATURES = 10
MAX_SEARCH = 5000

vision_tower = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14-336", 
    use_safetensors=True
).to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

D_SAE = sae.dict_size
target_fids = random.sample(range(D_SAE), NUM_FEATURES)
distributions = {fid: [] for fid in target_fids}

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
MAX_SEARCH = len(all_images)
random.shuffle(all_images)

for img_name in tqdm(all_images[:MAX_SEARCH]):
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
        f = sae.encode(h)[0, 1:, :]
        
    for fid in target_fids:
        acts = f[:, fid]
        non_zero = acts[acts > 0].cpu().numpy()
        distributions[fid].extend(non_zero)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, fid in enumerate(target_fids):
    data = distributions[fid]
    if len(data) == 0:
        axes[i].text(0.5, 0.5, "DEAD FEATURE", ha='center', color='red', weight='bold')
    else:
        axes[i].hist(data, bins=50, color='teal', alpha=0.7, log=True)
    
    axes[i].set_title(f"Feature {fid} (n={len(data)})")
    axes[i].set_xlabel("Activation Strength")
    axes[i].set_ylabel("Log Frequency")

plt.tight_layout()
plt.savefig("feature_distributions.png")