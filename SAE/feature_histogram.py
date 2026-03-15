import os
import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda: True 
os.environ["HF_HUB_DISABLE_TORCH_LOAD_SAFE_CHECK"] = "1"

import torch
original_load = torch.load
def trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = trusted_load

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True).to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
D_SAE = sae.dict_size

global_activations = torch.zeros(D_SAE, device=DEVICE)

for img_name in tqdm(all_images):
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            h = vision_tower(inputs.pixel_values.half(), output_hidden_states=True).hidden_states[11]
            f = sae.encode(h)[0, 1:, :]
            global_activations += f.sum(dim=0)
    except Exception:
        continue

data = global_activations.cpu().numpy()

dead_features = (data == 0).sum()

plt.figure(figsize=(12, 6))
plt.hist(data, bins=100, color='purple', alpha=0.7, log=True)
plt.title("Globálna hustota aktivácií pre všetky SAE Features", fontsize=16)
plt.xlabel("Súčet aktivácií", fontsize=12)
plt.ylabel("Počet Features", fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("activation_histogram.png", dpi=300)