import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH = "/home/cervenka25/LLaVA_Project/SAE/checkpoints/ae.pt"
IMG_PATH = "/home/cervenka25/large-data/test2014/COCO_test2014_000000000001.jpg"
DEVICE = "cuda"

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

img = Image.open(IMG_PATH).convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(DEVICE).half()
with torch.no_grad():
    h = vision_tower(**inputs, output_hidden_states=True).hidden_states[11]

with torch.no_grad():
    f = sae.encode(h)
    h_recon = sae.decode(f)

mse = F.mse_loss(h, h_recon)
l0 = (f > 0).float().sum() / f.shape[1]
print(f"Reconstruction MSE: {mse.item():.6f}")
print(f"Mean L0 (Sparsity): {l0.item():.2f}")