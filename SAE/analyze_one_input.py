import os, sys, torch, random
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.expanduser("~/Projects")) 
from dictionary_learning.trainers.top_k import AutoEncoderTopK

BASE_DIR   = os.path.expanduser("~/Projects")
IMAGE_DIR  = os.path.join(BASE_DIR, "data/test2014")
SAE_DIR    = os.path.join(BASE_DIR, "trained_sae")

DEVICE = "cuda:0"
LAYER_TO_ANALYZE = 12 

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

sae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
sae.load_state_dict(torch.load(os.path.join(SAE_DIR, f"ae_layer{LAYER_TO_ANALYZE}_topk64.pt"), map_location=DEVICE))
sae.eval()

random.seed(42)
all_imgs = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random_image = random.choice(all_imgs)
print(f"{random_image}")

raw_img = Image.open(os.path.join(IMAGE_DIR, random_image)).convert("RGB")
inputs = proc(images=raw_img, return_tensors="pt").to(DEVICE)
inputs['pixel_values'] = inputs['pixel_values'].half()

with torch.no_grad():
    h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER_TO_ANALYZE]
    acts = sae.encode(h[:, 1:, :].float()).max(dim=1).values.squeeze().cpu().numpy()

LIMIT = 1000
display_feats = list(range(LIMIT))
display_acts = acts[:LIMIT]

zero_count = np.sum(display_acts == 0)
active_count = LIMIT - zero_count

print(f"\n" + "="*50)
print(f"ANALÝZA PRE OBRÁZOK: {random_image}")
print(f"ROZSAH: Prvých {LIMIT} featuriek")
print(f"="*50)
print(f"-> Aktívnych featuriek: {active_count}")
print(f"-> Mŕtvych featuriek (0): {zero_count}")
print(f"-> Sparsity : {(zero_count/LIMIT)*100:.1f} %")
print(f"="*50 + "\n")

hover_texts = [
    f"<b>Feature ID:</b> {feat}<br>"
    f"<b>Sila Aktivácie:</b> {val:.3f}" 
    for feat, val in zip(display_feats, display_acts)
]

fig = go.Figure(data=[
    go.Bar(
        x=[f"F {f}" for f in display_feats], 
        y=display_acts, 
        hoverinfo="text",
        hovertext=hover_texts,
        marker_color='#d62728'
    )
])

fig.update_layout(
    yaxis_type="log",
    bargap=0,
    title=f"Fingerprint: Prvých {LIMIT} Featuriek | Obrázok: {random_image}<br><sup>Aktívne: {active_count} | Mŕtve: {zero_count} (Sparsity: {(zero_count/LIMIT)*100:.1f}%)</sup>",
    xaxis_title=f"SAE Feature ID (0 - {LIMIT-1})",
    yaxis_title="Sila Aktivácie (Log Scale)",
    template="plotly_white",
    xaxis=dict(showticklabels=False)
)

output_html = f"L{LAYER_TO_ANALYZE}_single_1000_fingerprint.html"
fig.write_html(os.path.join(BASE_DIR, output_html))
print(f"✅ Graf uložený ako: {output_html}")