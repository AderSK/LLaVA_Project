import os, sys, torch, random
import plotly.graph_objects as go
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/Projects")) 
from dictionary_learning.trainers.top_k import AutoEncoderTopK

BASE_DIR   = os.path.expanduser("~/Projects")
IMAGE_DIR  = os.path.join(BASE_DIR, "data/test2014")
SAE_DIR    = os.path.join(BASE_DIR, "trained_sae")

DEVICE = "cuda:0"
LAYER_TO_ANALYZE = 12 
BATCH_SIZE = 16 
SAMPLE_SIZE = 1280 

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

sae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
sae.load_state_dict(torch.load(os.path.join(SAE_DIR, f"ae_layer{LAYER_TO_ANALYZE}_topk64.pt"), map_location=DEVICE))
sae.eval()

random.seed(42)
all_imgs = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
sankey_subset = random.sample(all_imgs, min(SAMPLE_SIZE, len(all_imgs)))

def get_max_activations(image_paths):
    all_max_acts = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        raw_imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in batch_paths]
        inputs = proc(images=raw_imgs, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        with torch.no_grad():
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER_TO_ANALYZE]
            encoded = sae.encode(h[:, 1:, :].float()) 
            max_acts = encoded.max(dim=1).values
            all_max_acts.append(max_acts.cpu())
            
    return torch.cat(all_max_acts, dim=0)

max_acts_matrix = get_max_activations(sankey_subset)
active_feature_mask = (max_acts_matrix > 0).sum(dim=0) > 0
active_feature_indices = torch.nonzero(active_feature_mask).squeeze().tolist()

TARGET_FEATURE = random.choice(active_feature_indices)
print(f"\n randomly selected Feature ID: {TARGET_FEATURE}")

feat_max_acts = max_acts_matrix[:, TARGET_FEATURE].numpy()
active_vals = feat_max_acts[feat_max_acts > 0]

print(f"Feature {TARGET_FEATURE} activated on {len(active_vals)} out of {SAMPLE_SIZE} images.")

fig = go.Figure(data=[
    go.Histogram(
        x=active_vals, 
        marker_color='#1f77b4',
        opacity=0.8,
        marker_line_width=1.5,
        marker_line_color='white',
        hovertemplate="Strength: %{x}<br>Sample Count: %{y}<extra></extra>"
    )
])

fig.update_layout(
    title=f"Sample Distribution for Feature {TARGET_FEATURE} | Layer {LAYER_TO_ANALYZE}<br><sup>Total Active Samples: {len(active_vals)} / {SAMPLE_SIZE}</sup>",
    xaxis_title="Peak Activation Strength",
    yaxis_title="Count of Samples (Images)",
    template="plotly_white",
    bargap=0.05 
)

output_html = f"Activating_Samples_L{LAYER_TO_ANALYZE}_Feature_{TARGET_FEATURE}.html"
fig.write_html(os.path.join(BASE_DIR, output_html))
print(f"saved to: {output_html}")