import os, heapq, torch, glob, re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoImageProcessor, AutoModel
import torch.nn.functional as F

IMAGE_DIR   = "/home/adam/Projects/data/train2014"
BASE_OUTPUT = "/home/adam/Projects/sae_collages_MS_neurons_backup"

DEVICE    = "cuda:0"
LAYER     = 18
D_MODEL   = 1024
BATCH     = 32
GRID_SIZE    = 10
IMG_SIZE     = 100
HEADER_H     = 50

OUTPUT_DIR = os.path.join(BASE_OUTPUT, f"CLIP_Layer_{LAYER}_Neurons_MS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

existing_files = glob.glob(os.path.join(OUTPUT_DIR, f"collage_L{LAYER}_Neuron*_MS*.jpg"))
done_neurons = set()
for f in existing_files:
    match = re.search(r'Neuron(\d+)_MS', os.path.basename(f))
    if match:
        done_neurons.add(int(match.group(1)))

target_neurons = [n for n in range(D_MODEL) if n not in done_neurons]

if not target_neurons:
    import sys
    sys.exit()

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
dino_proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(DEVICE)
dino_model.eval()

def get_dino_embeddings(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = dino_proc(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    return F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)

def calculate_ms_score(embeddings, activations):
    N = activations.size(0)
    similarities = torch.matmul(embeddings, embeddings.T)
    a_min, a_max = activations.min(), activations.max()
    if a_max == a_min: return 0.0
    a_norm = (activations - a_min) / (a_max - a_min)
    relevance = torch.outer(a_norm, a_norm)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=DEVICE), diagonal=1)
    valid_relevance = relevance[mask]
    valid_similarities = similarities[mask]
    if valid_relevance.sum() == 0: return 0.0
    return ((valid_relevance * valid_similarities).sum() / valid_relevance.sum()).item()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
top_images = {neuron: [] for neuron in target_neurons}

with torch.no_grad():
    for i in tqdm(range(0, len(all_images), BATCH)):
        batch_paths = all_images[i:i+BATCH]
        valid_paths = [img for img in batch_paths if os.path.exists(os.path.join(IMAGE_DIR, img))]
        if not valid_paths: continue
        imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in valid_paths]
        inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        max_acts, _ = h[:, 1:, :].float().max(dim=1)
        
        for b_idx in range(len(valid_paths)):
            for neuron in target_neurons:
                max_act = max_acts[b_idx, neuron].item()
                heapq.heappush(top_images[neuron], (max_act, valid_paths[b_idx]))
                if len(top_images[neuron]) > (GRID_SIZE * GRID_SIZE):
                    heapq.heappop(top_images[neuron])

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
except IOError:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

for neuron in tqdm(target_neurons):
    best_images = sorted(top_images[neuron], key=lambda x: x[0], reverse=True)
    if len(best_images) < 2: continue 
        
    activations_tensor = torch.tensor([x[0] for x in best_images], device=DEVICE)
    img_paths = [os.path.join(IMAGE_DIR, x[1]) for x in best_images]
    ms_score = calculate_ms_score(get_dino_embeddings(img_paths), activations_tensor)
    
    canvas_w, canvas_h = GRID_SIZE * IMG_SIZE, (GRID_SIZE * IMG_SIZE) + HEADER_H
    collage = Image.new('RGB', (canvas_w, canvas_h), color="black")
    draw = ImageDraw.Draw(collage)
    draw.text((10, 15), f"CLIP L{LAYER} | OG Neuron {neuron} | MS Score: {ms_score:.4f}", fill="white", font=font)
    
    for idx, (act_val, filename) in enumerate(best_images):
        try:
            img = Image.open(os.path.join(IMAGE_DIR, filename)).convert("RGB")
            min_dim = min(img.width, img.height)
            left, top = (img.width - min_dim) / 2, (img.height - min_dim) / 2
            img = img.crop((left, top, left + min_dim, top + min_dim)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            
            x_offset, y_offset = (idx % GRID_SIZE) * IMG_SIZE, ((idx // GRID_SIZE) * IMG_SIZE) + HEADER_H
            collage.paste(img, (x_offset, y_offset))
            draw.rectangle([x_offset, y_offset, x_offset + 40, y_offset + 18], fill="black")
            draw.text((x_offset + 2, y_offset + 2), f"{act_val:.1f}", fill="cyan", font=small_font)
        except Exception:
            pass
            
    collage.save(os.path.join(OUTPUT_DIR, f"collage_L{LAYER}_Neuron{neuron}_MS{ms_score:.2f}.jpg"))