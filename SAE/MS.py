import os, random, heapq, sys, torch, glob, re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoImageProcessor, AutoModel
import torch.nn.functional as F

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR   = "/home/adam/Projects/data/train2014"
SAE_PATH    = "/home/adam/Projects/trained_sae_backup/ae_layer18_topk64.pt" 
BASE_OUTPUT = "/home/adam/Projects/sae_collages_MS_backup"

DEVICE    = "cuda:0"
LAYER     = 18
D_MODEL   = 1024
D_SAE     = 65536
K         = 64
BATCH     = 32
TARGET_TOTAL = 1024
GRID_SIZE    = 10
IMG_SIZE     = 100
HEADER_H     = 50

OUTPUT_DIR = os.path.join(BASE_OUTPUT, f"Layer_{LAYER}_TopK_{K}_MS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

existing_files = glob.glob(os.path.join(OUTPUT_DIR, f"collage_L{LAYER}_F*_MS*.jpg"))
done_features = set()
for f in existing_files:
    match = re.search(r'_F(\d+)_MS', os.path.basename(f))
    if match:
        done_features.add(int(match.group(1)))

num_to_generate = TARGET_TOTAL - len(done_features)

if num_to_generate <= 0:
    sys.exit()

available_features = [f for f in range(D_SAE) if f not in done_features]
target_features = random.sample(available_features, num_to_generate)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
ae = AutoEncoderTopK(D_MODEL, D_SAE, K).to(DEVICE)
ae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
ae.eval()

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
top_images = {feat: [] for feat in target_features}

with torch.no_grad():
    for i in tqdm(range(0, len(all_images), BATCH)):
        batch_paths = all_images[i:i+BATCH]
        valid_paths = [img for img in batch_paths if os.path.exists(os.path.join(IMAGE_DIR, img))]
        if not valid_paths: continue
        imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in valid_paths]
        inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        spatial_h = h[:, 1:, :].float()
        
        for b_idx in range(len(valid_paths)):
            acts = ae.encode(spatial_h[b_idx]).max(dim=0).values
            for feat in target_features:
                max_act = acts[feat].item()
                if max_act > 0:
                    heapq.heappush(top_images[feat], (max_act, valid_paths[b_idx]))
                    if len(top_images[feat]) > (GRID_SIZE * GRID_SIZE):
                        heapq.heappop(top_images[feat])

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
except IOError:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

for feat in tqdm(target_features):
    best_images = sorted(top_images[feat], key=lambda x: x[0], reverse=True)
    if len(best_images) < 2: continue
    
    activations_tensor = torch.tensor([x[0] for x in best_images], device=DEVICE)
    img_paths = [os.path.join(IMAGE_DIR, x[1]) for x in best_images]
    ms_score = calculate_ms_score(get_dino_embeddings(img_paths), activations_tensor)
    
    canvas_w, canvas_h = GRID_SIZE * IMG_SIZE, (GRID_SIZE * IMG_SIZE) + HEADER_H
    collage = Image.new('RGB', (canvas_w, canvas_h), color="black")
    draw = ImageDraw.Draw(collage)
    draw.text((10, 15), f"Layer {LAYER} | Feature {feat} | MS Score: {ms_score:.4f}", fill="white", font=font)
    
    for idx, (act_val, filename) in enumerate(best_images):
        try:
            img = Image.open(os.path.join(IMAGE_DIR, filename)).convert("RGB")
            min_dim = min(img.width, img.height)
            left, top = (img.width - min_dim) / 2, (img.height - min_dim) / 2
            img = img.crop((left, top, left + min_dim, top + min_dim)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            
            x_offset, y_offset = (idx % GRID_SIZE) * IMG_SIZE, ((idx // GRID_SIZE) * IMG_SIZE) + HEADER_H
            collage.paste(img, (x_offset, y_offset))
            draw.rectangle([x_offset, y_offset, x_offset + 40, y_offset + 18], fill="black")
            draw.text((x_offset + 2, y_offset + 2), f"{act_val:.1f}", fill="lime", font=small_font)
        except Exception:
            pass
            
    collage.save(os.path.join(OUTPUT_DIR, f"collage_L{LAYER}_F{feat}_MS{ms_score:.2f}.jpg"))