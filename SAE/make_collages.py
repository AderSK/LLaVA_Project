import os, sys, random, heapq, torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_PATH   = "/home/adam/Documents/sae_backup_lol/ae_layer18_topk64.pt" 
IMAGE_DIR  = "/home/adam/Projects/data/train2014"
OUTPUT_DIR = "/home/adam/Projects/sae_collages"

DEVICE    = "cuda:0"
LAYER     = 18
D_MODEL   = 1024
D_SAE     = 65536
K         = 64
BATCH     = 32

NUM_FEATURES = 10
GRID_SIZE    = 10
IMG_SIZE     = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

ae = AutoEncoderTopK(D_MODEL, D_SAE, K).to(DEVICE)
ae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
ae.eval()

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)
if not all_images:
    sys.exit(1)

alive_features = set()
with torch.no_grad():
    for i in range(0, min(1000, len(all_images)), BATCH):
        batch_paths = all_images[i:i+BATCH]
        images = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in batch_paths]
        inputs = proc(images=images, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        acts = ae.encode(h[:, 1:, :].reshape(-1, D_MODEL).float())
        
        fired = acts.sum(dim=0).nonzero(as_tuple=True)[0].tolist()
        alive_features.update(fired)
        
        if len(alive_features) >= 100: 
            break

target_features = random.sample(list(alive_features), NUM_FEATURES)

top_images = {feat: [] for feat in target_features}

search_images = all_images[:15000] 

with torch.no_grad():
    for i in tqdm(range(0, len(search_images), BATCH), desc="Scanning Images"):
        batch_paths = search_images[i:i+BATCH]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                images.append(Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB"))
                valid_paths.append(p)
            except: continue
                
        if not images: continue
        
        inputs = proc(images=images, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        h = vision_tower(**inputs, output_hidden_states=True).hidden_states[LAYER]
        h_spatial = h[:, 1:, :].float() 
        
        for b_idx in range(len(valid_paths)):
            acts = ae.encode(h_spatial[b_idx]) 
            
            for feat in target_features:
                max_act = acts[:, feat].max().item()
                
                if max_act > 0:
                    heap = top_images[feat]
                    if len(heap) < (GRID_SIZE * GRID_SIZE):
                        heapq.heappush(heap, (max_act, valid_paths[b_idx]))
                    else:
                        heapq.heappushpop(heap, (max_act, valid_paths[b_idx]))

for feat in target_features:
    best_images = sorted(top_images[feat], key=lambda x: x[0], reverse=True)
    
    if len(best_images) == 0:
        continue
        
    canvas_size = GRID_SIZE * IMG_SIZE
    collage = Image.new('RGB', (canvas_size, canvas_size))
    
    for idx, (act_val, img_path) in enumerate(best_images):
        if idx >= (GRID_SIZE * GRID_SIZE): break
            
        try:
            img = Image.open(os.path.join(IMAGE_DIR, img_path)).convert("RGB")
            min_dim = min(img.width, img.height)
            left = (img.width - min_dim) / 2
            top = (img.height - min_dim) / 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            
            row = idx // GRID_SIZE
            col = idx % GRID_SIZE
            collage.paste(img, (col * IMG_SIZE, row * IMG_SIZE))
        except: continue
            
    save_path = os.path.join(OUTPUT_DIR, f"feature_{feat}_collage.jpg")
    collage.save(save_path, quality=95)

