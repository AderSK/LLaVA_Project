import os, sys, glob, random, heapq, torch, re, gc
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

SAE_DIR     = "/home/cervenka25/LLaVA_Project/SAE/trained_sae"
IMAGE_DIR   = "/home/cervenka25/large-data/train2014"
BASE_OUTPUT = "/home/cervenka25/LLaVA_Project/SAE/sae_collages_FINAL"

DEVICE      = "cuda:0"
D_MODEL     = 1024
D_SAE       = 65536
BATCH       = 32
FEATURES_N  = 20
GRID_SIZE   = 10
IMG_SIZE    = 100
HEADER_H    = 50

os.makedirs(BASE_OUTPUT, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)
search_images = all_images 
warmup_images = all_images[:1000] 

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
except IOError:
    font = ImageFont.load_default()

sae_files = glob.glob(os.path.join(SAE_DIR, "*.pt"))

for sae_path in sae_files:
    filename = os.path.basename(sae_path)
    match = re.search(r'ae_layer(\d+)_topk(\d+)\.pt', filename)
    if not match:
        continue
        
    layer = int(match.group(1))
    k_val = int(match.group(2))
    
    print(f"\nProcessing Layer {layer} | K={k_val}")
    output_dir = os.path.join(BASE_OUTPUT, f"Layer_{layer}_TopK_{k_val}")
    os.makedirs(output_dir, exist_ok=True)
    
    ae = AutoEncoderTopK(D_MODEL, D_SAE, k_val).to(DEVICE)
    ae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    ae.eval()

    alive_features = set()
    with torch.no_grad():
        for i in range(0, len(warmup_images), BATCH):
            batch_paths = warmup_images[i:i+BATCH]
            valid_paths = [img for img in batch_paths if os.path.exists(os.path.join(IMAGE_DIR, img))]
            if not valid_paths: continue
            
            imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in valid_paths]
            inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[layer]
            spatial_h = h[:, 1:, :].float()
            
            for b_idx in range(len(valid_paths)):
                acts = ae.encode(spatial_h[b_idx]).max(dim=0).values
                alive_features.update((acts > 0).nonzero(as_tuple=True)[0].tolist())
                
    if len(alive_features) < FEATURES_N:
        target_features = list(alive_features)
    else:
        target_features = random.sample(list(alive_features), FEATURES_N)
        
    top_images = {feat: [] for feat in target_features}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(search_images), BATCH), desc=f"L{layer}_K{k_val}"):
            batch_paths = search_images[i:i+BATCH]
            valid_paths = [img for img in batch_paths if os.path.exists(os.path.join(IMAGE_DIR, img))]
            if not valid_paths: continue
                
            imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in valid_paths]
            inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[layer]
            spatial_h = h[:, 1:, :].float()
            
            for b_idx in range(len(valid_paths)):
                acts = ae.encode(spatial_h[b_idx]).max(dim=0).values
                for feat in target_features:
                    max_act = acts[feat].item()
                    if max_act > 0:
                        heapq.heappush(top_images[feat], (max_act, valid_paths[b_idx]))
                        if len(top_images[feat]) > (GRID_SIZE * GRID_SIZE):
                            heapq.heappop(top_images[feat])

    for feat in target_features:
        best_images = sorted(top_images[feat], key=lambda x: x[0], reverse=True)
        if len(best_images) < 2: continue
        
        canvas_w, canvas_h = GRID_SIZE * IMG_SIZE, (GRID_SIZE * IMG_SIZE) + HEADER_H
        collage = Image.new('RGB', (canvas_w, canvas_h), color="black")
        draw = ImageDraw.Draw(collage)
        draw.text((10, 15), f"Layer {layer} | K={k_val} | Feature {feat}", fill="white", font=font)
        
        for idx, (_, img_name) in enumerate(best_images):
            try:
                img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
                min_dim = min(img.width, img.height)
                left, top = (img.width - min_dim) / 2, (img.height - min_dim) / 2
                img = img.crop((left, top, left + min_dim, top + min_dim)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                
                x_offset, y_offset = (idx % GRID_SIZE) * IMG_SIZE, ((idx // GRID_SIZE) * IMG_SIZE) + HEADER_H
                collage.paste(img, (x_offset, y_offset))
            except Exception:
                pass
                
        collage.save(os.path.join(output_dir, f"collage_L{layer}_K{k_val}_F{feat}.jpg"))

    del ae
    torch.cuda.empty_cache()
    gc.collect()