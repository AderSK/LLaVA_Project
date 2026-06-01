import os, sys, torch, random, gc, json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.append(os.path.expanduser("~/LLaVA_Project/SAE"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_DIR  = "/home/cervenka25/large-data/test2014"
SAE_DIR    = "/home/cervenka25/LLaVA_Project/SAE/trained_sae"
OUTPUT_DIR = "/home/cervenka25/LLaVA_Project/SAE/layer_galleries"
JSON_PATH  = os.path.join(OUTPUT_DIR, "gallery_data.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = "cuda:0"
LAYERS      = [20, 22]
BATCH_SIZE  = 16 
SAMPLE_SIZE = 1280

IMG_SIZE    = 128
COLS        = 10   
ROWS        = 10   
HEADER_H    = 60   

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE)
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.seed(99)
sampled_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))

gallery_json_data = {
    "dataset_info": {
        "total_images_sampled": len(sampled_images),
        "seed": 42
    },
    "layers": {}
}

for L in LAYERS:
    
    feature_vault = {i: [] for i in range(65536)}
    
    sae_path = os.path.join(SAE_DIR, f"ae_layer{L}_topk64.pt")
    sae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
    sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    sae.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sampled_images), BATCH_SIZE)):
            batch_paths = sampled_images[i:i+BATCH_SIZE]
            raw_images = []
            valid_paths = []
            
            for p in batch_paths:
                try:
                    img = Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB")
                    raw_images.append(img)
                    valid_paths.append(p)
                except: continue
                    
            if not raw_images: continue
            
            inputs = proc(images=raw_images, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values']
            
            h = vision_tower(**inputs, output_hidden_states=True).hidden_states[L]
            acts = sae.encode(h[:, 1:, :].float()) 
            
            max_acts = acts.max(dim=1).values 
            
            for b_idx, path in enumerate(valid_paths):
                fired_features = torch.where(max_acts[b_idx] > 0)[0].cpu().numpy()
                for f_idx in fired_features:
                    val = max_acts[b_idx, f_idx].item()
                    feature_vault[f_idx].append((val, path))
                    
                    feature_vault[f_idx].sort(key=lambda x: x[0], reverse=True)
                    if len(feature_vault[f_idx]) > ROWS:
                        feature_vault[f_idx] = feature_vault[f_idx][:ROWS]

    
    feature_scores = [(f, feature_vault[f][0][0]) for f in range(65536) if len(feature_vault[f]) > 0]
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_scores[:COLS]
    
    layer_key = f"layer_{L}"
    gallery_json_data["layers"][layer_key] = {}
    
    canvas_w = COLS * IMG_SIZE
    canvas_h = HEADER_H + (ROWS * IMG_SIZE)
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except: font = ImageFont.load_default()

    for col_idx, (f_idx, max_val) in enumerate(top_features):
        x_offset = col_idx * IMG_SIZE
        
        draw.text((x_offset + 5, 10), f"Feat: {f_idx}\nMax: {max_val:.1f}", fill=(255, 255, 255), font=font)
        
        images_data = feature_vault[f_idx]
        
        gallery_json_data["layers"][layer_key][f"feature_{f_idx}"] = [
            {"image_name": img_path, "activation": round(act_val, 3)} 
            for act_val, img_path in images_data
        ]
        
        for row_idx, (act_val, img_path) in enumerate(images_data):
            y_offset = HEADER_H + (row_idx * IMG_SIZE)
            try:
                img = Image.open(os.path.join(IMAGE_DIR, img_path)).convert("RGB")
                aspect = img.width / img.height
                if aspect > 1:
                    new_w = int(img.height)
                    offset = (img.width - new_w) // 2
                    img = img.crop((offset, 0, offset + new_w, img.height))
                else:
                    new_h = int(img.width)
                    offset = (img.height - new_h) // 2
                    img = img.crop((0, offset, img.width, offset + new_h))
                    
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                canvas.paste(img, (x_offset, y_offset))
                
                draw.rectangle([x_offset, y_offset + IMG_SIZE - 20, x_offset + 50, y_offset + IMG_SIZE], fill=(0,0,0,150))
                draw.text((x_offset + 5, y_offset + IMG_SIZE - 18), f"{act_val:.1f}", fill=(0, 255, 0), font=font)
            except: pass

    save_path = os.path.join(OUTPUT_DIR, f"layer_{L:02d}_gallery.jpg")
    canvas.save(save_path, quality=90)
    
    del sae, feature_vault
    torch.cuda.empty_cache()
    gc.collect()

with open(JSON_PATH, "w") as f:
    json.dump(gallery_json_data, f, indent=4)
