import os, random, sys, shutil, gc
sys.path.append(os.path.abspath("dictionary_learning"))
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE

IMAGE_DIR = "/home/cervenka25/large-data/train2014"
SAVE_DIR  = "/home/cervenka25/large-data/trained_sae"

DEVICE = "cuda:0"

D_MODEL  = 1024
D_SAE    = 65536
K_VALUES = [64]
LAYERS   = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22]
LR       = 3e-4
STEPS    = 50000
BATCH    = 8
WARMUP   = 2000

os.makedirs(SAVE_DIR, exist_ok=True)

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

def get_activation_stream(image_paths, target_layer, batch_size=BATCH):
    while True:
        random.shuffle(image_paths)
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB"))
                except: continue
                
            if not images: continue
            
            inputs = proc(images=images, return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
            with torch.no_grad():
                h = vision_tower(**inputs, output_hidden_states=True).hidden_states[target_layer]
                acts = h[:, 1:, :].reshape(-1, D_MODEL) 
            yield acts.float() 

all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

for layer in LAYERS:
    
    for K in K_VALUES:
        final_path = os.path.join(SAVE_DIR, f"ae_layer{layer}_topk{K}.pt")
        
        if os.path.exists(final_path):
            continue
            

        trainer_cfg = {
            "trainer":        TopKTrainer,
            "dict_class":     AutoEncoderTopK,
            "activation_dim": D_MODEL,
            "dict_size":      D_SAE,
            "k":              K,
            "lr":             LR,
            "device":         DEVICE,
            "steps":          STEPS,
            "warmup_steps":   WARMUP,
            "layer":          layer,
            "lm_name":        "clip-vit-large-patch14-336",
        }

        temp_run_dir = os.path.join(SAVE_DIR, f"temp_run_l{layer}_k{K}")
        os.makedirs(temp_run_dir, exist_ok=True)

        trainSAE(
            data            = get_activation_stream(all_images, target_layer=layer),
            trainer_configs = [trainer_cfg],
            steps           = STEPS,
            save_dir        = temp_run_dir,
            log_steps       = 500,
            save_steps      = list(range(5000, STEPS, 5000)),
        )

        default_saved_file = os.path.join(temp_run_dir, "trainer_0", "ae.pt")

        if os.path.exists(default_saved_file):
            shutil.move(default_saved_file, final_path)
            shutil.rmtree(temp_run_dir, ignore_errors=True)
            print(f"✅ {final_path}")
        else:
            print(f"❌  {default_saved_file}")
            
        torch.cuda.empty_cache()
        gc.collect()
