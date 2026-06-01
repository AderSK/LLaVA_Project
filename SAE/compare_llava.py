import os, sys, torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append(os.path.abspath("dictionary_learning"))
from dictionary_learning.trainers.top_k import AutoEncoderTopK

IMAGE_PATH = "/home/adam/Projects/data/COCO_test2014_000000000275.jpg"
SAE_PATH   = "/home/adam/Documents/sae_backup_lol/ae_layer18_topk64.pt"

DEVICE        = "cuda:0"
LAYER         = 18
FEATURE_IDX   = 41045
BOOST_AMOUNTS = [0.0, 15.0, 30.0, 45.0, 60.0]

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(DEVICE)

ae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
ae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
ae.eval()

feature_vector = ae.decoder.weight[:, FEATURE_IDX].to(DEVICE).half()

def get_steering_hook(vector, boost):
    def hook(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        if hidden_states.dim() != 3:
            return output
            
        modified_hidden = hidden_states.clone()
        modified_hidden[:, 1:, :] = modified_hidden[:, 1:, :] + (vector * boost)
        
        if hasattr(output, 'last_hidden_state'):
            try:
                from dataclasses import replace
                return replace(output, last_hidden_state=modified_hidden)
            except:
                output.last_hidden_state = modified_hidden
                return output
        elif isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        else:
            return modified_hidden
            
    return hook

def generate_description(image, prompt_text):
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(DEVICE, torch.float16)
    
    with torch.no_grad():
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2, 
            do_sample=True
        )
    return processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

if hasattr(llava_model, 'vision_tower'):
    tower = llava_model.vision_tower
elif hasattr(llava_model, 'model') and hasattr(llava_model.model, 'vision_tower'):
    tower = llava_model.model.vision_tower
else:
    raise ValueError("Could not locate the vision tower!")

if hasattr(tower, 'vision_model'):
    target_layer = tower.vision_model.encoder.layers[LAYER]
else:
    target_layer = tower.encoder.layers[LAYER]

img = Image.open(IMAGE_PATH).convert("RGB")
prompt = "USER: <image>\nDescribe what you see in detail. ASSISTANT:"

for boost in BOOST_AMOUNTS:
    print(f"\nRUN: +{boost} BOOST")
    print("-" * 60)
    
    if boost == 0.0:
        desc = generate_description(img, prompt)
        print(f"{desc}\n")
    else:
        handle = target_layer.register_forward_hook(get_steering_hook(feature_vector, boost))
        try:
            desc = generate_description(img, prompt)
            print(f"{desc}\n")
        finally:
            handle.remove()
