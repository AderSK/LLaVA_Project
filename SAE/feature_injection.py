import torch, os, random
import torch.nn as nn
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from dictionary_learning.trainers.top_k import AutoEncoderTopK

if not hasattr(nn.Module, "set_submodule"):
    def set_submodule(self, target, module):
        if "." not in target:
            setattr(self, target, module)
        else:
            prefix, _, suffix = target.rpartition(".")
            setattr(self.get_submodule(prefix), suffix, module)
    nn.Module.set_submodule = set_submodule

SAE_PATH = "/home/cervenka25/large-data/trained_sae/trainer_0/ae.pt"
IMAGE_DIR = "/home/cervenka25/large-data/test2014"
DEVICE = "cuda:0"
INJECT_MULTIPLIER = 50.0

proc = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    device_map="auto" 
)
sae = AutoEncoderTopK.from_pretrained(SAE_PATH).to(DEVICE).half()

img_name = random.choice([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
prompt = "USER: <image>\nDescribe every single detail in this picture. ASSISTANT:"
inputs = proc(images=img, text=prompt, return_tensors="pt").to(DEVICE)

TARGET_FID = random.randint(0, sae.dict_size - 1)

def injection_hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    f = sae.encode(h)
    
    f[:, :, TARGET_FID] += INJECT_MULTIPLIER 
    
    return sae.decode(f)

with torch.no_grad():
    out_norm = model.generate(**inputs, max_new_tokens=60)
    print(f"\n[NORMAL]: {proc.decode(out_norm[0], skip_special_tokens=True).split('ASSISTANT:')[-1]}")

handle = model.model.vision_tower.vision_model.encoder.layers[11].register_forward_hook(injection_hook)
with torch.no_grad():
    out_steered = model.generate(**inputs, max_new_tokens=60)
    print(f"\n[INJECTED Feat {TARGET_FID}]: {proc.decode(out_steered[0], skip_special_tokens=True).split('ASSISTANT:')[-1]}")

handle.remove()