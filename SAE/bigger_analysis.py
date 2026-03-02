import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
import logging


LAYER = 11
BASE_PATH = "/home/cervenka25/large-data"
IMAGE_DIR = os.path.join(BASE_PATH, "test2014")
SAE_FILE  = os.path.join(BASE_PATH, "CLIP-ViT-L-14-SAE-L11/11_resid/1000104192.pt")
OUTPUT_DIR = "/home/cervenka25/LLaVA_Project/SAE/multi_seed"
DEVICE = "cuda"

N_SEEDS        = 5
TOP_K_FEATURES = 5
N_MAX_EXAMPLES = 5
N_BASELINE     = 200
N_SEARCH       = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

log.info("Loading model...")
bnb_config   = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor    = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", quantization_config=bnb_config, device_map={"": 0}
)
vision_tower = model.model.vision_tower
prompt       = "Describe the image"

log.info("Loading SAE weights...")
checkpoint = torch.load(SAE_FILE, map_location=DEVICE)
pre_b      = checkpoint['model_state_dict']['pre_b'].to(DEVICE).half()
W_enc      = checkpoint['model_state_dict']['enc'].to(DEVICE).half()
n_features = W_enc.shape[1]

def get_activations(img: Image.Image):
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h       = vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[LAYER]
        latents = torch.relu((h - pre_b) @ W_enc)
    acts_per_patch = latents[0, 1:, :] 
    acts_summed    = acts_per_patch.sum(dim=0)
    return acts_per_patch, acts_summed


def get_heatmap(acts_per_patch: torch.Tensor, feature_id: int):
    return acts_per_patch[:, feature_id].cpu().float().numpy().reshape(24, 24)


log.info(f"Computing baseline over {N_BASELINE} images...")
all_images     = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
baseline_names = random.sample(all_images, N_BASELINE)

baseline_sum   = torch.zeros(n_features, device=DEVICE, dtype=torch.float32)
baseline_sq    = torch.zeros(n_features, device=DEVICE, dtype=torch.float32)
baseline_count = 0

for img_name in tqdm(baseline_names, desc="Baseline"):
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        _, acts = get_activations(img)
        acts_f = acts.float()
        baseline_sum += acts_f
        baseline_sq  += acts_f ** 2
        baseline_count += 1
    except Exception as e:
        log.warning(f"Baseline failed for {img_name}: {e}")

baseline_mean = baseline_sum / baseline_count
baseline_var  = baseline_sq / baseline_count - baseline_mean ** 2
baseline_std  = baseline_var.sqrt()

dead_mask = baseline_std < 1e-6
n_dead    = dead_mask.sum().item()

baseline_std_safe = baseline_std.clone()
baseline_std_safe[dead_mask] = 1.0


seed_names    = random.sample(all_images, N_SEEDS)
seed_name_set = set(seed_names)
seeds         = []
features_to_search = {}

for seed_name in seed_names:
    img = Image.open(os.path.join(IMAGE_DIR, seed_name)).convert("RGB")
    acts_per_patch, acts_summed = get_activations(img)
    acts_f = acts_summed.float()

    z_scores = (acts_f - baseline_mean) / baseline_std_safe

    z_scores[dead_mask] = float('-inf')

    z_scores[acts_f == 0] = float('-inf')

    top_vals, top_ids = torch.topk(z_scores, TOP_K_FEATURES)

    heatmaps = {}
    for fid in top_ids.cpu().numpy():
        fid_int = int(fid)
        heatmaps[fid_int] = get_heatmap(acts_per_patch, fid_int)
        if fid_int not in features_to_search:
            features_to_search[fid_int] = []

    seeds.append({
        'img':      img,
        'name':     seed_name,
        'features': top_ids.cpu().numpy(),
        'z_scores': top_vals.cpu().numpy(),
        'raw_acts': acts_f[top_ids].cpu().numpy(),
        'heatmaps': heatmaps
    })
    log.info(f"  {seed_name}: features={top_ids.cpu().numpy()}, z={top_vals.cpu().numpy().round(1)}")

search_pool = random.sample(all_images, min(N_SEARCH, len(all_images))) if N_SEARCH else all_images
log.info(f"Searching {len(search_pool)} images...")
errors = 0

for img_name in tqdm(search_pool, desc="Searching"):
    if img_name in seed_name_set:
        continue
    try:
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        acts_per_patch, acts_summed = get_activations(img)
        acts_f = acts_summed.float()

        for fid in features_to_search.keys():
            raw_act = acts_f[fid].item()
            if raw_act == 0:
                continue

            z_score = ((acts_f[fid] - baseline_mean[fid]) / baseline_std_safe[fid]).item()

            features_to_search[fid].append({
                'raw_act': raw_act,
                'z_score': z_score,
                'path':    os.path.join(IMAGE_DIR, img_name),
                'name':    img_name,
                'heatmap': get_heatmap(acts_per_patch, fid)
            })

    except Exception as e:
        errors += 1
        log.warning(f"Search failed for {img_name}: {e}")

if errors:
    log.warning(f"{errors} images failed during search.")

for fid in features_to_search:
    features_to_search[fid] = sorted(
        features_to_search[fid], key=lambda x: x['z_score'], reverse=True
    )[:N_MAX_EXAMPLES]
    log.info(f"Feature {fid}: {len(features_to_search[fid])} examples found, "
             f"top z={[round(e['z_score'],1) for e in features_to_search[fid]]}")

for i, seed in enumerate(seeds):
    n_rows = TOP_K_FEATURES + 1
    n_cols = N_MAX_EXAMPLES * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 3))

    for col in range(n_cols):
        axes[0, col].axis('off')

    axes[0, 0].imshow(seed['img'])
    axes[0, 0].set_title(f"SEED {i+1}\n{seed['name']}", fontsize=8, fontweight='bold')
    axes[0, 0].axis('on')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    for j, fid in enumerate(seed['features']):
        ax = axes[0, j + 1]
        im = ax.imshow(seed['heatmaps'][fid], cmap='viridis')
        ax.set_title(f"Feature {fid}\nz={seed['z_scores'][j]:.1f} | raw={seed['raw_acts'][j]:.0f}",
                     fontsize=7, fontweight='bold')
        ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j, fid in enumerate(seed['features']):
        examples = features_to_search.get(fid, [])
        row      = j + 1

        for k in range(N_MAX_EXAMPLES):
            col_img  = k * 2
            col_heat = k * 2 + 1

            if k < len(examples):
                ex = examples[k]
                axes[row, col_img].imshow(Image.open(ex['path']))
                axes[row, col_img].set_title(
                    f"z={ex['z_score']:.1f} | raw={ex['raw_act']:.0f}\n{ex['name'][:22]}",
                    fontsize=6
                )
                axes[row, col_heat].imshow(ex['heatmap'], cmap='viridis')
            else:
                axes[row, col_img].text(0.5, 0.5, "no examples\nfound",
                                        ha='center', va='center', fontsize=8, color='gray')
                axes[row, col_heat].text(0.5, 0.5, "", ha='center', va='center')

            axes[row, col_img].axis('off')
            axes[row, col_heat].axis('off')

        axes[row, 0].set_ylabel(f"Feat {fid}", fontsize=8, fontweight='bold',
                                rotation=0, labelpad=50, va='center')

    plt.suptitle(
        f"Seed {i+1}: {seed['name']}\nTop {TOP_K_FEATURES} Features (z-score) & Max Activating Images",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"seed{i+1}_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {out_path}")

log.info("All done.")