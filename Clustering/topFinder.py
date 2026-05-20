#NEEDS TO BE UPDATED TO WORK W HEATMAP CONFIG NEW
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import normalize

# --- CONFIGURATION ---
MACRO_DIR     = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_100\Macro_Clustered_Concepts_50"
MICRO_PKL_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200"

def find_top_images_for_concept(
    layer_num,
    target_concept_id,
    top_x=10,
    macro_dir=MACRO_DIR,
    micro_pkl_dir=MICRO_PKL_DIR,
):
    print(f"--- Top {top_x} images | Layer {layer_num} | Concept {target_concept_id} ---")

    # 1. Load the packed .npz
    npz_path = os.path.join(macro_dir, f"layer_{layer_num}_packed_concepts.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    macro_centroid_key = f"concept_{target_concept_id}_macro_centroid"
    micro_vectors_key  = f"concept_{target_concept_id}_micro_vectors"
    images_key         = f"concept_{target_concept_id}_images"

    for key in [macro_centroid_key, micro_vectors_key, images_key]:
        if key not in data.files:
            raise KeyError(f"Key '{key}' not found in {npz_path}.")

    macro_centroid = data[macro_centroid_key].astype(np.float32)
    micro_vectors  = data[micro_vectors_key].astype(np.float32)
    image_names    = data[images_key]

    # 2. Cosine similarity of each micro-vector against the macro centroid
    macro_norm  = macro_centroid / (np.linalg.norm(macro_centroid) + 1e-8)
    micros_norm = normalize(micro_vectors, norm='l2')
    similarities = np.dot(micros_norm, macro_norm)

    # 3. Per-image: keep only the BEST scoring micro-vector per image
    #    (one image may contribute multiple micro-vectors to the concept)
    image_best = {}  # image_name -> (best_score, micro_vector_index)
    for idx, (img_name, score) in enumerate(zip(image_names, similarities)):
        if isinstance(img_name, bytes):
            img_name = img_name.decode('utf-8')
        img_name = str(img_name)
        if img_name not in image_best or score > image_best[img_name][0]:
            image_best[img_name] = (score, idx)

    # 4. Sort images by best score descending
    ranked = sorted(image_best.items(), key=lambda x: x[1][0], reverse=True)
    ranked = ranked[:top_x]

    print(f"\n{'Rank':<6} {'Score':<10} {'Image':<45} {'PKL exists'}")
    print("-" * 80)

    top_results = []
    for rank, (img_name, (score, _)) in enumerate(ranked, 1):
        img_stem      = img_name.replace(".jpg", "")
        pkl_name      = f"{img_stem}_activations_micro.pkl"
        pkl_path      = os.path.join(micro_pkl_dir, pkl_name)
        pkl_exists    = "✅" if os.path.exists(pkl_path) else "❌ MISSING"

        print(f"{rank:<6} {score:<10.4f} {img_name:<45} {pkl_exists}")

        top_results.append((img_name, pkl_path))

    print("-" * 80)
    print(f"\nlayer   = {layer_num}")
    print(f"concept = {target_concept_id}")
    print(f"\ntop_results = [")
    for img_name, pkl_path in top_results:
        print(f'    ("{img_name}", r"{pkl_path}"),')
    print("]")

    return top_results


# --- Execution ---
if __name__ == "__main__":
    results = find_top_images_for_concept(
        layer_num=10,
        target_concept_id=2,
        top_x=5,
    )
