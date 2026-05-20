import numpy as np
import cv2
import pickle
import pandas as pd
import os
from sklearn.preprocessing import normalize

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MACRO_DIR    = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_100\Macro_Clustered_Concepts_50"
PCA_DIR      = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\PCA_Reduced_Layers"
MAPPING_CSV  = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Layers_200_Sequentially\micro_cluster_mapping.csv"
DATASET_DIR  = r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset"
OUTPUT_DIR   = r"C:\Users\samko\Desktop\Bakalarka\Heatmaps"
GRID_SIZE    = 24  # 24x24 = 576 base image tokens

# The PKL stores data under integer layer keys 0,1,2,3,4
# We need to know which PKL key corresponds to model layer 10
# Based on your output the PKL has keys [0,1,2,3,4] — adjust this map as needed
# Run the inspector block below first to confirm
LAYER_TO_PKL_KEY = {
    10: 0,   # ← CONFIRM THIS — first of the 5 stored layers
    # add more mappings here once confirmed
}

# ─── CACHES ───────────────────────────────────────────────────────────────────
_mapping_df = None
_pca_cache  = {}  # layer_num -> normalized pca matrix
_npz_cache  = {}  # layer_num -> npz data

def get_mapping_df():
    global _mapping_df
    if _mapping_df is None:
        _mapping_df = pd.read_csv(MAPPING_CSV)
    return _mapping_df

def get_pca_matrix(layer_num):
    if layer_num not in _pca_cache:
        path = os.path.join(PCA_DIR, f"layer_{layer_num}_pca_reduced.npy")
        mat  = np.load(path).astype(np.float32)
        _pca_cache[layer_num] = normalize(mat, norm='l2')
        print(f"  [pca] layer_{layer_num}: shape={mat.shape}")
    return _pca_cache[layer_num]

def get_npz(layer_num):
    if layer_num not in _npz_cache:
        path = os.path.join(MACRO_DIR, f"layer_{layer_num}_packed_concepts.npz")
        _npz_cache[layer_num] = np.load(path, allow_pickle=True)
    return _npz_cache[layer_num]


def generate_concept_heatmap(layer_num, target_concept_id, image_name, pkl_path):
    print(f"\n[L{layer_num}|C{target_concept_id}] {image_name}")

    # ── STEP 1: Get concept micro-vectors for this image from .npz ─────────────
    data = get_npz(layer_num)
    micro_vectors_key = f"concept_{target_concept_id}_micro_vectors"
    images_key        = f"concept_{target_concept_id}_images"

    image_names_arr = np.array([
        s.decode('utf-8') if isinstance(s, bytes) else str(s)
        for s in data[images_key]
    ])
    valid_indices = np.where(image_names_arr == image_name)[0]

    if len(valid_indices) == 0:
        print(f"  ⚠️  Image not found in concept {target_concept_id}.")
        return False

    # These are 74-dim PCA-space, L2-normalized
    concept_vecs = data[micro_vectors_key][valid_indices].astype(np.float32)
    concept_vecs = normalize(concept_vecs, norm='l2')
    print(f"  {len(valid_indices)} concept centroid(s) | dim={concept_vecs.shape[1]}")

    # ── STEP 2: Get this image's rows from the PCA matrix via mapping CSV ──────
    mapping_df = get_mapping_df()
    row_match  = mapping_df[mapping_df['Image_Name'] == image_name]

    if row_match.empty:
        print(f"  ⚠️  Image not found in mapping CSV.")
        return False

    start_row = int(row_match.iloc[0]['Start_Row'])
    end_row   = int(row_match.iloc[0]['End_Row'])
    pca_norm  = get_pca_matrix(layer_num)

    # Rows belonging to this image in the PCA matrix = its 100 centroids (74-dim each)
    image_pca_rows = pca_norm[start_row : end_row + 1]  # shape (100, 74)
    print(f"  Image PCA rows: {start_row}–{end_row} ({image_pca_rows.shape[0]} centroids)")

    # ── STEP 3: Load PKL ────────────────────────────────────────────────────────
    with open(pkl_path, 'rb') as f:
        micro_data = pickle.load(f)

    # Resolve PKL layer key (int 0–4 based on stored layers)
    pkl_key = LAYER_TO_PKL_KEY.get(layer_num)
    if pkl_key is None or pkl_key not in micro_data['token_labels']:
        print(f"  ⚠️  PKL key for layer {layer_num} not found.")
        print(f"       Available PKL keys: {list(micro_data['token_labels'].keys())}")
        print(f"       Update LAYER_TO_PKL_KEY dict.")
        return False

    token_labels       = np.array(micro_data['token_labels'][pkl_key])        # (1902,)
    kept_token_indices = np.array(micro_data['kept_token_indices'][pkl_key])  # (1902,)
    print(f"  token_labels range: [{token_labels.min()}, {token_labels.max()}] | kept={len(kept_token_indices)}")

    # ── STEP 4: For each concept centroid, find matching row in image's PCA rows
    # Both image_pca_rows and concept_vecs are 74-dim, L2-normalized → cosine sim
    all_token_indices = []

    for pos, cvec in enumerate(concept_vecs):
        # Cosine sim between this concept centroid and the image's 100 PCA centroids
        sims              = np.dot(image_pca_rows, cvec)   # shape (100,)
        local_cluster_idx = int(np.argmax(sims))
        score             = float(sims[local_cluster_idx])
        print(f"  Centroid {pos}: best cluster={local_cluster_idx} score={score:.4f}")

        # Token indices for this cluster
        mask           = (token_labels == local_cluster_idx)
        orig_positions = kept_token_indices[mask]
        base_tokens    = orig_positions[orig_positions < GRID_SIZE * GRID_SIZE]

        if len(base_tokens) == 0:
            print(f"  ⚠️  No base-image tokens for cluster {local_cluster_idx} "
                  f"(total tokens with label: {mask.sum()})")
            continue

        print(f"  ✓ {len(base_tokens)} base tokens from cluster {local_cluster_idx}")
        all_token_indices.extend(base_tokens.tolist())

    if not all_token_indices:
        print(f"  No tokens found — skipping heatmap.")
        return False

    # ── STEP 5: Build heatmap ──────────────────────────────────────────────────
    token_arr = np.array(all_token_indices, dtype=int)
    mask_flat = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    np.add.at(mask_flat, token_arr, 1.0)
    if mask_flat.max() > 0:
        mask_flat /= mask_flat.max()
    heatmap_2d = mask_flat.reshape(GRID_SIZE, GRID_SIZE)

    # ── STEP 6: Overlay and save ───────────────────────────────────────────────
    img_path = os.path.join(DATASET_DIR, image_name)
    img      = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    heatmap_up = cv2.resize(heatmap_2d, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    heatmap_8  = np.uint8(255 * heatmap_up)
    colormap   = cv2.applyColorMap(heatmap_8, cv2.COLORMAP_JET)
    blended    = cv2.addWeighted(colormap, 0.5, img, 0.5, 0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem     = image_name.replace(".jpg", "")
    out_path = os.path.join(OUTPUT_DIR, f"heatmap_L{layer_num}_C{target_concept_id}_{stem}.jpg")
    cv2.imwrite(out_path, blended)
    print(f"  ✅ Saved ({len(token_arr)} tokens): {out_path}")
    return True


# ─── LAYER KEY INSPECTOR (run this first to confirm LAYER_TO_PKL_KEY) ─────────
def inspect_pkl_layers(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    keys = sorted(d['micro_clusters'].keys())
    print(f"PKL layer keys: {keys}")
    print("These correspond to actual model layers in order.")
    print("Check your activation extraction script to see which layers were saved.")

# ─── EXECUTION ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    LAYER   = 10
    CONCEPT = 2
    # Uncomment to confirm PKL keys first:
    # inspect_pkl_layers(r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000163333_activations_micro.pkl")

    top_results = [
    ("COCO_val2014_000000349101.jpg", r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000349101_activations_micro.pkl"),
    ("COCO_val2014_000000435444.jpg", r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000435444_activations_micro.pkl"),
    ("COCO_val2014_000000133680.jpg", r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000133680_activations_micro.pkl"),
    ("COCO_val2014_000000259342.jpg", r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000259342_activations_micro.pkl"),
    ("COCO_val2014_000000263664.jpg", r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_200\Micro_Clustered_Images_200\COCO_val2014_000000263664_activations_micro.pkl"),
    ]

    ok, fail = 0, []
    for image_name, pkl_path in top_results:
        try:
            if generate_concept_heatmap(LAYER, CONCEPT, image_name, pkl_path):
                ok += 1
            else:
                fail.append(image_name)
        except Exception as e:
            import traceback
            print(f"  ❌ {image_name}: {e}")
            traceback.print_exc()
            fail.append(image_name)

    print(f"\nDone: {ok}/{len(top_results)} saved.")
