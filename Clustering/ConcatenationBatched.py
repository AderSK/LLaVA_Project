import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

# --- CONFIGURATION ---
INTERMEDIATE_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_Clustered_Images"
# New output directory to prevent overwriting the sequential ones
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_Clustered_Layers_Batched"

def concatenate_layers_batched():
    input_path = Path(INTERMEDIATE_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_path.glob("*_micro.pkl"))
    if not pkl_files:
        print(f"No files found in {INTERMEDIATE_DIR}")
        return

    print(f"Batching {len(pkl_files)} files into 3D arrays...\n")
    
    layer_data = defaultdict(list)
    mapping_data = []
    start_time = time.time()

    for i, file_path in enumerate(pkl_files, 1):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        micro_clusters = data["micro_clusters"]

        # Map the 3D index directly to the image name
        mapping_data.append({
            "Image_Index": i - 1,
            "Image_Name": data["image_name"]
        })

        # Group activations by layer
        for layer_idx, clusters_array in micro_clusters.items():
            # clusters_array shape is (K_micro, 4096)
            layer_data[layer_idx].append(clusters_array)

        if i % 100 == 0 or i == len(pkl_files):
            print(f"  Processed {i}/{len(pkl_files)} files...")

    # --- SAVE MAPPING CSV ---
    csv_file = output_path / "micro_cluster_mapping.csv"
    pd.DataFrame(mapping_data).to_csv(csv_file, index=False)
    print(f"\nSaved mapping -> {csv_file.name}")

    # --- SAVE LAYER FILES ---
    for layer_idx, arrays in layer_data.items():
        # np.stack creates a new dimension, making it a 3D array: (Num_Images, K_micro, 4096)
        merged_3d = np.stack(arrays)
        save_file = output_path / f"layer_{layer_idx}_micro_clustered.npy"
        np.save(save_file, merged_3d)
        print(f"Saved Layer {layer_idx} -> {save_file.name} | Shape: {merged_3d.shape}")

    print(f"\nDone in {int(time.time() - start_time)}s")

if __name__ == "__main__":
    concatenate_layers_batched()
