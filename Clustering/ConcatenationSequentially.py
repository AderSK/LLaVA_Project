import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

# --- CONFIGURATION ---
INTERMEDIATE_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Micro_Clustered_Images_50"
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Micro_Clustered_Layers_50_Sequentially"

def concatenate_layers():
    input_path = Path(INTERMEDIATE_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_path.glob("*_micro.pkl"))
    if not pkl_files:
        print(f"No files found in {INTERMEDIATE_DIR}")
        return

    print(f"Concatenating {len(pkl_files)} files...\n")
    
    layer_data = defaultdict(list)
    mapping_data = []
    current_row = 0
    start_time = time.time()

    for i, file_path in enumerate(pkl_files, 1):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        micro_clusters = data["micro_clusters"]
        num_clusters = next(iter(micro_clusters.values())).shape[0]

        # Map rows to the image
        mapping_data.append({
            "Image_Name": data["image_name"],
            "Start_Row": current_row,
            "End_Row": current_row + num_clusters - 1
        })
        current_row += num_clusters

        # Group activations by layer
        for layer_idx, clusters_array in micro_clusters.items():
            layer_data[layer_idx].append(clusters_array)

        if i % 100 == 0 or i == len(pkl_files):
            print(f"  Processed {i}/{len(pkl_files)} files...")

    # --- SAVE MAPPING CSV ---
    csv_file = output_path / "micro_cluster_mapping.csv"
    pd.DataFrame(mapping_data).to_csv(csv_file, index=False)
    print(f"\nSaved mapping -> {csv_file.name}")

    # --- SAVE LAYER FILES ---
    for layer_idx, arrays in layer_data.items():
        save_file = output_path / f"layer_{layer_idx}_micro_clustered.npy"
        np.save(save_file, np.vstack(arrays))
        print(f"Saved Layer {layer_idx} -> {save_file.name}")

    print(f"\nDone in {int(time.time() - start_time)}s")

if __name__ == "__main__":
    concatenate_layers()
