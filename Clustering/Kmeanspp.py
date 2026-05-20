import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
import time
import torch
from fast_pytorch_kmeans import KMeans as PTKMeans

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\PCA_Reduced_Layers"
# We define the BASE directory, and the script will automatically create the _10, _50 folders inside it
BASE_OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA"

# Define the exact path to your mapping CSV
MAPPING_CSV_PATH = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Micro_Clustered_Layers_50_Sequentially\micro_cluster_mapping.csv"

# The list of K values to loop through
K_MACRO_LIST = [5, 10, 50, 100, 500]  

# Specific layers you want to process (e.g., [0, 5, 15, 31]). 
# Set to None or an empty list [] to process ALL layers found in the INPUT_DIR.
TARGET_LAYERS = [2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29,30,31]  


def run_kmeans_and_pack():
    input_path = Path(INPUT_DIR)
    base_output_path = Path(BASE_OUTPUT_DIR)
    mapping_path = Path(MAPPING_CSV_PATH)

    layer_files = sorted(input_path.glob("layer_*_pca_reduced.npy"))
    if not layer_files:
        print(f"No .npy layer files found in {INPUT_DIR}")
        return

    # --- NEW FILTERING LOGIC ---
    if TARGET_LAYERS:
        target_layers_str = [str(layer) for layer in TARGET_LAYERS]
        filtered_files = []
        for file_path in layer_files:
            layer_idx = file_path.stem.split('_')[1]
            if layer_idx in target_layers_str:
                filtered_files.append(file_path)
        
        layer_files = filtered_files
        
        if not layer_files:
            print(f"None of the target layers {TARGET_LAYERS} were found in {INPUT_DIR}")
            return
    # ---------------------------

    # Check for CUDA to ensure we are actually running on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for PyTorch K-Means: {device}")

    # 1. Load the Micro-Cluster Mapping CSV to map rows to image names
    print(f"Loading Mapping CSV: {mapping_path.name}")
    if not mapping_path.exists():
        print(f"Error: Mapping CSV not found at {mapping_path}")
        return
        
    mapping_df = pd.read_csv(mapping_path)
    
    # Dynamically calculate rows per image and create a 1-to-1 row map
    num_clusters = (mapping_df['End_Row'] - mapping_df['Start_Row'] + 1).values
    row_to_image_map = np.repeat(mapping_df['Image_Name'].values, num_clusters)
    
    print(f"{'='*60}")
    print(f"  Running PyTorch K-Means++ (GPU) & Packing ({len(layer_files)} layers)")
    if TARGET_LAYERS:
        print(f"  Selected Layers: {TARGET_LAYERS}")
    print(f"  Target K Values: {K_MACRO_LIST}")
    print(f"{'='*60}\n")

    start_total = time.time()

    for file_path in layer_files:
        start_layer = time.time()
        layer_idx = file_path.stem.split('_')[1]
        print(f"\nProcessing Layer {layer_idx}...")

        # 2. Load and Normalize the Micro-Clusters on CPU
        micro_matrix = np.load(file_path)
        micro_matrix = normalize(micro_matrix, norm='l2')

        # --- Move the entire layer matrix to GPU once ---
        tensor_matrix = torch.from_numpy(micro_matrix.astype(np.float32)).to(device)

        # 3. Loop through all the K values for this specific layer
        for k_macro in K_MACRO_LIST:
            
            # Create the specific output directory for this K
            output_path = base_output_path / f"Macro_Clustered_Concepts_{k_macro}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"  -> [K={k_macro}] Clustering {micro_matrix.shape[0]} micro-clusters on GPU...")

            # Initialize PyTorch KMeans model for this specific K
            pt_kmeans = PTKMeans(
                n_clusters=k_macro,
                max_iter=300,
                mode='euclidean',
                verbose=0,
                init_method='kmeans++'
            )

            # Run K-Means Clustering on GPU
            labels_tensor = pt_kmeans.fit_predict(tensor_matrix)
            
            # Move results back to CPU for numpy operations and saving
            labels = labels_tensor.cpu().numpy()
            macro_centroids = pt_kmeans.centroids.cpu().numpy().astype(np.float32)

            # Save Raw Backup Files to the K-specific folder
            np.save(output_path / f"layer_{layer_idx}_macro_centroids.npy", macro_centroids)
            np.save(output_path / f"layer_{layer_idx}_macro_labels.npy", labels)

            # Build the Packed .npz Archive
            npz_dict = {}
            for concept_id in range(k_macro):
                mask = (labels == concept_id)
                concept_micro_vectors = micro_matrix[mask]
                concept_images = row_to_image_map[mask]
                
                npz_dict[f"concept_{concept_id}_macro_centroid"] = macro_centroids[concept_id]
                npz_dict[f"concept_{concept_id}_micro_vectors"]  = concept_micro_vectors
                npz_dict[f"concept_{concept_id}_images"]         = np.array(concept_images, dtype=str)

            # Save the master archive
            archive_file = output_path / f"layer_{layer_idx}_packed_concepts.npz"
            np.savez_compressed(archive_file, **npz_dict)

            print(f"     Saved {archive_file.name} to {output_path.name}")

        elapsed = time.time() - start_layer
        print(f"Layer {layer_idx} completed all K values in {elapsed:.1f}s")

        # Free memory before loading the next layer
        del micro_matrix, tensor_matrix, labels, labels_tensor, macro_centroids, npz_dict
        torch.cuda.empty_cache() # Clear GPU memory specifically

    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  All target layers and all K values complete!")
    print(f"  Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_kmeans_and_pack()