import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import time

# --- CONFIGURATION ---
# Input directory is the output directory of your previous script
INPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Micro_Clustered_Layers_50_Sequentially"
# New directory to store PCA results
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\PCA_Reduced_Layers"

# --- TOGGLES ---
SavePcaStat = True      # If True, saves a .txt file with the explained variance percentages
SavePcaReduced = True   # If True, saves the PCA-transformed data as .npy files

# Number of components to keep (can be an integer like 100, or a float like 0.95 to keep 95% variance)
N_COMPONENTS = 0.8

def apply_pca_to_layers():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Create output directory if we are saving anything
    if SavePcaStat or SavePcaReduced:
        output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_path.glob("layer_*_micro_clustered.npy"))
    if not npy_files:
        print(f"No .npy files found in {INPUT_DIR}")
        return

    print(f"Found {len(npy_files)} layer files to process with PCA.\n")
    start_time = time.time()

    for i, file_path in enumerate(npy_files, 1):
        print(f"Processing {file_path.name}...")
        
        # Load the concatenated micro-clusters for the layer
        data = np.load(file_path)
        
        # Safety check: Ensure n_components isn't larger than samples or features
        n_comps = min(N_COMPONENTS, data.shape[0], data.shape[1])
        
        # Initialize and fit PCA
        pca = PCA(n_components=n_comps)
        reduced_data = pca.fit_transform(data)
        
        # Extract the layer prefix for naming the new files (e.g., "layer_0")
        layer_name = file_path.stem.replace("_micro_clustered", "")
        
        # --- SAVE PCA STATISTICS ---
        if SavePcaStat:
            variance_ratios = pca.explained_variance_ratio_
            
            # Format to: PC1 X.XX%, PC2 Y.YY%, ...
            stats_formatted = [f"PC{idx+1} {var*100:.4f}%" for idx, var in enumerate(variance_ratios)]
            stats_string = ", ".join(stats_formatted)
            
            # Save to a text file
            stat_file = output_path / f"{layer_name}_pca_stats.txt"
            with open(stat_file, "w") as f:
                f.write(f"Total Explained Variance by {n_comps} components: {sum(variance_ratios)*100:.2f}%\n\n")
                f.write(stats_string)
            print(f"  -> Saved Stats to {stat_file.name}")

        # --- SAVE REDUCED DATA ---
        if SavePcaReduced:
            save_file = output_path / f"{layer_name}_pca_reduced.npy"
            np.save(save_file, reduced_data)
            print(f"  -> Saved Reduced Data to {save_file.name}")

    print(f"\nDone in {int(time.time() - start_time)}s")

if __name__ == "__main__":
    apply_pca_to_layers()
