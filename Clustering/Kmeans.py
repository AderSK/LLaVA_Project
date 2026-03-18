"""
K-means Clustering for Dimensionality-Reduced Activations
==========================================================

Clusters PCA-reduced neural network activations using MiniBatchKMeans.
Saves cluster assignments, centroids, and image-level mappings for validation.

References:
-----------
[1] MacQueen, J. (1967).
    Some methods for classification and analysis of multivariate observations.
    Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics 
    and Probability, 1(14), 281-297.

[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
    Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
    Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
    & Duchesnay, E. (2011).
    Scikit-learn: Machine Learning in Python.
    Journal of Machine Learning Research, 12, 2825-2830.

[3] Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020).
    Array programming with NumPy.
    Nature, 585, 357-362.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.cluster import MiniBatchKMeans


def get_path(prompt: str) -> Path | None:
    """Get and validate file path from user input."""
    path_input = input(prompt).strip()
    if not path_input:
        return None
    clean = path_input.strip().strip('"').strip("'")
    p = Path(clean).expanduser().resolve()
    return p if p.exists() else None


def load_array_data(npy_path: Path) -> np.ndarray | None:
    """Load concatenated activation data from NPY file."""
    try:
        data = np.load(npy_path)
        return data
    except Exception as e:
        print(f"  Error loading {npy_path.name}: {e}")
        return None


def run_clustering(X: np.ndarray, n_clusters: int) -> Dict[str, Any] | None:
    """
    Run MiniBatchKMeans clustering on input matrix.
    
    Returns dict with labels, centers, inertia, and shape info.
    Returns None if clustering fails or insufficient samples.
    """
    if X.shape[0] < n_clusters:
        print(f"  Error: {X.shape[0]} samples < {n_clusters} clusters")
        return None
    
    try:
        # MiniBatchKMeans handles large datasets memory-efficiently
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=10000, 
            random_state=42, 
            n_init="auto"
        )
        labels = kmeans.fit_predict(X)
        
        return {
            "labels": labels,
            "centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }
    
    except Exception as e:
        print(f"  Clustering error: {e}")
        return None


def map_labels_to_images(labels: np.ndarray, mapping_csv: Path, n_clusters: int) -> pd.DataFrame | None:
    """
    Map token-level cluster labels back to their original images.
    Calculates percentage distribution and dominant cluster per image.
    """
    try:
        mapping_df = pd.read_csv(mapping_csv)
        image_analysis = []

        for _, row in mapping_df.iterrows():
            img_name = row['Image_Name']
            start = int(row['Start_Row'])
            end = int(row['End_Row']) 
            
            # Extract clusters specific to this image
            img_tokens_clusters = labels[start:end+1]
            unique_clusters, counts = np.unique(img_tokens_clusters, return_counts=True)
            
            total_tokens = len(img_tokens_clusters)
            dominant_cluster = unique_clusters[np.argmax(counts)]
            
            record = {
                "Image_Name": img_name,
                "Total_Tokens": total_tokens,
                "Dominant_Cluster": dominant_cluster
            }
            
            # Calculate distribution percentages
            for c in range(n_clusters):
                if c in unique_clusters:
                    idx = np.where(unique_clusters == c)[0][0]
                    percent = round((counts[idx] / total_tokens) * 100, 2)
                else:
                    percent = 0.0
                record[f"Cluster_{c}_%"] = percent
                
            image_analysis.append(record)
            
        return pd.DataFrame(image_analysis)

    except Exception as e:
        print(f"  Mapping error: {e}")
        return None


def save_results(result: Dict, npy_path: Path, n_clusters: int, out_folder: Path, mapped_df: pd.DataFrame = None) -> Path:
    """
    Save clustering results and mapped CSV data.
    """
    base_name = npy_path.stem
    out_pkl = out_folder / f"{base_name}_clusters{n_clusters}.pkl"
    
    result["source_file"] = str(npy_path.name)
    result["n_clusters"] = n_clusters
    
    try:
        with open(out_pkl, "wb") as f:
            pickle.dump(result, f)
            
        if mapped_df is not None:
            out_csv = out_folder / f"{base_name}_clusters{n_clusters}_mapped.csv"
            mapped_df.to_csv(out_csv, index=False)
            print(f"    Mapped CSV saved: {out_csv.name}")
            
        return out_pkl
    except Exception as e:
        print(f"    Error saving: {e}")
        return None


def main() -> None:
    """Main pipeline: input -> load -> cluster -> map -> save."""
    
    # ---------------------------------------------------------
    # AUTOMATED MODE (Triggered by the Runner script)
    # ---------------------------------------------------------
    if len(sys.argv) > 1:
        npy_input = Path(sys.argv[1])
        csv_input = Path(sys.argv[2]) if sys.argv[2].lower() != "none" else None
        out_input = Path(sys.argv[3])
        n_clusters = int(sys.argv[4])
        
        X = load_array_data(npy_input)
        if X is None:
            return
            
        result = run_clustering(X, n_clusters)
        if result is None:
            return
            
        print(f"    Inertia: {result['inertia']:.2f}")
        
        mapped_df = None
        if csv_input:
            mapped_df = map_labels_to_images(result["labels"], csv_input, n_clusters)
            
        out_path = save_results(result, npy_input, n_clusters, out_input, mapped_df)
        if out_path:
            print(f"    Saved: {out_path.name}")
        return

    # ---------------------------------------------------------
    # INTERACTIVE MODE (Triggered if you run this script directly)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("K-means Clustering with Token Mapping")
    print("=" * 70 + "\n")
    
    # Get NPY data file
    print("Step 1: Select Concatenated Data")
    while True:
        npy_input = get_path("Enter concatenated .npy file path: ")
        if not npy_input or npy_input.suffix != '.npy':
            print("Error: Please provide a valid .npy file")
            continue
        break
        
    # Get CSV Mapping file
    print("\nStep 2: Select Row Mapping File")
    while True:
        csv_input = get_path("Enter tracking .csv file path (or leave blank to skip mapping): ")
        if csv_input is None:
            print("Mapping skipped. Will only output raw labels.")
            break
        if not csv_input.exists() or csv_input.suffix != '.csv':
            print("Error: Please provide a valid .csv file")
            continue
        break
    
    # Get output folder
    print("\nStep 3: Select output folder")
    while True:
        out_input = get_path("Enter output folder: ")
        if not out_input or not out_input.is_dir():
            print("Error: Invalid folder")
            continue
        break
    
    # Get clustering parameters
    print("\nStep 4: Clustering parameters")
    while True:
        try:
            k_input = input("Number of clusters (2-100): ").strip()
            n_clusters = int(k_input)
            if 2 <= n_clusters <= 100:
                break
            else:
                print("Error: Enter 2-100")
        except ValueError:
            print("Error: Enter a number")
    
    # Process file
    print("\n" + "-" * 70)
    print(f"Processing {npy_input.name}...\n")
    
    X = load_array_data(npy_input)
    if X is None:
        return
        
    print(f"  Shape: {X.shape}")
    
    result = run_clustering(X, n_clusters)
    if result is None:
        return
        
    print(f"  Inertia: {result['inertia']:.2f}")
    
    # Map back to images if CSV was provided
    mapped_df = None
    if csv_input:
        print("  Mapping clusters to original images...")
        mapped_df = map_labels_to_images(result["labels"], csv_input, n_clusters)
    
    out_path = save_results(result, npy_input, n_clusters, out_input, mapped_df)
    
    if out_path:
        print(f"  Base results saved: {out_path.name}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
