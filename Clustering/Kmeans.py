"""
K-means Clustering for Dimensionality-Reduced Activations
==========================================================

Clusters PCA-reduced neural network activations using K-means.
Saves cluster assignments and centroids for validation.

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

Author: Samuel Púček
Date: January 2026
"""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from sklearn.cluster import KMeans


def get_path(prompt: str) -> Path | None:
    """Get and validate file path from user input."""
    path_input = input(prompt).strip()
    if not path_input:
        return None
    clean = path_input.strip().strip('"').strip("'")
    p = Path(clean).expanduser().resolve()
    return p if p.exists() else None


def find_files(folder: Path, pattern: str) -> List[Path]:
    """Find and sort files matching glob pattern in folder."""
    return sorted(list(folder.glob(pattern)))


def get_available_layers(pca_path: Path) -> List[Any]:
    """Extract list of available layer indices from PCA file."""
    try:
        with open(pca_path, "rb") as f:
            data = pickle.load(f)
        
        if "activations" not in data:
            return []
        
        return list(data["activations"].keys())
    
    except Exception:
        return []


def load_activation_data(pca_path: Path, layer_idx: Any) -> np.ndarray | None:
    """
    Load activation data from PCA file for specific layer.
    
    Extracts layer data from activations dict and handles wrapped formats.
    Returns None if layer not found or data invalid.
    """
    try:
        with open(pca_path, "rb") as f:
            data = pickle.load(f)
        
        if "activations" not in data:
            print(f"  Error: No activations in {pca_path.name}")
            return None
        
        layer_data = data["activations"].get(layer_idx)
        
        if layer_data is None:
            print(f"  Error: Layer {layer_idx} not in {pca_path.name}")
            return None
        
        # Handle list/tuple wrapped data
        if isinstance(layer_data, (list, tuple)):
            layer_data = layer_data[0] if len(layer_data) > 0 else None
        
        if layer_data is None:
            return None
        
        return np.asarray(layer_data, dtype=np.float32)
    
    except Exception as e:
        print(f"  Error loading {pca_path.name}: {e}")
        return None


def run_clustering(X: np.ndarray, n_clusters: int) -> Dict[str, Any] | None:
    """
    Run K-means clustering on input matrix.
    
    Returns dict with labels, centers, inertia, and shape info.
    Returns None if clustering fails or insufficient samples.
    """
    if X.shape[0] < n_clusters:
        print(f"  Error: {X.shape[0]} samples < {n_clusters} clusters")
        return None
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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


def cluster_file(pca_path: Path, n_clusters: int, layer_idx: Any) -> Dict[str, Any] | None:
    """
    Load activations and run clustering on single file.
    
    Handles loading, validation, and K-means execution for one PCA file.
    """
    
    print(f"  {pca_path.name} (Layer {layer_idx})")
    
    # Load activation data
    X = load_activation_data(pca_path, layer_idx)
    if X is None:
        return None
    
    print(f"    Shape: {X.shape}")
    
    # Run K-means
    result = run_clustering(X, n_clusters)
    if result is None:
        return None
    
    print(f"    Inertia: {result['inertia']:.2f}")
    
    return result


def save_results(result: Dict, pca_path: Path, layer_idx: Any, n_clusters: int, out_folder: Path) -> Path:
    """
    Save clustering results with metadata to new PKL file.
    
    Includes source reference and layer info for validation script integration.
    """
    
    out_name = pca_path.stem + f"_layer{layer_idx}_clusters{n_clusters}.pkl"
    out_path = out_folder / out_name
    
    # Add metadata for validation script
    result["source_file"] = str(pca_path.name)
    result["layer"] = layer_idx
    result["n_clusters"] = n_clusters
    result["file_names"] = [pca_path.stem]
    
    try:
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        return out_path
    except Exception as e:
        print(f"    Error saving: {e}")
        return None


def main() -> None:
    """Main pipeline: input -> load -> cluster -> save."""
    
    print("\n" + "=" * 70)
    print("K-means Clustering")
    print("=" * 70 + "\n")
    
    # Get PCA files
    print("Step 1: Select PCA file(s)")
    while True:
        pca_input = get_path("Enter PCA file or folder: ")
        
        if not pca_input:
            print("Error: Invalid path")
            continue
        
        if pca_input.is_dir():
            pca_files = find_files(pca_input, "*.pkl")
            if not pca_files:
                print("Error: No .pkl files found")
                continue
            print(f"Found {len(pca_files)} PCA file(s)")
        else:
            pca_files = [pca_input]
            print(f"Using: {pca_input.name}")
        
        break
    
    # Get output folder
    print("\nStep 2: Select output folder")
    while True:
        out_input = get_path("Enter output folder: ")
        
        if not out_input or not out_input.is_dir():
            print("Error: Invalid folder")
            continue
        
        break
    
    # Get clustering parameters
    print("\nStep 3: Clustering parameters")
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
    
    # Get layer index
    layer_input = input("Layer index (0-based, or leave blank for all layers): ").strip()
    
    if layer_input:
        # Specific layer
        layer_indices = [int(layer_input)]
        process_all_layers = False
    else:
        # All layers - will be determined per file
        layer_indices = None
        process_all_layers = True
        print("Will process all available layers.")
    
    # Process files
    print("\n" + "-" * 70)
    print("Processing...\n")
    
    success_count = 0
    total_processed = 0
    
    for pca_file in pca_files:
        # Determine layers to process
        if process_all_layers:
            layers_to_process = get_available_layers(pca_file)
            if not layers_to_process:
                print(f"  {pca_file.name}: No layers found, skipping.")
                continue
            print(f"  {pca_file.name}: Found {len(layers_to_process)} layer(s)")
        else:
            layers_to_process = layer_indices
        
        # Process each layer
        for layer_idx in layers_to_process:
            total_processed += 1
            result = cluster_file(pca_file, n_clusters, layer_idx)
            
            if result is None:
                continue
            
            out_path = save_results(result, pca_file, layer_idx, n_clusters, out_input)
            if out_path:
                print(f"    Saved: {out_path.name}")
                success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"COMPLETE - Processed {success_count}/{total_processed} layer(s)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
