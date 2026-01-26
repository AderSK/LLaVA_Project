"""
Cluster Validation Metrics
==========================

Computes validation metrics for K-means clustering results:
- Silhouette coefficient
- Davies-Bouldin index
- Calinski-Harabasz index
- Dunn index
- Inertia

References:
-----------
[1] Rousseeuw, P.J. (1987).
    Silhouettes: a graphical aid to the interpretation and validation of 
    cluster analysis. Journal of Computational and Applied Mathematics, 
    20, 53-65.

[2] Davies, D.L., & Bouldin, D.W. (1979).
    A cluster separation measure. IEEE Transactions on Pattern Analysis 
    and Machine Intelligence, 2, 224-227.

[3] Caliński, T., & Harabasz, J. (1974).
    A dendrite method for cluster analysis. Communications in Statistics, 
    3(1), 1-27.

[4] Dunn, J.C. (1974).
    Well-separated clusters and optimal fuzzy partitions. Journal of 
    Cybernetics, 4(1), 95-104.

[5] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
    Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
    Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
    & Duchesnay, E. (2011).
    Scikit-learn: Machine Learning in Python.
    Journal of Machine Learning Research, 12, 2825-2830.

[6] Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020).
    Array programming with NumPy.
    Nature, 585, 357-362.

[7] Virtanen, P., Gommers, R., Oliphant, T.E., et al. (2020).
    SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.
    Nature Methods, 17, 261-272.

Author: Samuel Púček
Date: January 2026
"""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.spatial.distance import pdist


def get_path(prompt: str) -> Path | None:
    """Get and validate file path from user input."""
    path_input = input(prompt).strip()
    if not path_input:
        return None
    clean = path_input.strip().strip('"').strip("'")
    p = Path(clean).expanduser().resolve()
    return p if p.exists() else None


def find_pca_file(pca_folder: Path, image_name: str) -> Path:
    """Find PCA file matching image name stem with flexible glob matching."""
    stem = Path(image_name).stem
    matches = sorted(list(pca_folder.glob(f"{stem}*.pkl")))
    
    if not matches:
        raise FileNotFoundError(f"No PCA file found for '{stem}'")
    
    return matches[0]


def load_pca_data(pca_folder: Path, image_names: List[str], layer_idx: Any) -> np.ndarray:
    """
    Load and consolidate activation data from multiple PCA files.
    
    Reconstructs design matrix by loading specified layer from each file
    and vertically stacking. Handles list/tuple wrapped data.
    """
    X_list = []
    
    for image_name in image_names:
        fpath = find_pca_file(pca_folder, image_name)
        
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        
        layer_data = data["activations"].get(layer_idx)
        
        if layer_data is None:
            raise ValueError(f"Layer {layer_idx} not in {fpath.name}")
        
        # Handle wrapped data formats
        if isinstance(layer_data, (list, tuple)):
            layer_data = layer_data[0] if len(layer_data) > 0 else None
        
        if layer_data is not None:
            X_list.append(np.asarray(layer_data, dtype=np.float32))
    
    if not X_list:
        raise ValueError(f"No valid data for layer {layer_idx}")
    
    return np.vstack(X_list)


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Dunn index for cluster separation quality.
    
    Ratio of minimum inter-cluster distance to maximum intra-cluster distance.
    Higher values indicate better separated, compact clusters.
    """
    unique_labels = np.unique(labels)
    
    if len(unique_labels) == 1:
        return 0.0
    
    # Compute cluster centers
    centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    # Minimum distance between cluster centers
    if len(centers) > 1:
        inter_dists = pdist(centers)
        min_between = np.min(inter_dists) if len(inter_dists) > 0 else 1e-10
    else:
        min_between = 1e-10
    
    # Maximum distance within clusters
    max_within = 1e-10
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_dists = pdist(cluster_points)
            if len(intra_dists) > 0:
                max_within = max(max_within, np.max(intra_dists))
    
    return min_between / (max_within + 1e-10)


def compute_metrics(cluster_path: Path, pca_folder: Path) -> Dict[str, Any] | None:
    """
    Compute all validation metrics for single clustering result.
    
    Loads clustering file, reconstructs activations from PCA files,
    and calculates five complementary metrics.
    """
    
    try:
        # Load clustering result
        with open(cluster_path, "rb") as f:
            cluster_data = pickle.load(f)
        
        layer_idx = cluster_data["layer"]
        labels = cluster_data["labels"]
        image_names = cluster_data.get("file_names", [cluster_path.stem])
        inertia = cluster_data.get("inertia", None)
        
        print(f"  {cluster_path.name} (Layer {layer_idx})")
        
        # Load activation data from PCA files
        X = load_pca_data(pca_folder, image_names, layer_idx)
        n_samples = X.shape[0]
        
        # Compute metrics
        sample_size = min(n_samples, 5000)
        sil = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        dunn = dunn_index(X, labels)
        
        return {
            "Layer": layer_idx,
            "Silhouette": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Dunn": dunn,
            "Inertia": inertia if inertia is not None else np.nan,
            "Samples": n_samples,
        }
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def print_results(results: List[Dict[str, Any]]) -> None:
    """
    Print results table with interpretation guide.
    
    Formats metrics in aligned columns and provides thresholds
    for interpreting each metric value.
    """
    
    if not results:
        print("No results.")
        return
    
    # Sort by layer index if possible
    try:
        results = sorted(results, key=lambda x: int(str(x["Layer"])))
    except (ValueError, TypeError):
        pass
    
    print("\n" + "=" * 110)
    print(f"{'Layer':<10} | {'Silhouette ↑':<14} | {'DB Index ↓':<12} | "
          f"{'CH Index ↑':<15} | {'Dunn ↑':<12} | {'Inertia':<12} | {'Samples':<10}")
    print("-" * 110)
    
    for r in results:
        inertia_str = f"{r['Inertia']:.2e}" if not np.isnan(r.get("Inertia", np.nan)) else "N/A"
        print(
            f"{str(r['Layer']):<10} | {r['Silhouette']:<14.4f} | {r['Davies-Bouldin']:<12.4f} | "
            f"{r['Calinski-Harabasz']:<15.2f} | {r['Dunn']:<12.4f} | {inertia_str:<12} | {r['Samples']:<10}"
        )
    
    print("=" * 110)
    
    # Interpretation guide
    print("\nMetric Interpretation:")
    print("  ↑ = Higher is better | ↓ = Lower is better")
    print()
    print("  Silhouette ↑: [-1, 1] - Good if > 0.5")
    print("  Davies-Bouldin ↓: [0, ∞] - Good if < 1.0")
    print("  Calinski-Harabasz ↑: [0, ∞) - Good if > 1000")
    print("  Dunn ↑: [0, ∞) - Good if > 0.5")
    print("  Inertia ↓: Sum of squared distances")


def main() -> None:
    """Main pipeline: input -> load -> compute -> display."""
    
    print("\n" + "=" * 110)
    print("Clustering Validation")
    print("=" * 110 + "\n")
    
    # Get clustering files
    print("Step 1: Clustering results")
    while True:
        cluster_path = get_path("Enter clustering file or folder: ")
        
        if not cluster_path:
            print("Error: Invalid path")
            continue
        
        if cluster_path.is_dir():
            cluster_files = sorted(list(cluster_path.glob("*.pkl")))
            if not cluster_files:
                print("Error: No .pkl files found")
                continue
            print(f"Found {len(cluster_files)} file(s)")
        else:
            cluster_files = [cluster_path]
            print(f"Using: {cluster_path.name}")
        
        break
    
    # Get PCA folder
    print("\nStep 2: PCA data folder")
    while True:
        pca_folder = get_path("Enter PCA folder: ")
        
        if not pca_folder or not pca_folder.is_dir():
            print("Error: Invalid folder")
            continue
        
        pca_files = list(pca_folder.glob("*.pkl"))
        print(f"Found {len(pca_files)} PCA file(s)")
        break
    
    # Process files
    print("\n" + "-" * 110)
    print("Computing metrics...\n")
    
    all_results = []
    for cluster_file in cluster_files:
        result = compute_metrics(cluster_file, pca_folder)
        if result:
            all_results.append(result)
    
    # Display results
    print_results(all_results)


if __name__ == "__main__":
    main()
