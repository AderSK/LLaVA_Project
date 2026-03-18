"""
Cluster Validation Metrics
==========================

Computes validation metrics for K-means clustering results directly from 
concatenated NPY files and MiniBatchKMeans output:
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


"""

from __future__ import annotations

import pickle
import numpy as np
import sys
import json
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


def load_npy_data(npy_folder: Path, source_file_name: str) -> np.ndarray:
    """
    Load the concatenated activation data from the NPY file.
    Matches the source_file logged during clustering.
    """
    fpath = npy_folder / source_file_name
    
    if not fpath.exists():
        # Fallback to look for similar names if exact match fails
        matches = list(npy_folder.glob(f"*{Path(source_file_name).stem}*.npy"))
        if not matches:
            raise FileNotFoundError(f"Could not find NPY file matching '{source_file_name}' in {npy_folder}")
        fpath = matches[0]
        
    try:
        data = np.load(fpath)
        return data
    except Exception as e:
        raise ValueError(f"Error loading {fpath.name}: {e}")


def dunn_index(X: np.ndarray, labels: np.ndarray, sample_size: int = 5000) -> float:
    """
    Calculate Dunn index for cluster separation quality.
    Safely down-samples the input to prevent RAM overflow on large datasets.
    """
    if len(X) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]
        
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0
    
    centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    if len(centers) > 1:
        inter_dists = pdist(centers)
        min_between = np.min(inter_dists) if len(inter_dists) > 0 else 1e-10
    else:
        min_between = 1e-10
    
    max_within = 1e-10
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_dists = pdist(cluster_points)
            if len(intra_dists) > 0:
                max_within = max(max_within, np.max(intra_dists))
    
    return min_between / (max_within + 1e-10)


def compute_metrics(cluster_path: Path, npy_folder: Path) -> Dict[str, Any] | None:
    """
    Compute all validation metrics for a single clustering result.
    """
    try:
        with open(cluster_path, "rb") as f:
            cluster_data = pickle.load(f)
        
        if "labels" not in cluster_data or "source_file" not in cluster_data:
            print(f"  Skipping {cluster_path.name}: Not a valid clustering results file.")
            return None
            
        source_file = cluster_data["source_file"]
        labels = cluster_data["labels"]
        n_clusters = cluster_data.get("n_clusters", len(np.unique(labels)))
        inertia = cluster_data.get("inertia", None)
        
        # Load raw data from NPY
        X = load_npy_data(npy_folder, source_file)
        n_samples = X.shape[0]
        
        if n_samples != len(labels):
            raise ValueError(f"Shape mismatch: NPY has {n_samples} rows, but labels array has {len(labels)} rows.")
            
        # Safe Down-sampling for heavy metrics
        max_samples_for_distance = 10000 
        
        sil = silhouette_score(X, labels, sample_size=min(n_samples, max_samples_for_distance), random_state=42)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        dunn = dunn_index(X, labels, sample_size=max_samples_for_distance)
        
        return {
            "Clusters": n_clusters,
            "Source": source_file,
            "Silhouette": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Dunn": dunn,
            "Inertia": inertia if inertia is not None else np.nan,
            "Samples": n_samples,
        }
    
    except Exception as e:
        print(f"  Error processing {cluster_path.name}: {e}")
        return None


def print_results(results: List[Dict[str, Any]]) -> None:
    """Print results table with interpretation guide."""
    if not results:
        print("No valid results computed.")
        return
    
    try:
        results = sorted(results, key=lambda x: int(x["Clusters"]))
    except Exception:
        pass
    
    print("\n" + "=" * 125)
    print(f"{'Source NPY':<25} | {'K':<4} | {'Silhouette ↑':<14} | {'DB Index ↓':<12} | "
          f"{'CH Index ↑':<15} | {'Dunn ↑':<10} | {'Inertia':<12} | {'Samples':<10}")
    print("-" * 125)
    
    for r in results:
        inertia_str = f"{r['Inertia']:.2e}" if not np.isnan(r.get("Inertia", np.nan)) else "N/A"
        src_name = r['Source'][:22] + "..." if len(r['Source']) > 25 else r['Source']
        
        print(
            f"{src_name:<25} | {r['Clusters']:<4} | {r['Silhouette']:<14.4f} | {r['Davies-Bouldin']:<12.4f} | "
            f"{r['Calinski-Harabasz']:<15.2f} | {r['Dunn']:<10.4f} | {inertia_str:<12} | {r['Samples']:<10}"
        )
    
    print("=" * 125)
    print("\nMetric Interpretation:")
    print("  ↑ = Higher is better | ↓ = Lower is better")
    print("  Silhouette ↑: [-1, 1] - Good if > 0.5")
    print("  Davies-Bouldin ↓: [0, ∞] - Good if < 1.0")
    print("  Calinski-Harabasz ↑: [0, ∞) - Good if > 1000")
    print("  Dunn ↑: [0, ∞) - Good if > 0.5")


def main() -> None:
    # ---------------------------------------------------------
    # AUTOMATED MODE (Triggered by the run_pipeline.py Orchestrator)
    # ---------------------------------------------------------
    if len(sys.argv) > 1:
        cluster_path = Path(sys.argv[1])
        npy_folder = Path(sys.argv[2])
        
        result = compute_metrics(cluster_path, npy_folder)
        if result:
            # Print as JSON so the orchestrator can parse the metrics easily
            print(json.dumps(result))
        return

    # ---------------------------------------------------------
    # INTERACTIVE MODE (If run directly by user)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("Clustering Validation Pipeline")
    print("=" * 70 + "\n")
    
    print("Step 1: Clustering Results (The .pkl files generated by K-Means)")
    while True:
        cluster_path = get_path("Enter path to the clustered .pkl file (or folder of .pkls): ")
        if not cluster_path:
            continue
            
        if cluster_path.is_dir():
            cluster_files = sorted(list(cluster_path.glob("*.pkl")))
            if not cluster_files:
                print("Error: No .pkl files found")
                continue
        else:
            cluster_files = [cluster_path]
            
        print(f"Found {len(cluster_files)} valid clustering file(s).")
        break
    
    print("\nStep 2: Original Data (The folder containing the .npy files)")
    while True:
        npy_folder = get_path("Enter folder containing the concatenated .npy files: ")
        if not npy_folder or not npy_folder.is_dir():
            print("Error: Invalid folder")
            continue
        break
    
    print("\n" + "-" * 70)
    print("Computing metrics...\n")
    
    all_results = []
    for f in cluster_files:
        result = compute_metrics(f, npy_folder)
        if result:
            all_results.append(result)
            
    print_results(all_results)

if __name__ == "__main__":
    main()
