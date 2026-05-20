import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import calinski_harabasz_score

# --- CONFIGURATION ---
MACRO_DIRS = [
    r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Macro_Clustered_Concepts_5",
    r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Macro_Clustered_Concepts_10",
    r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Macro_Clustered_Concepts_50",
    r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Macro_Clustered_Concepts_100",
    r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\Macro_Clustered_Concepts_500"
    
]

OUTPUT_CSV = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\clustering_scores_comparison.csv"

CHUNK_SIZE = 1000  # Rows processed at once during GPU distance calculations


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def dunn_index_pytorch(X: torch.Tensor, labels: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> float:
    """
    Dunn Index on GPU (chunked to prevent VRAM overflow).
    Uses upper-triangle chunking to avoid self-distances in intra-cluster diameter.
    """
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    max_intra = torch.tensor(1e-10, device=X.device)
    min_inter = torch.tensor(float("inf"), device=X.device)

    # 1. Max intra-cluster diameter (chunked, upper-triangle only)
    for label in unique_labels:
        pts = X[labels == label]
        n_pts = len(pts)
        if n_pts < 2:
            continue
        for i in range(0, n_pts, chunk_size):
            chunk_i = pts[i : i + chunk_size]
            for j in range(i, n_pts, chunk_size):
                chunk_j = pts[j : j + chunk_size]
                dists = torch.cdist(chunk_i, chunk_j)
                if i == j:
                    # Mask diagonal (self-distances = 0)
                    eye_mask = ~torch.eye(
                        dists.shape[0], dists.shape[1],
                        dtype=torch.bool, device=X.device
                    )
                    if eye_mask.any():
                        max_intra = torch.max(max_intra, dists[eye_mask].max())
                else:
                    max_intra = torch.max(max_intra, dists.max())

    # 2. Min inter-cluster distance (chunked)
    for i in range(len(unique_labels)):
        pts_i = X[labels == unique_labels[i]]
        if len(pts_i) == 0:
            continue
        for j in range(i + 1, len(unique_labels)):
            pts_j = X[labels == unique_labels[j]]
            if len(pts_j) == 0:
                continue
            for k in range(0, len(pts_i), chunk_size):
                chunk_k = pts_i[k : k + chunk_size]
                dists = torch.cdist(chunk_k, pts_j)
                min_inter = torch.min(min_inter, dists.min())

    if torch.isinf(min_inter):
        return 0.0

    return float(min_inter / max_intra)


def silhouette_score_pytorch(X: torch.Tensor, labels: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> float:
    """
    Silhouette Score on GPU — fully vectorized inner loop.

    For each chunk of points:
      - Compute dist_chunk : [chunk_size, n_samples] in one cdist call.
      - For every cluster k, compute the mean distance from each chunk point
        to cluster k using a boolean mask — all K clusters in one Python loop
        of length K (not n_samples × K as in the naive version).
      - Build mean_dists : [K, chunk_size], mask out each point's own cluster,
        and take the column-wise min → b[] for the whole chunk at once.

    Python iterations: (n_samples / chunk_size) × K
    vs. naive:          n_samples × K
    Speedup: ~chunk_size× fewer Python iterations (e.g. ×1000 for chunk_size=1000).
    """
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0

    n_samples = X.shape[0]
    a = torch.zeros(n_samples, device=X.device)
    b = torch.full((n_samples,), float("inf"), device=X.device)

    # Precompute boolean masks for every cluster: shape [K, n_samples]
    cluster_masks = torch.stack([(labels == lbl) for lbl in unique_labels])   # [K, n]
    cluster_sizes = cluster_masks.sum(dim=1).float()                           # [K]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_len = end - start
        chunk_X = X[start:end]
        chunk_labels = labels[start:end]

        # Full distance matrix for this chunk: [chunk_len, n_samples]
        dist_chunk = torch.cdist(chunk_X, X)

        # --- a[]: mean intra-cluster distance (exclude self) ---
        # Map each chunk point to its cluster index in unique_labels
        chunk_cluster_ids = (
            chunk_labels.unsqueeze(1) == unique_labels.unsqueeze(0)
        ).int().argmax(dim=1)                                                  # [chunk_len]

        for local_i in range(chunk_len):
            k_idx = chunk_cluster_ids[local_i].item()
            mask = cluster_masks[k_idx]                                        # [n_samples] bool
            n_same = cluster_sizes[k_idx].item()
            if n_same > 1:
                # self-distance is 0 so summing it in is harmless; divide by n-1
                a[start + local_i] = dist_chunk[local_i][mask].sum() / (n_same - 1)
            # else a stays 0 (singleton cluster)

        # --- b[]: mean distance to nearest other cluster (fully vectorized) ---
        # mean_dists[k, i] = average distance from chunk point i to all points in cluster k
        mean_dists = torch.zeros((n_clusters, chunk_len), device=X.device)
        for k_idx in range(n_clusters):
            mask = cluster_masks[k_idx]                                        # [n_samples]
            if mask.sum() > 0:
                mean_dists[k_idx] = dist_chunk[:, mask].mean(dim=1)           # [chunk_len]
            else:
                mean_dists[k_idx] = float("inf")

        # Exclude each point's own cluster before taking the minimum
        own_cluster = chunk_cluster_ids.unsqueeze(0)                           # [1, chunk_len]
        cluster_idx = torch.arange(n_clusters, device=X.device).unsqueeze(1)  # [K, 1]
        is_own = cluster_idx == own_cluster                                    # [K, chunk_len]
        mean_dists[is_own] = float("inf")

        b[start:end] = mean_dists.min(dim=0).values                           # [chunk_len]

    # Guard: degenerate case where b is still inf
    b = torch.where(torch.isinf(b), torch.zeros_like(b), b)

    denom = torch.max(a, b)
    sil = torch.where(denom > 0, (b - a) / denom, torch.zeros_like(a))
    sil = torch.nan_to_num(sil, nan=0.0)

    return float(sil.mean())


def davies_bouldin_pytorch(X: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Davies-Bouldin Index on GPU — fully vectorized ratio matrix.
    """
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)

    centroids = torch.zeros((n_clusters, X.shape[1]), device=X.device)
    dispersions = torch.zeros(n_clusters, device=X.device)

    for i, label in enumerate(unique_labels):
        cluster_pts = X[labels == label]
        centroid = cluster_pts.mean(dim=0)
        centroids[i] = centroid
        if len(cluster_pts) > 0:
            dispersions[i] = torch.norm(cluster_pts - centroid, dim=1).mean()

    # Pairwise centroid distances: [K, K]
    centroid_dists = torch.cdist(centroids, centroids, p=2)

    # R[i, j] = (s_i + s_j) / d(c_i, c_j)
    s_i = dispersions.unsqueeze(1)                          # [K, 1]
    s_j = dispersions.unsqueeze(0)                          # [1, K]
    numerator = s_i + s_j                                   # [K, K]

    safe_dists = centroid_dists.clone()
    safe_dists.fill_diagonal_(float("inf"))                 # avoid div-by-zero on diagonal

    R = numerator / safe_dists                              # [K, K]
    R.fill_diagonal_(0.0)

    db_index = R.max(dim=1).values.mean()
    return float(db_index)


# ---------------------------------------------------------------------------
# File loading & scoring
# ---------------------------------------------------------------------------

def _parse_layer(stem: str) -> int:
    """
    Extract the layer number from a filename stem robustly using regex.
    Falls back to -1 if no integer is found.
    Example: 'layer_12_packed_concepts' -> 12
    """
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else -1


def score_packed_npz(file_path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    """
    Load one .npz file, move its data to GPU, and compute all four metrics.
    """
    try:
        print(f"\nLoading {file_path.name}...")
        all_vectors: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with np.load(file_path) as data:
            # Count by vector keys — always consistent with what we actually load
            vector_keys = sorted(
                [k for k in data.files if k.endswith("_micro_vectors")],
                key=lambda k: int(re.search(r"(\d+)", k).group(1)),
            )
            num_concepts = len(vector_keys)

            if num_concepts == 0:
                print("  [SKIP] No micro_vectors found in file.")
                return None

            for concept_id, v_key in enumerate(vector_keys):
                vecs = data[v_key]
                if vecs.ndim == 2 and vecs.shape[0] > 0:
                    all_vectors.append(vecs)
                    all_labels.append(np.full(vecs.shape[0], concept_id, dtype=np.int64))

        if not all_vectors:
            print("  [SKIP] All concept arrays were empty.")
            return None

        X_np = np.vstack(all_vectors).astype(np.float32)
        labels_np = np.concatenate(all_labels)
        n_samples = X_np.shape[0]

        print(f"  -> {n_samples:,} vectors | {num_concepts} clusters | scoring on {device.type.upper()}...")

        t0 = time.time()
        X = torch.from_numpy(X_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)

        t_sil = time.time()
        sil = silhouette_score_pytorch(X, labels)
        print(f"     Silhouette:        {sil:.4f}  ({time.time() - t_sil:.1f}s)")

        t_db = time.time()
        db = davies_bouldin_pytorch(X, labels)
        print(f"     Davies-Bouldin:    {db:.4f}  ({time.time() - t_db:.1f}s)")

        t_ch = time.time()
        ch = calinski_harabasz_score(X_np, labels_np)
        print(f"     Calinski-Harabasz: {ch:.2f}  ({time.time() - t_ch:.1f}s)")

        t_dunn = time.time()
        dunn = dunn_index_pytorch(X, labels)
        print(f"     Dunn:              {dunn:.4f}  ({time.time() - t_dunn:.1f}s)")

        print(f"  -> Total: {time.time() - t0:.1f}s")

        # Free GPU memory immediately
        del X, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "Layer": _parse_layer(file_path.stem),
            "Clusters": num_concepts,
            "Samples": n_samples,
            "Silhouette": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Dunn": dunn,
        }

    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("\nNo results to display.")
        return

    sep = "=" * 108
    header = (
        f"{'Layer':<6} | {'K':<4} | {'Samples':<9} | "
        f"{'Silhouette ↑':<14} | {'DB Index ↓':<12} | "
        f"{'CH Index ↑':<16} | {'Dunn ↑':<10}"
    )
    row_fmt = (
        "{Layer:<6} | {Clusters:<4} | {Samples:<9,} | "
        "{Silhouette:<14.4f} | {Davies-Bouldin:<12.4f} | "
        "{Calinski-Harabasz:<16.2f} | {Dunn:<10.4f}"
    )

    print(f"\n{sep}")
    print(header)
    print("-" * 108)
    for r in results:
        print(row_fmt.format(**r))
    print(sep)
    print("\nMetric guide:")
    print("  Silhouette        ↑  [-1, 1]   — good if > 0")
    print("  Davies-Bouldin    ↓  [0, ∞]    — good if close to 0")
    print("  Calinski-Harabasz ↑  [0, ∞)   — higher = denser, better-separated clusters")
    print("  Dunn              ↑  [0, ∞)   — higher = better inter/intra-cluster separation")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results: List[Dict[str, Any]] = []
    wall_start = time.time()

    for dir_str in MACRO_DIRS:
        macro_path = Path(dir_str)
        if not macro_path.exists():
            print(f"\n[WARNING] Directory not found, skipping: {dir_str}")
            continue

        npz_files = sorted(macro_path.glob("*_packed_concepts.npz"))
        if not npz_files:
            print(f"\n[WARNING] No *_packed_concepts.npz files in: {dir_str}")
            continue

        print(f"\n--- Directory: {macro_path.name} ({len(npz_files)} files) ---")
        for f in npz_files:
            result = score_packed_npz(f, device)
            if result:
                all_results.append(result)

    all_results.sort(key=lambda r: (r["Layer"], r["Clusters"]))
    print_results(all_results)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SUCCESS] Saved results to: {OUTPUT_CSV}")

    print(f"\nTotal time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
