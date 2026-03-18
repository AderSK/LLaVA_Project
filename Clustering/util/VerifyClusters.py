import pickle
import numpy as np
from pathlib import Path

def verify_cluster_integrity(data):
    labels = data['labels']
    k = data['n_clusters']
    layer = data['layer']
    
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    print(f"\n=== Verification Report: Layer {layer} ===")
    print(f"Total Vectors: {len(labels)}")
    print(f"Target Clusters (k): {k}")
    print(f"Actual Clusters found: {len(unique)}")
    
    # 1. Check for Collapsed Clusters
    if len(unique) < k:
        print(f"!! ALERT: {k - len(unique)} clusters are empty (Collapsed).")
    
    # 2. Check for Dominant Clusters (Imbalance)
    max_perc = (max(counts) / len(labels)) * 100
    min_perc = (min(counts) / len(labels)) * 100
    print(f"Largest Cluster: {max_perc:.1f}% of data")
    print(f"Smallest Cluster: {min_perc:.1f}% of data")
    
    if max_perc > 70:
        print("!! WARNING: Highly imbalanced. One cluster is capturing almost everything.")

    # 3. Identify Prototype Indices
    # These are the vectors that are mathematically the "definition" of the cluster
    print("\nCluster Representative IDs (all):")
    for i in range(k):
        # Find the first index belonging to this cluster as a proxy for 'prototype'
        idx = np.where(labels == i)[0][0]
        source = data['source_indices'][idx]
        print(f" Cluster {i}: Token {source[1]} from File Index {source[0]}")

def main():
    path = input("Enter path to a clustered layer .pkl: ").strip().strip('"')
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    verify_cluster_integrity(data)

if __name__ == "__main__":
    main()
