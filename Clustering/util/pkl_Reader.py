import os
import pickle
from pathlib import Path
import numpy as np
import torch

def normalize_path(path: str) -> str:
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1].strip()
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path

def get_layer_items(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]

def num_tokens_of_array(arr):
    if arr is None:
        return 0
    if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
        return int(arr.shape[0])
    return 0

def print_kmeans_pickle(data: dict):
    """Handler for printing K-Means output dictionaries"""
    print("\n=== K-MEANS CLUSTERING RESULTS ===")
    
    # Print general metadata
    print(f"Source file: {data.get('source_file', 'Unknown')}")
    print(f"Clusters (K): {data.get('n_clusters', 'Unknown')}")
    print(f"Total Samples (Tokens): {data.get('n_samples', 'Unknown')}")
    print(f"Feature Dimension: {data.get('n_features', 'Unknown')}")
    
    if 'inertia' in data:
        print(f"Inertia (Within-cluster sum of squares): {data['inertia']:.2f}")

    print("\nChoose mode <1:Print 5 Labels/Centers , 2:Print X Labels/Centers , 3:Print All , 4:Quit>")
    
    labels = data.get("labels", [])
    centers = data.get("centers", [])
    
    max_items = None
    while True:
        mode = input("Mode: ").strip()
        if mode == "1":
            max_items = 5
            break
        elif mode == "2":
            try:
                x_str = input("Enter X (number to print): ").strip()
                max_items = int(x_str)
                if max_items <= 0:
                    print("X must be positive.")
                    continue
                break
            except ValueError:
                print("Invalid number for X.")
        elif mode == "3":
            max_items = len(labels)
            break
        elif mode == "4":
            print("Exiting program.")
            return
        else:
            print("Invalid mode. Please choose 1, 2, 3, or 4.")

    np.set_printoptions(threshold=np.inf, legacy='1.25')

    print("\n--- CLUSTER LABELS ---")
    if len(labels) > 0:
        print_len = min(max_items, len(labels))
        print(f"Showing first {print_len} of {len(labels)} labels:")
        print(labels[:print_len])
        if len(labels) > max_items:
            print(f"... ({len(labels) - max_items} more labels not printed)")
    else:
        print("No labels found.")

    print("\n--- CLUSTER CENTERS ---")
    if len(centers) > 0:
        print_len = min(max_items, len(centers))
        print(f"Showing first {print_len} of {len(centers)} centroid vectors:")
        for i in range(print_len):
            print(f"  Center {i}: {centers[i][:10]} ... (truncated for readability)")
        if len(centers) > max_items:
            print(f"... ({len(centers) - max_items} more centers not printed)")
    else:
        print("No centers found.")
        
    print("\n" + "=" * 80)
    print("Done.")

def main():
    path = input("Enter path to .pkl file: ")
    path = normalize_path(path)

    print(f"\nResolved path: {path}")
    if not os.path.exists(path):
        print("Error: file does not exist.")
        return

    print("\nLoading pickle file...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return

    # DETECT FILE TYPE
    if not isinstance(data, dict):
        print(f"Error: Expected dict, got {type(data)}")
        return

    if "labels" in data and "centers" in data:
        # This is a K-Means output file
        print_kmeans_pickle(data)
        return
        
    elif "image_name" in data and "activations" in data:
        # This is the original LLaVA activations file
        image_name = data["image_name"]
        activations = data["activations"]

        print("\n=== LLaVA ACTIVATIONS FILE ===")
        print(f"Image name: {image_name}")
        
        print("\nToken counts per layer (per tensor):")
        layer_indices = sorted(activations.keys(), key=lambda x: int(x))

        for layer_idx in layer_indices:
            items = get_layer_items(activations[layer_idx])
            if not items:
                print(f"  Layer {layer_idx}: no tensors stored")
                continue

            print(f"  Layer {layer_idx}:")
            for t_idx, t in enumerate(items):
                if isinstance(t, torch.Tensor):
                    print(f"    Tensor {t_idx}: num_tokens={t.shape[0]}, hidden_dim={t.shape[1]}, dtype={t.dtype}")
                elif isinstance(t, np.ndarray):
                    print(f"    Tensor {t_idx}: num_tokens={t.shape[0]}, hidden_dim={t.shape[1]}, dtype={t.dtype}")
                else:
                    print(f"    Tensor {t_idx}: not a tensor/array, type={type(t)}")

        print("\nChoose mode <1:Print 5 Tokens , 2:Print X Tokens , 3:Print All Tokens , 4:Quit>")
        
        first_layer_items = get_layer_items(activations[layer_indices[0]])
        t = first_layer_items[0] if first_layer_items else None
        max_tokens_per_tensor = None

        while True:
            mode = input("Mode: ").strip()
            if mode == "1":
                max_tokens_per_tensor = 5
                break
            elif mode == "2":
                if t is None:
                    print("No tensors available.")
                    continue
                while True:
                    x_str = input("Enter X: ").strip()
                    try:
                        max_tokens_per_tensor = int(x_str)
                        if max_tokens_per_tensor <= 0:
                            continue
                        break
                    except ValueError:
                        pass
                break
            elif mode == "3":
                max_tokens_per_tensor = None
                break
            elif mode == "4":
                return
            else:
                print("Invalid mode.")

        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf)

        print("\n=== PRINTING ACTIVATIONS ===")
        for layer_idx in layer_indices:
            items = get_layer_items(activations[layer_idx])
            print("\n" + "-" * 80)
            print(f"Layer {layer_idx}:")

            for t_idx, act in enumerate(items):
                print(f"\n  Tensor {t_idx}:")
                if act is None:
                    continue

                tokens_total = num_tokens_of_array(act)
                print(f"    Shape: {tuple(act.shape)}")

                tokens_slice = act if max_tokens_per_tensor is None else act[:max_tokens_per_tensor]
                print(tokens_slice)

                if max_tokens_per_tensor is not None and tokens_total > max_tokens_per_tensor:
                    print(f"\n    ... ({tokens_total - max_tokens_per_tensor} more token(s))")

        print("\n" + "=" * 80)
        
    else:
        print("Error: Unknown dictionary format.")
        print(f"Keys found: {list(data.keys())}")

if __name__ == "__main__":
    main()
