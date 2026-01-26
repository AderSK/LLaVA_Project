"""
PCA Dimensionality Reduction for Neural Network Activations
===========================================================

Reduces activation tensor dimensions using PCA before clustering.
Handles various data formats and provides layer-wise statistics.

References:
-----------
[1] Pearson, K. (1901).
    On lines and planes of closest fit to systems of points in space.
    The London, Edinburgh, and Dublin Philosophical Magazine and Journal 
    of Science, 2(11), 559-572.

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
from typing import Dict, Any

from sklearn.decomposition import PCA


def normalize_path(path_str: str) -> Path:
    """Clean up file path input by removing quotes and expanding home directory."""
    if not path_str:
        return None
    clean_str = path_str.strip().strip('"').strip("'")
    return Path(clean_str).expanduser().resolve()


def validate_dimensions(target_dim: int) -> bool:
    """Check if dimension input is reasonable (between 1 and 10000)."""
    if target_dim < 1:
        print("Error: Need at least 1 dimension.")
        return False
    if target_dim > 10000:
        print("Warning: Very large dimension. Continue? (y/n): ", end="")
        return input().strip().lower() == "y"
    return True


def reshape_tensor(tensor: Any) -> np.ndarray | None:
    """
    Convert various tensor formats to 2D array (n_samples, n_features).
    
    Handles direct 2D arrays, 3D tensors reshaped to 2D, and list/tuple wrappers.
    Returns None if conversion fails.
    """
    if tensor is None:
        return None
    
    try:
        # Handle list/tuple wrapping
        if isinstance(tensor, (list, tuple)):
            if len(tensor) == 0:
                return None
            X = np.vstack([np.asarray(t, dtype=np.float32) for t in tensor])
        else:
            X = np.asarray(tensor, dtype=np.float32)
        
        # Reshape 3D to 2D if needed (batch, seq, hidden) -> (batch*seq, hidden)
        if X.ndim == 3:
            b, s, h = X.shape
            print(f"    Reshaping {X.shape} -> ({b*s}, {h})")
            X = X.reshape(-1, h)
        
        # Handle 1D cases
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Sanity check for final shape
        if X.ndim != 2:
            print(f"    Error: Ended up with shape {X.shape}, expected 2D")
            return None
        
        return X
    
    except Exception as e:
        print(f"    Error reshaping: {e}")
        return None


def apply_pca(X: np.ndarray, target_dim: int, layer_key: Any) -> tuple:
    """
    Run PCA with automatic component selection.
    
    Fit PCA and return reduced data plus explained variance ratio.
    Automatically caps components at min(samples, features) to avoid errors.
    """
    n_samples, n_features = X.shape
    
    # Can't have more components than features or samples
    actual_dim = min(target_dim, n_features, n_samples)
    
    if actual_dim < target_dim:
        print(f"    Note: Requested {target_dim} but data only allows {actual_dim}")
    
    try:
        pca = PCA(n_components=actual_dim, svd_solver="auto")
        X_reduced = pca.fit_transform(X).astype(np.float32)
        
        explained = np.sum(pca.explained_variance_ratio_)
        
        return X_reduced, explained
    
    except Exception as e:
        print(f"    PCA failed: {e}")
        return None, None


def load_and_reduce(src_path: Path, target_dim: int) -> Dict[Any, np.ndarray | None] | None:
    """
    Load pickle file with activations dict and apply PCA layer-by-layer.
    
    Returns dict mapping layer indices to reduced activation arrays.
    Skips layers that fail during processing.
    """
    
    print(f"Loading: {src_path.name}")
    try:
        with open(src_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {src_path}")
        return None
    except Exception as e:
        print(f"Error loading: {e}")
        return None
    
    # Check for required structure
    if "activations" not in data:
        print("Error: No 'activations' key in file")
        return None
    
    activations = data["activations"]
    if not isinstance(activations, dict):
        print("Error: 'activations' should be a dict")
        return None
    
    reduced = {}
    
    print(f"\nProcessing {len(activations)} layer(s)...\n")
    
    for layer_key, tensor in activations.items():
        print(f"  Layer {layer_key}:")
        
        if tensor is None:
            print(f"    Skipped (None)")
            reduced[layer_key] = None
            continue
        
        # Reshape to 2D
        X = reshape_tensor(tensor)
        if X is None:
            print(f"    Skipped (reshape failed)")
            reduced[layer_key] = None
            continue
        
        n_samples, n_features = X.shape
        print(f"    Input: ({n_samples}, {n_features})")
        
        # Apply PCA
        X_red, exp_var = apply_pca(X, target_dim, layer_key)
        
        if X_red is None:
            print(f"    Skipped (PCA failed)")
            reduced[layer_key] = None
            continue
        
        n_comp = X_red.shape[1]
        print(f"    Output: ({n_samples}, {n_comp})")
        print(f"    Explained variance: {exp_var*100:.2f}%")
        
        reduced[layer_key] = X_red
    
    return reduced


def save_results(src_path: Path, reduced: Dict, target_dim: int, original: Dict) -> Path:
    """
    Save reduced activations to new PKL file with metadata.
    
    Preserves original dict structure and adds PCA metadata for reproducibility.
    """
    
    out_name = src_path.stem + f"_PCA{target_dim}.pkl"
    out_path = src_path.parent / out_name
    
    # Copy original, replace activations
    save_data = original.copy()
    save_data["activations"] = reduced
    
    # Add metadata for future reference
    if "pca_metadata" not in save_data:
        save_data["pca_metadata"] = {}
    
    save_data["pca_metadata"]["target_components"] = target_dim
    save_data["pca_metadata"]["original_file"] = str(src_path.name)
    
    try:
        with open(out_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"\nSaved: {out_path}")
        return out_path
    except Exception as e:
        print(f"Error saving: {e}")
        raise


def main() -> None:
    """Main pipeline: input -> load -> reduce -> save."""
    
    print("\n" + "=" * 70)
    print("PCA Dimensionality Reduction")
    print("=" * 70 + "\n")
    
    # Get source file
    while True:
        src_input = input("Enter path to source .pkl file: ").strip()
        if not src_input:
            print("Error: Path required")
            continue
        
        src_path = normalize_path(src_input)
        if not src_path or not src_path.exists():
            print(f"Error: File not found")
            continue
        
        break
    
    # Get target dimensions
    while True:
        try:
            dim_input = input("Enter target PCA dimensions (1-10000): ").strip()
            target_dim = int(dim_input)
            if validate_dimensions(target_dim):
                break
        except ValueError:
            print("Error: Enter a number")
    
    # Load and process
    try:
        with open(src_path, "rb") as f:
            original_data = pickle.load(f)
        
        reduced_data = load_and_reduce(src_path, target_dim)
        
        if reduced_data is None:
            print("Error: Processing failed")
            return
        
        # Save results
        out_path = save_results(src_path, reduced_data, target_dim, original_data)
        
        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Source: {src_path.name}")
        print(f"Output: {out_path.name}")
        print(f"Dimensions: {target_dim}")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
