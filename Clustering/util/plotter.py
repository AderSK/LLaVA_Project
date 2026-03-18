"""
Interactive 3D/2D Visualization of Neural Network Activations with K-means Clusters
====================================================================================

Visualizes dimensionality-reduced activations with K-means cluster assignments.
Supports both 2D and 3D scatter plots with interactive layer navigation via slider.
For testing and visualization purposes only.
"""

import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def normalize_path(path_str):
    """Clean up file path input by removing quotes and expanding home directory."""
    if not path_str:
        return None
    return Path(path_str.strip().strip('"').strip("'")).expanduser().resolve()


def get_layer_data(activations, layer_idx):
    """Extract layer data from activations dict, handling wrapped formats."""
    val = activations.get(layer_idx)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val


def get_cluster_labels_for_image(cluster_folder, layer_idx, target_image_name):
    """
    Extract cluster labels for specific image from clustering result file.
    
    Matches image by filename stem and filters labels by source file index.
    Returns None if labels cannot be found or extracted.
    """
    matches = sorted(list(cluster_folder.glob(f"*layer{layer_idx}_*.pkl")))
    if not matches:
        return None

    try:
        with open(matches[0], "rb") as f:
            c_data = pickle.load(f)

        file_names = c_data.get("file_names", [])
        target_stem = Path(target_image_name).stem
        image_file_idx = -1
        
        for i, name in enumerate(file_names):
            if Path(name).stem == target_stem:
                image_file_idx = i
                break
        
        if image_file_idx == -1:
            return None

        labels_all = c_data.get("labels")
        source_indices = c_data.get("source_indices")
        
        if labels_all is None or source_indices is None:
            return None
        
        image_labels = np.array([
            labels_all[i] for i, src in enumerate(source_indices) 
            if src[0] == image_file_idx
        ])
        
        return image_labels if len(image_labels) > 0 else None

    except Exception as e:
        print(f"Error extracting labels for layer {layer_idx}: {e}")
        return None


def calculate_global_bounds(activations, layer_indices, dim=3):
    """Calculate consistent axis limits across all layers."""
    print("Calculating global axis limits...")
    all_data = []
    
    for idx in layer_indices:
        d = get_layer_data(activations, idx)
        if d is not None and d.size > 0:
            if isinstance(d, np.ndarray) and d.ndim >= 2:
                all_data.append(d[:, :dim])
            else:
                continue
    
    if not all_data:
        return None, None
    
    full_stack = np.vstack(all_data)
    g_min = full_stack.min(axis=0)
    g_max = full_stack.max(axis=0)
    margin = (g_max - g_min) * 0.1
    
    return g_min - margin, g_max + margin


def main():
    """Main visualization pipeline."""
    
    pca_path = normalize_path(input("Enter path to PCA .pkl file: "))
    cluster_dir = normalize_path(input("Enter folder containing clustered .pkl files: "))
    
    if not pca_path or not pca_path.exists():
        print("PCA file not found.")
        return
    
    if not cluster_dir or not cluster_dir.exists():
        print("Cluster folder not found.")
        return

    try:
        with open(pca_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading PCA file: {e}")
        return

    activations = data.get("activations")
    if activations is None:
        print("Error: No 'activations' in PCA file")
        return

    image_name = data.get("image_name", "Unknown Image")
    
    try:
        layer_indices = sorted(activations.keys(), key=lambda x: int(x))
    except (ValueError, TypeError):
        layer_indices = sorted(activations.keys())

    if not layer_indices:
        print("No layers found in PCA file")
        return

    plot_dim = input("Plot in 2D or 3D? [2/3]: ").strip()
    if plot_dim not in ["2", "3"]:
        plot_dim = "3"
    plot_dim = int(plot_dim)

    print(f"Plotting in {plot_dim}D with {len(layer_indices)} layers")
    
    g_min, g_max = calculate_global_bounds(activations, layer_indices, dim=plot_dim)
    
    if g_min is None or g_max is None:
        print("Error: Could not calculate plot bounds")
        return

    fig = plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')

    if plot_dim == 2:
        ax = fig.add_subplot(111)
        ax.set_xlim(g_min[0], g_max[0])
        ax.set_ylim(g_min[1], g_max[1])
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        scatter = ax.scatter([], [], s=20, alpha=0.7, edgecolors='none')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(g_min[0], g_max[0])
        ax.set_ylim(g_min[1], g_max[1])
        ax.set_zlim(g_min[2], g_max[2])
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.set_zlabel("Dimension 2")
        scatter = ax.scatter([], [], [], s=20, alpha=0.7, edgecolors='none')

    def update_plot(layer_idx):
        """Update scatter plot for given layer."""
        layer_data = get_layer_data(activations, layer_idx)
        
        if layer_data is None:
            ax.set_title(f"Layer {layer_idx} | No Data Found")
            return

        if not isinstance(layer_data, np.ndarray):
            layer_data = np.asarray(layer_data)

        if layer_data.ndim < 2 or layer_data.shape[1] < plot_dim:
            ax.set_title(f"Layer {layer_idx} | Insufficient Dimensions")
            return

        labels = get_cluster_labels_for_image(cluster_dir, layer_idx, image_name)
        
        if labels is not None and len(labels) == len(layer_data):
            colors = cmap(labels / (labels.max() + 1))
            scatter.set_color(colors)
            ax.set_title(f"Layer {layer_idx} | {len(np.unique(labels))} Clusters | {image_name}")
        else:
            scatter.set_color('steelblue')
            status = "No Cluster Labels" if labels is None else f"Size Mismatch ({len(labels)} vs {len(layer_data)})"
            ax.set_title(f"Layer {layer_idx} | {status} | {image_name}")

        if plot_dim == 2:
            scatter.set_offsets(layer_data[:, :2])
        else:
            scatter._offsets3d = (
                layer_data[:, 0],
                layer_data[:, 1],
                layer_data[:, 2]
            )

        fig.canvas.draw_idle()

    update_plot(layer_indices[0])

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Layer', 
        0, len(layer_indices) - 1, 
        valinit=0, 
        valstep=1
    )

    def on_slider_change(val):
        """Callback for slider value change."""
        idx = int(slider.val)
        if idx < len(layer_indices):
            update_plot(layer_indices[idx])

    slider.on_changed(on_slider_change)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
