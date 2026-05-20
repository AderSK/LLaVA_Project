import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\PCA_Reduced_Layers"
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_andPCA\PCA_Plots"

def parse_stats_text(text):
    """Parses the text to extract the individual variance percentages."""
    matches = re.findall(r'PC\d+\s+([0-9.]+)%', text)
    individual_vars = np.array([float(m) for m in matches])
    return individual_vars

def plot_pca_variance(individual_vars, title, save_path):
    """Creates an image with 2 side-by-side line graphs (Individual and Cumulative)."""
    components = np.arange(1, len(individual_vars) + 1)
    cumulative_vars = np.cumsum(individual_vars)

    # Create 2 subplots horizontally (1 row, 2 columns) with a wider figure size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main title for the entire image
    fig.suptitle(f'PCA Explained Variance - {title}', fontsize=16, fontweight='bold')

    # --- 1. Left Graph: Individual Variance (Line Plot) ---
    ax1.plot(components, individual_vars, color='tab:blue', marker='o', markersize=3, 
             linewidth=2, label='Individual Variance')
    ax1.set_title('Individual Component Variance', fontsize=12)
    ax1.set_xlabel('Principal Component (PC)', fontsize=10)
    ax1.set_ylabel('Explained Variance (%)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 2. Right Graph: Cumulative Variance (Line Plot) ---
    ax2.plot(components, cumulative_vars, color='tab:red', marker='o', markersize=3, 
             linewidth=2, label='Cumulative Variance')
    ax2.set_title('Cumulative Component Variance', fontsize=12)
    ax2.set_xlabel('Principal Component (PC)', fontsize=10)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Set y-axis limit for cumulative chart to have a bit of padding above the max value
    max_cum = min(105.0, cumulative_vars[-1] + 5)
    ax2.set_ylim(0, max_cum)
    ax2.legend()

    # Automatically adjust spacing so titles and labels don't overlap
    fig.tight_layout()
    # Add a little extra space at the top so the suptitle doesn't overlap
    fig.subplots_adjust(top=0.88)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  -> Plot saved to {save_path.name}")
    plt.close()

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    txt_files = sorted(input_path.glob("*_pca_stats.txt"))
    
    # Throw an error if no data is found
    if not txt_files:
        raise FileNotFoundError(f"No *_pca_stats.txt files found in directory: {input_path}")
        
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(txt_files)} PCA stats files. Plotting...\n")
    
    for file_path in txt_files:
        print(f"Plotting {file_path.name}...")
        with open(file_path, 'r') as f:
            content = f.read()
            
        individual_vars = parse_stats_text(content)
        layer_name = file_path.stem.replace("_pca_stats", "")
        save_file = output_path / f"{layer_name}_pca_plot.png"
        
        plot_pca_variance(individual_vars, title=layer_name, save_path=save_file)
        
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
