"""
K-Means Orchestrator & Leaderboard
==================================

Automatically runs K-means from K=2 to 200, scores each result using the 
validation script, and maintains a leaderboard to find the optimal K size.

"""

import subprocess
import sys
import json
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
KMEANS_SCRIPT = "Kmeans.py"
SCORING_SCRIPT = "ScoreClusters.py"

NPY_FILE = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Concatenated_Layers\layer_0_concatenated.npy"
CSV_FILE = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Concatenated_Layers\image_row_mapping.csv"
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Results"

# The metric to sort the top 5 leaderboard by (Options: "Silhouette", "Davies-Bouldin", "Calinski-Harabasz")
SORT_METRIC = "Silhouette"
# True if higher is better (Silhouette), False if lower is better (Davies-Bouldin)
SORT_ASCENDING = False 

def run_pipeline():
    kmeans_path = Path(__file__).parent / KMEANS_SCRIPT
    score_path = Path(__file__).parent / SCORING_SCRIPT
    npy_dir = Path(NPY_FILE).parent
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Starting Automated K-Means & Validation Pipeline (K=2 to 200)")
    print("=" * 80)
    
    leaderboard = []

    for k in range(2, 201, 2):
        print(f"\n---> [K={k}] 1. Running Clustering...")
        
        # 1. Run K-Means
        kmeans_cmd = [
            sys.executable, str(kmeans_path), NPY_FILE, CSV_FILE, OUTPUT_DIR, str(k)
        ]
        
        try:
            subprocess.run(kmeans_cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"     [Error] Clustering failed for K={k}")
            continue

        # Find the generated .pkl file
        base_name = Path(NPY_FILE).stem
        pkl_file = Path(OUTPUT_DIR) / f"{base_name}_clusters{k}.pkl"
        
        if not pkl_file.exists():
            print(f"     [Error] Could not find output file for K={k}")
            continue

        print(f"     [K={k}] 2. Running Validation Metrics...")
        
        # 2. Run Scoring
        score_cmd = [
            sys.executable, str(score_path), str(pkl_file), str(npy_dir)
        ]
        
        try:
            # Capture the JSON output from the scoring script
            score_result = subprocess.run(score_cmd, check=True, capture_output=True, text=True)
            
            # The scoring script prints status updates. We only want the final JSON line.
            for line in score_result.stdout.strip().split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    metrics = json.loads(line)
                    leaderboard.append(metrics)
                    print(f"     [Success] Silhouette: {metrics['Silhouette']:.4f} | DB Index: {metrics['Davies-Bouldin']:.4f}")
                    break
                    
        except subprocess.CalledProcessError:
            print(f"     [Error] Validation failed for K={k}")
            continue

    # --- 3. PRINT LEADERBOARD ---
    if not leaderboard:
        print("\nPipeline finished, but no results were recorded.")
        return
        
    print("\n" + "=" * 80)
    print(f"🏆 TOP 5 OPTIMAL K SIZES (Ranked by {SORT_METRIC}) 🏆")
    print("=" * 80)
    
    # Sort the list of dictionaries based on the chosen metric
    leaderboard.sort(key=lambda x: x[SORT_METRIC], reverse=not SORT_ASCENDING)
    
    # Save the full leaderboard to CSV for your thesis
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_csv = Path(OUTPUT_DIR) / "clustering_leaderboard_all.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    
    # Print the top 5
    print(f"{'Rank':<5} | {'K':<5} | {'Silhouette ↑':<14} | {'DB Index ↓':<12} | {'CH Index ↑':<15}")
    print("-" * 65)
    
    for i, res in enumerate(leaderboard[:5], 1):
        print(f"#{i:<4} | {res['Clusters']:<5} | {res['Silhouette']:<14.4f} | {res['Davies-Bouldin']:<12.4f} | {res['Calinski-Harabasz']:<15.2f}")
        
    print("\n" + "=" * 80)
    print(f"Full metrics saved to: {leaderboard_csv.name}")

if __name__ == "__main__":
    run_pipeline()
