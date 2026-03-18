"""
K-means Automation Runner
=========================

Automatically calls the K-means clustering script in increments of 2
(from K=2 to K=200). 
"""

import subprocess
import sys
from pathlib import Path

# --- CONFIGURATION ---
# The name of your actual K-means script file
KMEANS_SCRIPT_NAME = "Kmeans.py" 

# File Paths (Change these to your actual paths)
NPY_FILE = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Concatenated_Layers\layer_0_concatenated.npy"
CSV_FILE = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Concatenated_Layers\image_row_mapping.csv"
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Results"

def run_clustering_loop():
    script_path = Path(__file__).parent / KMEANS_SCRIPT_NAME
    
    if not script_path.exists():
        print(f"Error: Could not find {KMEANS_SCRIPT_NAME} in the current folder.")
        return
        
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Starting Automated Clustering Loop (K=2 to K=200)")
    print("=" * 70)
    
    # Loop from 2 to 200, stepping by 2 (2, 4, 6, 8 ... 200)
    for k in range(2, 201, 2):
        print(f"\n[Runner] Initiating subprocess for K={k}...")
        
        # Build the command just like you would type it in the terminal
        command = [
            sys.executable,          # 'py' or 'python'
            str(script_path),        # The script to run
            NPY_FILE,                # sys.argv[1]
            CSV_FILE if CSV_FILE else "none", # sys.argv[2]
            OUTPUT_DIR,              # sys.argv[3]
            str(k)                   # sys.argv[4]
        ]
        
        try:
            # Run the command and wait for it to finish
            result = subprocess.run(command, check=True)
            if result.returncode == 0:
                print(f"[Runner] Successfully completed K={k}")
        except subprocess.CalledProcessError as e:
            print(f"[Runner] Error occurred during K={k}. Moving to next...")
            continue
            
    print("\n" + "=" * 70)
    print("All clustering iterations have finished!")
    print("=" * 70)

if __name__ == "__main__":
    run_clustering_loop()
