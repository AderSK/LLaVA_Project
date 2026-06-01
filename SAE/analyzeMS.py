import os, glob, re, statistics

SAE_DIR = "/home/adam/Projects/sae_collages_MS_backup/Layer_18_TopK_64_MS"
NEURON_DIR = "/home/adam/Projects/sae_collages_MS_neurons_backup/CLIP_Layer_18_Neurons_MS"

def analyze_directory(directory, regex_pattern, label_name):
    files = glob.glob(os.path.join(directory, "*.jpg"))
    data = []
    
    for f in files:
        match = re.search(regex_pattern, os.path.basename(f))
        if match:
            item_id = int(match.group(1))
            score = float(match.group(2))
            data.append((item_id, score))
            
    if not data:
        return

    scores = [x[1] for x in data]
    
    mean_val = statistics.mean(scores)
    median_val = statistics.median(scores)
    
    threshold_count = sum(1 for s in scores if s > 0.5)
    threshold_pct = (threshold_count / len(scores)) * 100
    
    data.sort(key=lambda x: x[1], reverse=True)
    top_10 = data[:10]
    
    print(f"{'='*40}")
    print(f"--- {label_name.upper()} ANALYSIS ---")
    print(f"{'='*40}")
    print(f"Total Count: {len(scores)}")
    print(f"Mean (Avg):  {mean_val:.4f}")
    print(f"Median:      {median_val:.4f}")
    print(f"MS > 0.5:    {threshold_count} ({threshold_pct:.1f}%)")
    print(f"\nTOP 10 {label_name.upper()}:")
    
    for i, (item_id, score) in enumerate(top_10, 1):
        print(f"{i:2d}. ID: {item_id:<6} | MS Score: {score:.2f}")
    print("\n")

if __name__ == "__main__":
    sae_pattern = r'_F(\d+)_MS(-?[0-9.]+)\.jpg'
    neuron_pattern = r'_Neuron(\d+)_MS(-?[0-9.]+)\.jpg'
    
    analyze_directory(SAE_DIR, sae_pattern, "SAE Features")
    analyze_directory(NEURON_DIR, neuron_pattern, "Original Neurons")