import numpy as np
from pathlib import Path

# Tell NumPy to print numbers cleanly without the "np.float16()" wrapper
np.set_printoptions(legacy='1.25')

def inspect_npy_file(file_path, num_images, num_tokens, num_values):
    clean_path = file_path.strip('"').strip("'")
    path = Path(clean_path)
    
    if not path.exists():
        print(f"\n[!] Error: File not found at '{clean_path}'")
        return
        
    print(f"\nLoading: {path.name}...")
    
    try:
        data = np.load(path)
        
        print("\n" + "="*70)
        print(f"FILE: {path.name}")
        print("="*70)
        print(f"Shape (Dimensions): {data.shape}")
        
        is_3d = len(data.shape) == 3
        if is_3d:
            print(f"Interpretation: [{data.shape[0]} Images, {data.shape[1]} Tokens, {data.shape[2]} Hidden Values]")
        elif len(data.shape) == 2:
            print(f"Interpretation: [{data.shape[0]} Images, {data.shape[1]} Hidden Values]")
            
        memory_mb = data.nbytes / (1024 * 1024)
        print(f"Memory Size: {memory_mb:.2f} MB")
        
        print("\n" + "-"*70)
        print(f"--- Data Preview ---")
        actual_images = min(num_images, data.shape[0])
        
        if is_3d:
            actual_tokens = min(num_tokens, data.shape[1])
            actual_values = min(num_values, data.shape[2])
            print(f"Showing {actual_images} images | {actual_tokens} tokens per image | {actual_values} values per token:\n")
            for img_idx in range(actual_images):
                print(f"Image {img_idx} (Row {img_idx}): [")
                for tok_idx in range(actual_tokens):
                    # Convert to standard Python floats for clean printing
                    vals = [float(x) for x in data[img_idx, tok_idx, :actual_values]]
                    comma = "," if tok_idx < actual_tokens - 1 else ""
                    print(f"  Token {tok_idx}: {vals} ...{comma}")
                print("]\n")
                
        elif len(data.shape) == 2:
            actual_values = min(num_values, data.shape[1])
            print(f"Showing {actual_images} images | {actual_values} values per image:\n")
            for img_idx in range(actual_images):
                # Convert to standard Python floats for clean printing
                vals = [round(float(x), 6) for x in data[img_idx, :actual_values]]
                print(f"Image {img_idx} (Row {img_idx}): {vals} ...")
                
        print("="*70)
        
    except Exception as e:
        print(f"\n[!] Failed to load or read the file. Error: {e}")

if __name__ == "__main__":
    print("NumPy (.npy) Inspector Tool")
    try:
        print("\n--- Display Preferences ---")
        img_input = input("How many IMAGES (Rows)? [Default: 2]: ").strip()
        num_images = int(img_input) if img_input else 2
        
        tok_input = input("How many TOKENS? (Ignores if 2D data) [Default: 3]: ").strip()
        num_tokens = int(tok_input) if tok_input else 3
        
        val_input = input("How many VALUES? [Default: 5]: ").strip()
        num_values = int(val_input) if val_input else 5
    except ValueError:
        num_images, num_tokens, num_values = 2, 3, 5

    while True:
        print("\n" + "-"*70)
        user_input = input("Enter the full path to the .npy file (or 'q' to quit): ").strip()
        if user_input.lower() in ['quit', 'q', 'exit']: break
        if not user_input: continue
        inspect_npy_file(user_input, num_images, num_tokens, num_values)
