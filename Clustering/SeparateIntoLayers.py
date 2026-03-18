import os
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Activations"
OUTPUT_DIR = r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Concatenated_Layers"
OUTPUT_FORMAT = "npy" 

def concatenate_and_track_rows():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    pkl_files = list(input_path.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {INPUT_DIR}")
        return
        
    print(f"Found {len(pkl_files)} activation files. Processing variable token lengths...")

    layer_data = {}
    
    # We will use this list to build a Pandas DataFrame for perfect tracking
    mapping_data = []
    
    # A counter to keep track of our current row index as we stack
    current_row_index = 0

    # --- STEP 1: LOAD, TRACK, AND GROUP ---
    for i, file_path in enumerate(pkl_files, 1):
        if i % 20 == 0:
            print(f"  Loading file {i}/{len(pkl_files)}...")
            
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            image_name = data["image_name"]
            activations = data["activations"]
            
            # Figure out how many tokens this specific image has (using layer 0 as reference)
            # Assuming shape is (tokens, 4096) or (1, tokens, 4096)
            ref_array = activations[list(activations.keys())[0]]
            
            # Clean dimension if it's 3D
            if len(ref_array.shape) == 3:
                ref_array = np.squeeze(ref_array, axis=0)
                
            num_tokens = ref_array.shape[0]
            start_row = current_row_index
            end_row = current_row_index + num_tokens - 1
            
            # Record the exact row span for this image
            mapping_data.append({
                "Image_Name": image_name,
                "Start_Row": start_row,
                "End_Row": end_row,
                "Total_Tokens": num_tokens
            })
            
            current_row_index += num_tokens
            
            # Store the arrays for vstacking
            for layer_idx, act_array in activations.items():
                if act_array is not None:
                    if layer_idx not in layer_data:
                        layer_data[layer_idx] = []
                        
                    if len(act_array.shape) == 3:
                        act_array = np.squeeze(act_array, axis=0)
                        
                    layer_data[layer_idx].append(act_array)
                    
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Save the mapping table to a CSV file
    mapping_df = pd.DataFrame(mapping_data)
    csv_file = output_path / "image_row_mapping.csv"
    mapping_df.to_csv(csv_file, index=False)
    print(f"\nSaved exact row mapping to: {csv_file.name}")
    print(mapping_df.head()) # Print a preview of the tracking sheet

    # --- STEP 2: VSTACK AND SAVE ---
    print("\nVertically stacking token vectors...")
    
    for layer_idx, arrays in layer_data.items():
        print(f"  Processing Layer {layer_idx}...")
        
        # Blindly stack them all together (this is safe because we tracked the rows!)
        concatenated_array = np.vstack(arrays)
        
        print(f"    Final shape for Layer {layer_idx}: {concatenated_array.shape}")

        if OUTPUT_FORMAT.lower() == "npy":
            save_file = output_path / f"layer_{layer_idx}_concatenated.npy"
            np.save(save_file, concatenated_array)
        else:
            save_file = output_path / f"layer_{layer_idx}_concatenated.pt"
            tensor_data = torch.from_numpy(concatenated_array)
            torch.save(tensor_data, save_file)
            
        print(f"    Saved to: {save_file.name}")

    print(f"\nDone! Total rows processed: {current_row_index}")

if __name__ == "__main__":
    concatenate_and_track_rows()
