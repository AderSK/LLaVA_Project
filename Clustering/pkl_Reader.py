import os
import pickle
from pathlib import Path

import torch


def normalize_path(path: str) -> str:
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1].strip()
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def get_layer_items(val):
    """
    Normalize stored activation value to a list of tensors:
    None -> []
    single tensor -> [tensor]
    list/tuple -> list(tensors)
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def num_tokens_of_tensor(t: torch.Tensor) -> int:
    """Assume tokens are along dim 0: shape (num_tokens, hidden_dim, ...)"""
    if t.ndim == 0:
        return 1
    return int(t.shape[0])


def main():
    # 1) Ask for path
    path = input("Enter path to activations.pkl: ")
    path = normalize_path(path)

    print(f"\nResolved path: {path}")
    if not os.path.exists(path):
        print("Error: file does not exist.")
        return

    # 2) Load pickle
    print("\nLoading pickle file...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return

    if not isinstance(data, dict) or "image_name" not in data or "activations" not in data:
        print("Error: Unexpected file format. Expected dict with keys 'image_name' and 'activations'.")
        print(f"Top-level type: {type(data)}")
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
        return

    image_name = data["image_name"]
    activations = data["activations"]

    print("\n=== FILE SUMMARY ===")
    print(f"Image name: {image_name}")
    print(f"Type of 'activations': {type(activations)}")

    # 3) Show token counts per tensor per layer
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
                tokens = num_tokens_of_tensor(t)
                print(f"    Tensor {t_idx}: num_tokens={int(t.shape[0])},length of each token={int(t.shape[1])} data type={t.dtype}")
            else:
                print(f"    Tensor {t_idx}: not a torch.Tensor, type={type(t)}")

    # 4) Ask for mode
    """
    Modes:
    1 -> Print 5 Tokens [[1][2][3][4][5]]
    2 -> Print X Tokens [[1],[2],....,[x]]
    3 -> Print All Tokens [[1],[2],....,[last]]
    4 -> Quit Program
    """
    print("\nChoose mode <1:Print 5 Tokens , 2:Print X Tokens , 3:Print All Tokens , 4:Quit Program>")
    max_tokens_per_tensor = None
    while True:
        mode = input("Mode: ").strip()
        if mode == "1":
            max_tokens_per_tensor = 5
            break  # valid mode, exit loop
        elif mode == "2":
            while True:
                x_str = input("Enter X (number of tokens per tensor to print): ").strip()
                try:
                    max_tokens_per_tensor = int(x_str)
                    if max_tokens_per_tensor <= 0:
                        print("X must be positive. Please try again.")
                        continue
                    if max_tokens_per_tensor > t.shape[0]:
                        print(f"X cannot be more than the number of tokens in the tensor ({t.shape[0]}). Please try again.")
                        continue
                    break  # valid number, exit inner loop
                except ValueError:
                    print("Invalid number for X. Please enter an integer.")
            break  # exit outer loop after valid mode
        elif mode == "3":
            max_tokens_per_tensor = t.shape[0]
            break  # valid mode, exit loop
        elif mode == "4":
            print("Exiting program.")
            exit(0)  # quit the program
        else:
            print("Invalid mode. Please choose 1, 2, 3, or 4.")
            # loop continues to ask again


    # 5) Configure torch printing (no truncation of rows/cols)
    torch.set_printoptions(profile="full")

    print("\n=== PRINTING ACTIVATIONS ===")
    for layer_idx in layer_indices:
        items = get_layer_items(activations[layer_idx])

        print("\n" + "-" * 80)
        print(f"Layer {layer_idx}:")

        if not items:
            print("  No tensors stored for this layer.")
            continue

        for t_idx, act in enumerate(items):
            print("\n  " + "-" * 40)
            print(f"  Tensor {t_idx} for layer {layer_idx}:")
            print(f"    Type: {type(act)}")

            if not isinstance(act, torch.Tensor):
                print("    (Not a torch.Tensor; raw repr below)")
                act_str = repr(act)
                indented = "\n".join("    " + line for line in act_str.splitlines())
                print(indented)
                continue

            tokens_total = num_tokens_of_tensor(act)
            print(f"    Shape: {tuple(act.shape)}")
            print(f"    Data type: {act.dtype}")
            print(f"    Device: {act.device}")
            print(f"    Total tokens: {tokens_total}")
            print(f"    Printing first {max_tokens_per_tensor} token(s):")

            # Slice along token dimension 0
            if max_tokens_per_tensor == tokens_total:
                tokens_slice = act
            else:
                tokens_slice = act[:max_tokens_per_tensor]

            # Print values
            tokens_str = repr(tokens_slice)
            indented = "\n".join("    " + line for line in tokens_str.splitlines())
            print(indented)

            if max_tokens_per_tensor is not None and tokens_total > max_tokens_per_tensor:
                print(f"\n    ... ({tokens_total - max_tokens_per_tensor} more token(s) not printed in this tensor)")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()