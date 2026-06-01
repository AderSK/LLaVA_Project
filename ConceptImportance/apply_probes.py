"""
apply_probes.py

Run all trained binary concept probes on a directory of activation .pkl files
(from collect_activations.py) and save predictions per image to a JSON file.

Usage:
    python apply_probes.py \
        --probes_dir ./probes \
        --activations_dir ./flickr_activations \
        --layer 20 \
        --output probe_predictions.json

    # Override the saved threshold (e.g. require higher confidence)
    python apply_probes.py \
        --probes_dir ./probes \
        --activations_dir ./flickr_activations \
        --layer 20 \
        --threshold 0.7 \
        --output probe_predictions_strict.json
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn


# ─── BinaryProbe — must match train_concept_probes.py ────────────────────────

class BinaryProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def load_probe(path, device):
    # weights_only=False is required because we save metadata + state_dict together
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["metadata"]
    model = BinaryProbe(meta["input_dim"], meta["hidden_dims"], meta["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["concept"], meta


def get_activation_files(activations_dir, layer):
    activations_dir = Path(activations_dir)
    return sorted(activations_dir.glob(f"*_layer{layer}.pkl"))


def load_feature(pkl_path, device):
    """Load activation .pkl and return (mean-pooled feature [1, hidden_dim], data dict)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    activation = data["activation"]
    if isinstance(activation, torch.Tensor):
        feature = activation.float().mean(dim=0)
    else:
        feature = torch.tensor(activation, dtype=torch.float32).mean(dim=0)
    return feature.unsqueeze(0).to(device), data


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained concept probes to a directory of activation .pkl files"
    )
    parser.add_argument("--probes_dir", default="probes",
                        help="Directory containing *_probe.pt files")
    parser.add_argument("--activations_dir", required=True,
                        help="Directory with per-image *_layerN.pkl activation files")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer index whose activations to use (must match probe layer)")
    parser.add_argument("--output", default="probe_predictions.json",
                        help="Output JSON file with predictions")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override probe threshold (default: per-probe saved threshold)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for probe forward passes (default: 32)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all probes
    probes_dir = Path(args.probes_dir)
    probe_files = sorted(probes_dir.glob("*_probe.pt"))
    if not probe_files:
        print(f"No probe files found in {probes_dir}")
        return

    print(f"Loading {len(probe_files)} probes from {probes_dir}/")
    probes = {}  # concept -> (model, threshold)
    for pf in probe_files:
        model, concept, meta = load_probe(pf, device)
        if meta.get("layer") != args.layer:
            print(f"  [warn] {pf.name} was trained on layer {meta.get('layer')}, "
                  f"not {args.layer} — make sure the activations match the probe.")
        threshold = args.threshold if args.threshold is not None else meta.get("threshold", 0.5)
        probes[concept] = (model, threshold)
        print(f"  Loaded {concept:<16}  layer={meta.get('layer')}  "
              f"hidden={meta.get('hidden_dims')}  threshold={threshold}")

    # Find activation files
    act_files = get_activation_files(args.activations_dir, args.layer)
    if not act_files:
        print(f"\nNo activation files matching *_layer{args.layer}.pkl in {args.activations_dir}")
        return
    print(f"\nFound {len(act_files)} activation files.\n")

    predictions = {}

    # Process in batches: load N features at a time, run all probes on the batch
    for batch_start in range(0, len(act_files), args.batch_size):
        batch_files = act_files[batch_start:batch_start + args.batch_size]
        feats = []
        metas = []
        for pkl_path in batch_files:
            try:
                feature, data = load_feature(pkl_path, device)
                feats.append(feature)
                metas.append((pkl_path, data))
            except Exception as e:
                print(f"  [error] {pkl_path.name}: {e}")
                predictions[str(pkl_path)] = {
                    "image_name": pkl_path.stem,
                    "image_path": str(pkl_path),
                    "error": str(e),
                }

        if not feats:
            continue

        batch = torch.cat(feats, dim=0)  # [B, hidden_dim]

        # Run every probe on the batch
        per_concept_probs = {}
        for concept, (model, _threshold) in probes.items():
            probs = model.predict_proba(batch).cpu().tolist()
            per_concept_probs[concept] = probs

        # Distribute back into per-image dicts
        for i, (pkl_path, data) in enumerate(metas):
            image_path = data.get("image_path", str(pkl_path))
            image_name = data.get("image_name", pkl_path.stem)

            preds = {}
            for concept, (_model, threshold) in probes.items():
                prob = per_concept_probs[concept][i]
                preds[concept] = {
                    "prob": float(prob),
                    "label": int(prob >= threshold),
                }

            predictions[image_path] = {
                "image_name": image_name,
                "image_path": image_path,
                "predictions": preds,
            }

        done = batch_start + len(batch_files)
        # Print a sample line
        sample_path, sample_data = metas[0]
        sample_preds = predictions[sample_data.get("image_path", str(sample_path))]["predictions"]
        sample_detected = [c for c, v in sample_preds.items() if v["label"] == 1]
        print(f"  [{done}/{len(act_files)}]  e.g. {sample_data.get('image_name', sample_path.stem)}: {sample_detected}")

    # Save
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to: {args.output}")

    # Frequency summary
    valid = [v for v in predictions.values() if "predictions" in v]
    if valid:
        print(f"\nProbe-predicted concept frequency  (n={len(valid)})\n")
        print(f"  {'concept':<18} {'count':>10}   bar")
        print(f"  {'-'*18} {'-'*10}   {'-'*30}")
        for concept in probes.keys():
            count = sum(1 for v in valid if v["predictions"][concept]["label"] == 1)
            pct = count / len(valid)
            bar = "█" * int(pct * 30)
            print(f"  {concept:<18} {count:4d}/{len(valid)}  {bar}  ({pct*100:.1f}%)")


if __name__ == "__main__":
    main()
