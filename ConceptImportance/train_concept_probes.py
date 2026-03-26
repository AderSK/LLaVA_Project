"""
train_concept_probes.py

Trains one independent binary MLP per concept.
Each probe takes a LLaVA layer activation vector as input and predicts
whether that concept is present in the image (1) or not (0).

Models are saved to --output_dir as:
    people_probe.pt
    animals_probe.pt
    ... etc.

A summary JSON is also written with per-concept metrics.

Usage:
    python train_concept_probes.py \
        --labels concept_labels.json \
        --activations_dir activations \
        --layer 20

    # More options:
    python train_concept_probes.py \
        --labels concept_labels.json \
        --activations_dir activations \
        --layer 20 \
        --hidden_dims 256 128 \
        --epochs 40 \
        --output_dir probes

    # Run inference with a saved probe:
    python train_concept_probes.py \
        --predict activations/coco_00001_layer20.pkl \
        --probes_dir probes
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score
)

# ── Must match generate_concept_labels.py ─────────────────────────────────────
CONCEPTS = [
    "people",
    "animals",
    "outdoor",
    "indoor",
    "urban",
    "nature",
    "food",
    "vehicles",
    "text_or_signs",
    "nighttime",
]


# ==============================================================================
# Dataset
# ==============================================================================

class ConceptDataset(Dataset):
    """
    Loads activation .pkl files and returns (feature_vector, binary_label) pairs
    for a single concept.

    The activation tensor has shape [seq_len, hidden_dim] (float16).
    We mean-pool over the sequence to get a fixed-size vector [hidden_dim].
    """

    def __init__(self, labels_json_path, activations_dir, layer, concept):
        self.concept = concept
        self.samples = []  # list of (pkl_path, label)

        with open(labels_json_path) as f:
            label_data = json.load(f)

        activations_dir = Path(activations_dir)
        missing = 0

        for img_key, entry in label_data.items():
            if entry.get("labels") is None:
                continue

            # Locate activation file
            act_file = entry.get("activation_file")
            if act_file is None:
                stem = Path(entry["image_name"]).stem
                act_file = activations_dir / f"{stem}_layer{layer}.pkl"
            else:
                act_file = Path(act_file)

            if not act_file.exists():
                missing += 1
                continue

            label = float(entry["labels"].get(concept, 0))
            self.samples.append((act_file, label))

        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        if missing:
            print(f"  [{concept}] {missing} activation files missing")
        print(f"  [{concept}] {len(self.samples)} samples  "
              f"(pos={n_pos}, neg={n_neg}, "
              f"balance={100*n_pos/max(1,len(self.samples)):.1f}% positive)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        act_file, label = self.samples[idx]
        with open(act_file, "rb") as f:
            data = pickle.load(f)

        activation = data["activation"]  # float16 [seq_len, hidden_dim]
        if isinstance(activation, torch.Tensor):
            feature = activation.float().mean(dim=0)
        else:
            feature = torch.tensor(activation, dtype=torch.float32).mean(dim=0)

        return feature, torch.tensor(label, dtype=torch.float32)


# ==============================================================================
# Model — single binary MLP probe
# ==============================================================================

class BinaryProbe(nn.Module):
    """
    Simple MLP binary classifier.
    Input:  activation vector [hidden_dim]
    Output: single logit (apply sigmoid to get probability)
    """

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

        layers.append(nn.Linear(prev_dim, 1))  # single output logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [batch]

    def predict_proba(self, x):
        """Returns probability that concept is present (0-1)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict(self, x, threshold=0.5):
        """Returns binary prediction."""
        return (self.predict_proba(x) >= threshold).long()


# ==============================================================================
# Training a single probe
# ==============================================================================

def compute_pos_weight(dataset, indices):
    """
    Compute BCEWithLogitsLoss pos_weight to handle class imbalance.
    pos_weight = n_negative / n_positive
    """
    labels = [dataset[i][1].item() for i in indices]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def train_probe(concept, dataset, args, device):
    """Train a single binary probe for one concept. Returns (model, metrics)."""

    print(f"\n{'─'*55}")
    print(f"  Training probe: [{concept}]")
    print(f"{'─'*55}")

    if len(dataset) == 0:
        print("  No samples — skipping.")
        return None, None

    # Train / val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    if train_size == 0:
        print("  Not enough samples for a train/val split — skipping.")
        return None, None

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    # Class-balanced loss weight
    pos_weight = compute_pos_weight(dataset, train_ds.indices).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    first_feat, _ = dataset[0]
    input_dim = first_feat.shape[0]
    model = BinaryProbe(input_dim, args.hidden_dims, args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                logits = model(features)
                val_loss += criterion(logits, labels.to(device)).item()
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(labels.tolist())
        val_loss /= len(val_loader)

        # AUC (only meaningful if both classes present)
        auc_str = "N/A"
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            auc_str = f"{auc:.3f}"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ✓"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"AUC={auc_str}{marker}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation on val set
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            probs = model.predict_proba(features.to(device)).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    all_preds = [1 if p >= args.threshold else 0 for p in all_probs]

    metrics = {
        "concept": concept,
        "n_train": train_size,
        "n_val": val_size,
        "n_pos_train": int(sum(dataset[i][1].item() for i in train_ds.indices)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": float(roc_auc_score(all_labels, all_probs))
               if len(set(all_labels)) > 1 else None,
    }

    print(f"\n  Val results → "
          f"acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  "
          f"auc={metrics['auc'] if metrics['auc'] else 'N/A'}")
    print(classification_report(all_labels, all_preds,
                                 target_names=["absent", "present"],
                                 zero_division=0))

    return model, metrics


# ==============================================================================
# Save / Load
# ==============================================================================

def save_probe(model, concept, input_dim, args, output_dir):
    path = Path(output_dir) / f"{concept}_probe.pt"
    torch.save({
        "concept": concept,
        "state_dict": model.state_dict(),
        "metadata": {
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "threshold": args.threshold,
            "layer": args.layer,
        }
    }, path)
    return path


def load_probe(path, device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["metadata"]
    model = BinaryProbe(meta["input_dim"], meta["hidden_dims"], meta["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["concept"], meta


# ==============================================================================
# Inference on a single activation file
# ==============================================================================

def predict_single(pkl_path, probes_dir, device, threshold=0.5):
    """Load all saved probes and run them on one activation file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    activation = data["activation"]
    if isinstance(activation, torch.Tensor):
        feature = activation.float().mean(dim=0).unsqueeze(0).to(device)
    else:
        feature = torch.tensor(activation, dtype=torch.float32).mean(dim=0).unsqueeze(0).to(device)

    probes_dir = Path(probes_dir)
    probe_files = sorted(probes_dir.glob("*_probe.pt"))

    if not probe_files:
        print(f"No probe files found in {probes_dir}")
        return

    print(f"\nPredictions for: {Path(pkl_path).name}")
    print(f"{'Concept':<20} {'Prob':>6}  {'Label':>8}")
    print("─" * 40)

    gt = data.get("labels", {})
    for probe_file in probe_files:
        model, concept, meta = load_probe(probe_file, device)
        prob = model.predict_proba(feature).item()
        label = "YES" if prob >= meta.get("threshold", threshold) else "no"
        gt_str = ""
        if concept in gt:
            gt_str = f"  (gt: {'YES' if gt[concept] else 'no'})"
        print(f"  {concept:<18} {prob:>6.3f}  {label:>8}{gt_str}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train one binary MLP probe per concept from LLaVA activations"
    )

    # Data
    parser.add_argument("--labels", default="concept_labels.json",
                        help="Path to concept_labels.json from generate_concept_labels.py")
    parser.add_argument("--activations_dir", default="activations",
                        help="Directory containing per-image .pkl activation files")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer index whose activations to use (must match what was saved)")

    # Which concepts to train (default: all)
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts to train, e.g. --concepts people animals. "
                             "Default: train all 10.")

    # Model architecture
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128],
                        help="Hidden layer sizes e.g. --hidden_dims 256 128")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    # Output
    parser.add_argument("--output_dir", default="probes",
                        help="Directory to save trained probe .pt files")

    # Inference
    parser.add_argument("--predict", default=None,
                        help="Path to a single .pkl activation file — runs all saved probes on it")
    parser.add_argument("--probes_dir", default="probes",
                        help="Directory of saved probes (used with --predict)")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Inference mode ────────────────────────────────────────────────────────
    if args.predict is not None:
        predict_single(args.predict, args.probes_dir, device, args.threshold)
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    concepts_to_train = args.concepts if args.concepts else CONCEPTS
    invalid = [c for c in concepts_to_train if c not in CONCEPTS]
    if invalid:
        print(f"Unknown concepts: {invalid}")
        print(f"Valid options: {CONCEPTS}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Labels    : {args.labels}")
    print(f"Activations: {args.activations_dir}/  (layer {args.layer})")
    print(f"Concepts  : {concepts_to_train}")
    print(f"Architecture: {args.hidden_dims}  dropout={args.dropout}")
    print(f"Training  : {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"Output dir: {output_dir}/")

    all_metrics = []
    trained_probes = {}

    for concept in concepts_to_train:
        dataset = ConceptDataset(
            args.labels, args.activations_dir, args.layer, concept
        )

        model, metrics = train_probe(concept, dataset, args, device)

        if model is None:
            continue

        # Save probe
        first_feat, _ = dataset[0]
        input_dim = first_feat.shape[0]
        save_path = save_probe(model, concept, input_dim, args, output_dir)
        print(f"  Saved → {save_path}")

        all_metrics.append(metrics)
        trained_probes[concept] = str(save_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SUMMARY — all probes (layer {args.layer})")
    print(f"{'='*55}")
    print(f"  {'Concept':<20} {'AUC':>6}  {'F1':>6}  {'Acc':>6}")
    print(f"  {'─'*44}")

    for m in all_metrics:
        auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "  N/A"
        print(f"  {m['concept']:<20} {auc_str:>6}  "
              f"{m['f1']:>6.3f}  {m['accuracy']:>6.3f}")

    if all_metrics:
        valid_aucs = [m["auc"] for m in all_metrics if m["auc"] is not None]
        if valid_aucs:
            print(f"\n  Mean AUC: {np.mean(valid_aucs):.3f}")

    # Save summary JSON
    summary_path = output_dir / "probe_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "layer": args.layer,
            "hidden_dims": args.hidden_dims,
            "threshold": args.threshold,
            "metrics": all_metrics,
            "probe_files": trained_probes,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"\nTo run inference on an image:")
    print(f"  python train_concept_probes.py \\")
    print(f"      --predict activations/<image>_layer{args.layer}.pkl \\")
    print(f"      --probes_dir {output_dir}")


if __name__ == "__main__":
    main()
