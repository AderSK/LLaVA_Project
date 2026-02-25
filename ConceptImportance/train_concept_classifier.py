"""
train_concept_classifier.py

Trains a MLP that maps LLaVA layer activations → concept probabilities.

Each head outputs a sigmoid probability (0-1) for one concept (multi-label).
At inference time you can also get a softmax distribution over concepts to
see *which* concept the activation most strongly indicates.

Usage:
    # Basic training
    python train_concept_classifier.py \
        --labels concept_labels.json \
        --activations_dir activations \
        --layer 20

    # With options
    python train_concept_classifier.py \
        --labels concept_labels.json \
        --activations_dir activations \
        --layer 20 \
        --hidden_dims 512 256 \
        --epochs 30 \
        --threshold 0.5 \
        --save_model concept_mlp.pt

    # Inference on a single .pkl file
    python train_concept_classifier.py \
        --predict activations/my_image_layer20.pkl \
        --load_model concept_mlp.pt
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score

# Must match the list in generate_concept_labels.py
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

class ActivationDataset(Dataset):
    """
    Loads per-image activation .pkl files produced by generate_concept_labels.py.
    Each item: (feature_vector [hidden_dim], label_vector [n_concepts])

    The activation tensor has shape [seq_len, hidden_dim] (float16).
    We reduce over sequence length with mean pooling to get [hidden_dim].
    """

    def __init__(self, labels_json_path, activations_dir, layer, concepts=CONCEPTS):
        self.concepts = concepts
        self.samples = []

        with open(labels_json_path) as f:
            label_data = json.load(f)

        activations_dir = Path(activations_dir)
        missing = 0

        for img_key, entry in label_data.items():
            if entry.get("labels") is None:
                continue

            # Find the activation file
            act_file = entry.get("activation_file")
            if act_file is None:
                # Try to infer the path
                stem = Path(entry["image_name"]).stem
                act_file = activations_dir / f"{stem}_layer{layer}.pkl"
            else:
                act_file = Path(act_file)

            if not act_file.exists():
                missing += 1
                continue

            label_vec = torch.tensor(
                [float(entry["labels"].get(c, 0)) for c in concepts],
                dtype=torch.float32
            )
            self.samples.append((act_file, label_vec))

        print(f"Dataset: {len(self.samples)} samples loaded, {missing} activation files missing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        act_file, label_vec = self.samples[idx]
        with open(act_file, "rb") as f:
            data = pickle.load(f)

        activation = data["activation"]  # float16 Tensor [seq_len, hidden_dim]
        # Mean-pool over sequence length → [hidden_dim]
        if isinstance(activation, torch.Tensor):
            feature = activation.float().mean(dim=0)
        else:
            feature = torch.tensor(activation, dtype=torch.float32).mean(dim=0)

        return feature, label_vec


# ==============================================================================
# Model
# ==============================================================================

class ConceptMLP(nn.Module):
    """
    MLP classifier for concept detection from activation vectors.

    Architecture:
        Input [hidden_dim]
        → Linear → LayerNorm → GELU → Dropout
        → (repeat for each hidden layer)
        → Linear → [n_concepts]   (logits)

    During inference:
        sigmoid(logits)  → per-concept probability  (independent, multi-label)
        softmax(logits)  → relative concept ranking (which concept dominates)
    """

    def __init__(self, input_dim, hidden_dims, n_concepts, dropout=0.3):
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

        layers.append(nn.Linear(prev_dim, n_concepts))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits

    def predict_proba(self, x):
        """Multi-label sigmoid probabilities for each concept."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict_softmax(self, x):
        """Softmax distribution — shows which concept dominates."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# ==============================================================================
# Training
# ==============================================================================

def train(model, train_loader, val_loader, epochs, lr, device, concepts):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
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

        scheduler.step()
        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                val_loss += criterion(logits, labels).item()
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Per-concept AUC
        aucs = []
        for i, c in enumerate(concepts):
            if all_labels[:, i].sum() > 0 and all_labels[:, i].sum() < len(all_labels):
                aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            else:
                aucs.append(float("nan"))
        mean_auc = np.nanmean(aucs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ← best"
        else:
            marker = ""

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"mean_AUC={mean_auc:.3f}{marker}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, loader, device, concepts, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            all_probs.append(model.predict_proba(features).cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_probs >= threshold).astype(int)

    print("\n" + "="*60)
    print(f"Evaluation (threshold={threshold})")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=concepts, zero_division=0))

    print("Per-concept AUC-ROC:")
    for i, c in enumerate(concepts):
        if all_labels[:, i].sum() > 0 and all_labels[:, i].sum() < len(all_labels):
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            print(f"  {c:18s}  AUC={auc:.3f}")
        else:
            print(f"  {c:18s}  AUC=N/A (single class in split)")


# ==============================================================================
# Inference helper
# ==============================================================================

def predict_single(model, pkl_file, device, concepts, threshold=0.5):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    activation = data["activation"]
    if isinstance(activation, torch.Tensor):
        feature = activation.float().mean(dim=0)
    else:
        feature = torch.tensor(activation, dtype=torch.float32).mean(dim=0)

    feature = feature.unsqueeze(0).to(device)  # [1, hidden_dim]

    sigmoid_probs = model.predict_proba(feature)[0]
    softmax_probs = model.predict_softmax(feature)[0]

    print(f"\nPredictions for: {Path(pkl_file).name}")
    print(f"{'Concept':<20} {'Sigmoid':>8}  {'Softmax':>8}  {'Label':>6}")
    print("-" * 50)
    for i, c in enumerate(concepts):
        sig = sigmoid_probs[i].item()
        sft = softmax_probs[i].item()
        label = "YES" if sig >= threshold else "no"
        print(f"{c:<20} {sig:>8.3f}  {sft:>8.3f}  {label:>6}")

    # Ground-truth if available
    gt = data.get("labels")
    if gt:
        print("\nGround-truth labels:", {c: v for c, v in gt.items() if v == 1})


# ==============================================================================
# Checkpoint helpers
# ==============================================================================

def save_checkpoint(model, path, metadata):
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    print(f"Model saved to: {path}")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["metadata"]
    model = ConceptMLP(
        input_dim=meta["input_dim"],
        hidden_dims=meta["hidden_dims"],
        n_concepts=meta["n_concepts"],
        dropout=meta.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from: {path}")
    print(f"  input_dim={meta['input_dim']}, hidden={meta['hidden_dims']}, "
          f"n_concepts={meta['n_concepts']}, layer={meta.get('layer')}")
    return model, meta


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MLP concept classifier on LLaVA activations")

    # Data
    parser.add_argument("--labels", default="concept_labels.json")
    parser.add_argument("--activations_dir", default="activations")
    parser.add_argument("--layer", type=int, default=20,
                        help="Which layer's activations to use (must match what was saved)")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256],
                        help="Hidden layer sizes, e.g. --hidden_dims 512 256 128")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary classification")
    parser.add_argument("--save_model", default="concept_mlp.pt")

    # Inference mode
    parser.add_argument("--predict", default=None,
                        help="Path to a single .pkl activation file to classify")
    parser.add_argument("--load_model", default=None,
                        help="Path to a saved .pt model for inference")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Inference only ----
    if args.predict is not None:
        assert args.load_model is not None, "Provide --load_model for inference"
        model, _ = load_checkpoint(args.load_model, device)
        predict_single(model, args.predict, device, CONCEPTS, args.threshold)
        return

    # ---- Load dataset ----
    dataset = ActivationDataset(args.labels, args.activations_dir, args.layer, CONCEPTS)
    if len(dataset) == 0:
        print("No samples found. Check that --labels and --activations_dir are correct.")
        return

    # Infer input_dim from first sample
    first_feat, _ = dataset[0]
    input_dim = first_feat.shape[0]
    print(f"Input dimension (hidden_dim of layer {args.layer}): {input_dim}")

    # Train / val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"Split: {train_size} train / {val_size} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- Build model ----
    model = ConceptMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        n_concepts=len(CONCEPTS),
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model}")
    print(f"Trainable parameters: {total_params:,}")

    # Label balance stats
    print("\nLabel distribution in training set:")
    all_labels = torch.stack([dataset[i][1] for i in train_ds.indices])
    for i, c in enumerate(CONCEPTS):
        pos = all_labels[:, i].sum().item()
        print(f"  {c:18s}  {int(pos):4d}/{train_size} positive  ({100*pos/train_size:.1f}%)")

    # ---- Train ----
    print(f"\nTraining for {args.epochs} epochs...")
    model = train(model, train_loader, val_loader, args.epochs, args.lr, device, CONCEPTS)

    # ---- Evaluate ----
    evaluate(model, val_loader, device, CONCEPTS, args.threshold)

    # ---- Save ----
    metadata = {
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "n_concepts": len(CONCEPTS),
        "concepts": CONCEPTS,
        "dropout": args.dropout,
        "layer": args.layer,
        "threshold": args.threshold,
    }
    save_checkpoint(model, args.save_model, metadata)

    print("\nDone! Quick-start inference:")
    print(f"  python train_concept_classifier.py "
          f"--predict <activation.pkl> --load_model {args.save_model}")


if __name__ == "__main__":
    main()
