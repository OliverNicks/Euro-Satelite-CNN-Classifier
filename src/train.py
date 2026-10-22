"""Training script for EuroSAT Satellite Image Classification using SateliteCNN.

Usage:
    python -m src.train --data-dir data --epochs 25 --batch-size 32 --lr 0.001
"""

import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import SateliteDataset, get_train_transforms, get_inference_transforms, preprocess_dataset
from src.model import SateliteCNN


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(loader)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def train(
    data_dir: str,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 4,
    save_path: str = "app/satellite_model.pth",
    device_name: str = "auto",
) -> None:
    set_seed(42)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Using training device: {device}")

    print(f"Scanning dataset in: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    df, labels_dict = preprocess_dataset(data_dir)
    print(f"Found {len(df)} images across {len(labels_dict)} classes: {list(labels_dict.keys())}")

    # Stratified train/val split (80/20)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train split: {len(train_df)} | Validation split: {len(val_df)}")

    train_ds = SateliteDataset(train_df, transform=get_train_transforms())
    val_ds = SateliteDataset(val_df, transform=get_inference_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SateliteCNN(num_classes=len(labels_dict)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            patience_counter = 0
            print(f"  -> Saved new best model to {save_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best checkpoint saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SateliteCNN on EuroSAT data.")
    parser.add_argument("--data-dir", default="data", help="Path to class-subfolder dataset.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs (default: 25).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4).")
    parser.add_argument("--save-path", default="app/satellite_model.pth", help="Checkpoint output path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()
    try:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            save_path=args.save_path,
            device_name=args.device,
        )
    except Exception as err:
        print(f"Training error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
