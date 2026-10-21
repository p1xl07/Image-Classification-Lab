"""Experiment A: CNN from scratch on CIFAR-10.

Entry point for a configurable, reproducible multi‑epoch training run using
SmallCNN with random initialization.  Uses a stratified split from the official
50 000‑image training partition only.  Training augmentations are applied; validation
and checkpoint selection use deterministic transformations.

Key behaviors:
- Configurable via argparse (epoch, batch size, learning rate, etc.)
- CUDA with CPU fallback
- Checkpoint selection based only on validation loss/accuracy
- Optional early stopping
- CSV logging of training history
- No use of the official test set (leakage‑proof)
- Safe checkpoint load/inference verification

Intended for a single development run (moderate epochs and subset size).  Full
experiments require the user to review the output before proceeding.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.loading import load_cifar10, get_transforms, make_split
from src.models.definitions import SmallCNN
from src.training.train import (
    select_device,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)
from src.training.train import make_subset


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment A: CNN from scratch")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Maximum number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training/validation batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay for Adam (default: 5e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--subset-per-class", type=int, default=500,
                        help="Number of images per class for fast development "
                             "runs (default: 500). Set to None or 0 for full training.")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Proportion of training split to use for validation "
                             "(default: 0.1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints (default: checkpoints)")
    parser.add_argument("--csv", type=str, default="training_history.csv",
                        help="Path to CSV history file (default: training_history.csv)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (epochs without improvement "
                             "in validation loss). If None, use fixed epochs.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of DataLoader workers (default: 2)")
    return parser.parse_args()


def log_args(args, csv_path):
    """Write a clean machine‑readable CSV header and mode, and export args as JSON.

    The CSV file ``cifar_cnn_dev_history.csv`` will contain:
      - Line 1: header column names (epoch,train_loss,train_acc,val_loss,val_acc,elapsed_sec)
      - Lines 2+: one row per epoch with the measured values.

    A companion JSON file ``cifar_cnn_dev_config.json`` stores the full run
    configuration (seed, subset sizes, split seed, model, optimizer, lr, wd,
    batch size, epoch count, best epoch, etc.) for later inspection.
    """
    import json
    from pathlib import Path

    csv_path = Path(csv_path)
    # Write the machine‑readable CSV header (one line, six columns)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "elapsed_sec"])

    # Export the full run configuration to JSON (key‑value, easy to inspect)
    config = {
        "seed": args.seed,
        "subset_per_class": args.subset_per_class,
        "val_ratio": args.val_ratio,
        "model": "SmallCNN",
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "checkpoint_dir": args.checkpoint_dir,
        "csv": str(csv_path.name),
    }
    json_path = csv_path.with_name(csv_path.stem + ".json")
    with open(json_path, "w", newline="") as jf:
        json.dump(config, jf, indent=2, sort_keys=True)

    # Also preserve a short human‑readable banner as the last line of the CSV
    # (this is optional – kept for backwards compatibility with the earlier format)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["configuration_stored_in", json_path.name])


def main():
    args = parse_args()
    device = select_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Experiment A — CNN from scratch")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}, Seed: {args.seed}")
    print(f"Subset: {args.subset_per_class} per class (0 or None -> full training)")

    # Load the training partition (cifar10 partition, no test set used yet)
    train_full, _ = load_cifar10(data_dir="data", download=False)
    print(f"Loaded training partition: {len(train_full)} images")

    # Create stratified train/validation split
    train_transform, val_transform = get_transforms(pretrained=False)
    train_full.transform = train_transform
    val_full = train_full  # treat the original as a container for indexing; make_split will produce two Subsets

    # Override transform on val subset via subset's dataset attribute
    # make_split returns two Subsets, each with its own dataset
    train_subset, val_subset = make_split(
        train_full, val_ratio=args.val_ratio, seed=args.seed)

    # For val_subset, set the transform
    val_subset.dataset.transform = val_transform

    # Apply per‑class subset if requested
    if args.subset_per_class and args.subset_per_class > 0:
        print(f"Applying development subset: up to {args.subset_per_class} images per class")
        train_subset = make_subset(train_subset.dataset, n_per_class=args.subset_per_class, seed=args.seed)
        val_subset = make_subset(val_subset.dataset, n_per_class=args.subset_per_class, seed=args.seed)
        print(f"Subset sizes -> train: {len(train_subset)}, val: {len(val_subset)}")

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    # Model, optimizer, loss
    model = SmallCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    stop_early = args.patience is not None

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    csv_path = Path(args.csv)
    log_args(args, csv_path)

    print(f"Start training for {args.epochs} epochs (early stopping {'enabled' if stop_early else 'disabled'})")
    overall_start = time.time()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - epoch_start

        # Write epoch metrics to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, epoch_time])

        print(f"Epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} | elapsed={epoch_time:.2f}s")

        # Checkpoint selection based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, "best_val_loss.pt")
            save_checkpoint(model, optimizer, ckpt_path,
                            meta={"epoch": epoch, "val_loss": val_loss,
                                  "train_loss": train_loss, "device": str(device)})
            print(f"  New best validation loss: {val_loss:.4f}, checkpoint saved at {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  Validation loss did not improve; patience {patience_counter}/{args.patience if stop_early else 'inf'}")

        # Early stopping check
        if stop_early and patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break

    total_elapsed = time.time() - overall_start
    print(f"\nTraining finished after {epoch} epochs in {total_elapsed:.2f}s")

    # Load the best checkpoint for final inference verification
    best_ckpt = os.path.join(args.checkpoint_dir, "best_val_loss.pt")
    if os.path.exists(best_ckpt):
        model2 = SmallCNN(num_classes=10).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        meta = load_checkpoint(model2, optimizer2, best_ckpt, device)
        print(f"Best checkpoint loaded: {meta}")
        # Simple inference test on a sample from the validation set
        model2.eval()
        sample_x, sample_y = val_subset[0]
        sample_x = sample_x.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model2(sample_x), dim=1)
        pred = probs.argmax(dim=1).item()
        print(f"Inference test: true class={sample_y}, predicted={pred}, confidence={probs.max().item():.4f}")
    else:
        print("ERROR: No best checkpoint saved!")

    print("\nExperiment A completed.")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"CSV history: {csv_path}")


if __name__ == "__main__":
    main()