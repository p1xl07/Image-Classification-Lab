"""Experiment C: Fine-tuned pretrained ResNet-18 on CIFAR-10.

Initializes from ImageNet-pretrained weights (not from Exp B).  Keeps earlier
backbone blocks frozen; unfreezes the final block (layer4) and the new 10-class
classification head for fine-tuning.  Uses a lower LR for pretrained layers
that may be supported if needed; here a single Adam optimizer over the
trainable set is sufficient given the frozen backbone.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.loading import load_cifar10, get_transforms, make_split
from src.models.definitions import build_resnet18
from src.training.train import (
    select_device, train_one_epoch, evaluate, save_checkpoint, load_checkpoint,
    make_subset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment C: Fine-tuned ResNet-18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-per-class", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_c")
    parser.add_argument("--csv", type=str, default="cifar_resnet_finetuned_history.csv")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Experiment C — Fine-tuned pretrained ResNet-18")
    print(f"Device: {device}, Init: ImageNet weights, Freeze: backbone except layer4 + fc")

    train_full, _ = load_cifar10(data_dir="data", download=False)
    train_t, val_t = get_transforms(pretrained=True)
    train_full.transform = train_t
    train_subset, val_subset = make_split(train_full, val_ratio=args.val_ratio, seed=args.seed)
    val_subset.dataset.transform = val_t
    if args.subset_per_class and args.subset_per_class > 0:
        train_subset = make_subset(train_subset.dataset, n_per_class=args.subset_per_class, seed=args.seed)
        val_subset = make_subset(val_subset.dataset, n_per_class=args.subset_per_class, seed=args.seed)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Initialize from pretrained weights (not from Exp B checkpoint)
    model = build_resnet18(num_classes=10, pretrained=True, freeze_backbone=False).to(device)
    # Unfreeze only layer4 + fc; freeze earlier backbone
    for name, p in model.named_parameters():
        if "layer4" in name or "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"Trainable params count: {len(trainable)} (layer4 + fc) | Frozen: {len(frozen)}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    csv_path = Path(args.csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "elapsed_sec"])

    total_start = time.time()
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])
        print(f"Epoch {epoch} | train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f} | {elapsed:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, os.path.join(args.checkpoint_dir, "best_val_loss.pt"),
                            meta={"epoch": epoch, "val_loss": val_loss, "device": str(device), "experiment": "C"})
        else:
            patience_counter += 1
        if args.patience and patience_counter >= args.patience:
            print("Early stopping.")
            break

    total = time.time() - total_start
    ckpt_path = os.path.join(args.checkpoint_dir, "best_val_loss.pt")
    meta = load_checkpoint(model, optimizer, ckpt_path, device) if os.path.exists(ckpt_path) else {}
    print(f"Exp C finished in {total:.1f}s. Best epoch={meta.get('epoch')}, best val_loss={meta.get('val_loss'):.4f}")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224).to(device))
    print(f"Inference check: output shape={tuple(out.shape)}, all finite={torch.isfinite(out).all().item()}")

if __name__ == "__main__":
    main()
