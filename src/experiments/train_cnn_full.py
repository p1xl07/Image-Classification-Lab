"""Experiment A: SmallCNN from scratch — FULL DATA (45000/5000 split)."""
import argparse, csv, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.loading import load_cifar10, get_transforms, make_split
from src.models.definitions import SmallCNN
from src.training.train import select_device, train_one_epoch, evaluate, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-indices", type=str, default="split_indices.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_full_a")
    parser.add_argument("--csv", type=str, default="cifar_cnn_full_history.csv")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print("Experiment A — FULL DATA (SmallCNN from scratch)")
    print(f"Device: {device}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # Load split artifact to use exact indices
    import json
    with open(args.split_indices) as f:
        split_data = json.load(f)
    # Note: full run uses split_indices.json directly with Subset; if unavailable fall back
    train_full, _ = load_cifar10(data_dir="data", download=False)
    train_t, val_t = get_transforms(pretrained=False)
    train_full.transform = train_t
    # Create subsets using saved indices for reproducibility
    train_subset = torch.utils.data.Subset(train_full, split_data["train_indices"])
    val_subset = torch.utils.data.Subset(train_full, split_data["val_indices"])
    # Apply deterministic val transform to val subset's dataset
    val_subset.dataset.transform = val_t

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SmallCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","elapsed_sec"])

    overall_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        with open(args.csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])
        print(f"Epoch {epoch} | train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f} | {elapsed:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, os.path.join(args.checkpoint_dir, "best_val_loss.pt"),
                            meta={"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss, "device": str(device), "experiment": "A_full"})
        else:
            patience_counter += 1
        if args.patience and patience_counter >= args.patience:
            print("Early stopping.")
            break

    total = time.time() - overall_start
    print(f"Exp A full done in {total:.1f}s. Best val_loss={best_val_loss:.4f}")

    # Verification: load best and confirm finite outputs on sample
    best_ckpt = os.path.join(args.checkpoint_dir, "best_val_loss.pt")
    if os.path.exists(best_ckpt):
        model2 = SmallCNN(num_classes=10).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        meta = load_checkpoint(model2, optimizer2, best_ckpt, device)
        print(f"Best checkpoint loaded: epoch={meta.get('epoch')}, val_loss={meta.get('val_loss')}")
        model2.eval()
        with torch.no_grad():
            out = model2(torch.randn(2, 3, 32, 32).to(device))
        print(f"Inference: shape={tuple(out.shape)}, all finite={torch.isfinite(out).all().item()}")
    print("Experiment A full completed; official test set NOT evaluated.")

if __name__ == "__main__":
    main()
