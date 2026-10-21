"""Experiment C: Fine-tuned ResNet-18 — FULL DATA (45000/5000 split)."""
import argparse, csv, os, time, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.loading import load_cifar10, get_transforms
from src.models.definitions import build_resnet18
from src.training.train import select_device, train_one_epoch, evaluate, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-indices", type=str, default="split_indices.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_full_c")
    parser.add_argument("--csv", type=str, default="cifar_resnet_finetuned_full_history.csv")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    device = select_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    print("Experiment C FULL — Fine-tuned ResNet-18 (layer4+fc)")
    train_full, _ = load_cifar10(data_dir="data", download=False)
    train_t, val_t = get_transforms(pretrained=True)
    train_full.transform = train_t
    with open(args.split_indices) as f: split_data = json.load(f)
    train_subset = torch.utils.data.Subset(train_full, split_data["train_indices"])
    val_subset = torch.utils.data.Subset(train_full, split_data["val_indices"])
    val_subset.dataset.transform = val_t
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_resnet18(num_classes=10, pretrained=True, freeze_backbone=False).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = ("layer4" in n or "fc" in n)
    trainable = sum(p.requires_grad for p in model.parameters())
    frozen = sum(not p.requires_grad for p in model.parameters())
    print(f"Trainable={trainable} (layer4+fc), Frozen={frozen}")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float("inf"); patience_counter = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.csv, "w", newline="") as f: csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","elapsed_sec"])
    t_start = time.time()
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, device, optimizer, criterion)
        vl, va = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        with open(args.csv, "a", newline="") as f: csv.writer(f).writerow([epoch, tl, ta, vl, va, elapsed])
        print(f"Epoch {epoch} | train={tl:.4f}/{ta:.4f} val={vl:.4f}/{va:.4f} | {elapsed:.2f}s")
        if vl < best_val_loss:
            best_val_loss = vl; patience_counter = 0
            save_checkpoint(model, optimizer, os.path.join(args.checkpoint_dir, "best_val_loss.pt"), meta={"epoch":epoch,"val_loss":vl,"device":str(device),"experiment":"C_full"})
        else:
            patience_counter += 1
        if args.patience and patience_counter >= args.patience: print("Early stop."); break
    total = time.time() - t_start
    best_path = os.path.join(args.checkpoint_dir, "best_val_loss.pt")
    meta = load_checkpoint(model, optimizer, best_path, device) if os.path.exists(best_path) else {}
    print(f"Exp C full done in {total:.1f}s. Best val_loss={meta.get('val_loss'):.4f}@epoch{meta.get('epoch')}")
    model.eval()
    with torch.no_grad(): out = model(torch.randn(2, 3, 224, 224).to(device))
    print(f"Finite inference: shape={tuple(out.shape)}, all_finite={torch.isfinite(out).all().item()}")
    print("No official test set loaded or evaluated.")

if __name__ == "__main__":
    main()
