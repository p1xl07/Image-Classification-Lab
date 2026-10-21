"""One-epoch GPU smoke test.

Uses the real functions in this repository:
- src.data.loading.load_cifar10 (download=False) + get_transforms + make_subset
- src.models.definitions.SmallCNN
- src.training.train.train_one_epoch, evaluate, save_checkpoint, load_checkpoint

Runs a single training epoch on a small deterministic subset, saves and
reloads a checkpoint, and runs inference on a sample. Reports elapsed time,
subset sizes, checkpoint verification, and whether inference succeeded.

Accuracy from this smoke test is NOT a meaningful model result.
"""
import time
import torch

from src.data.loading import load_cifar10, get_transforms
from src.models.definitions import SmallCNN
from src.training.train import (
    select_device,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    make_subset,
)


def main():
    device = select_device()
    print(f"[smoke] selected device: {device}")
    assert device.type == "cuda", f"Expected CUDA device, got {device}"

    # 1. Load CIFAR-10 (no download — must already be present)
    train_full, _ = load_cifar10(data_dir="data", download=False)
    print(f"[smoke] train partition loaded: {len(train_full)} images")

    # 2. Apply transforms + create a small deterministic subset
    train_t, _ = get_transforms(pretrained=False)
    train_full.transform = train_t
    n_per_class = 200
    subset = make_subset(train_full, n_per_class=n_per_class, seed=42)
    print(f"[smoke] subset size: {len(subset)} images "
          f"(up to {n_per_class} per class)")

    # 3. Build the model and move it to the device
    model = SmallCNN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] SmallCNN trainable parameters: {n_params:,}")

    # 4. One optimizer update over one epoch
    loader = torch.utils.data.DataLoader(
        subset, batch_size=64, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, loader, device, optimizer, criterion)
    train_time = time.time() - t0
    print(f"[smoke] epoch 1: loss={train_loss:.4f} acc={train_acc:.4f} "
          f"({train_time:.2f}s)")

    # 5. Validation pass
    val_loss, val_acc = evaluate(model, loader, device, criterion)
    print(f"[smoke] val (subset): loss={val_loss:.4f} acc={val_acc:.4f}")

    # 6. Checkpoint save/reload
    ckpt_path = "checkpoints/smoke_test.pt"
    import os
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(model, optimizer, ckpt_path,
                    meta={"epoch": 1, "device": str(device)})
    model2 = SmallCNN(num_classes=10).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(model2, optimizer2, ckpt_path, device)
    print(f"[smoke] checkpoint saved to {ckpt_path}; meta={meta}")

    # 7. Inference on a sample
    sample_x, sample_y = train_full[0]
    sample_x = sample_x.unsqueeze(0).to(device)
    model2.eval()
    with torch.no_grad():
        probs = torch.softmax(model2(sample_x), dim=1)
    pred = probs.argmax(dim=1).item()
    print(f"[smoke] inference: true={sample_y} pred={pred} "
          f"prob={probs.max().item():.4f} — succeeded")

    print("[smoke] ALL STEPS COMPLETED")


if __name__ == "__main__":
    main()