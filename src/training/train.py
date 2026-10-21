"""Training loop and checkpoint utilities.

Provides a minimal but real training/evaluation loop used for the smoke test
and later experiments. Designed to run on CUDA when available with a CPU
fallback. No hyperparameter search framework is included here.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


def select_device():
    """Return the torch device to use (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device, criterion=None):
    """Run one pass over ``loader`` in eval mode; return mean loss and accuracy."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, device, optimizer, criterion=None):
    """Run one training epoch; return mean loss and accuracy."""
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def save_checkpoint(model, optimizer, path, meta=None):
    """Save model + optimizer state (and optional metadata) to ``path``."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": meta or {},
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load a checkpoint into ``model``/``optimizer``; returns the saved meta."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("meta", {})


def make_subset(dataset, n_per_class, seed=42):
    """Create a deterministic subset with up to ``n_per_class`` samples per class.

    Returns a ``torch.utils.data.Subset``. Requires the dataset to expose
    ``__getitem__`` returning ``(image, label)``.
    """
    if not n_per_class:
        # Return the whole dataset if n_per_class is None or zero
        return Subset(dataset, list(range(len(dataset))))
    rng = torch.Generator().manual_seed(seed)
    labels = [dataset[i][1] for i in range(len(dataset))]
    idx = []
    for cls in range(10):
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        if not cls_idx:
            continue
        cls_idx = torch.tensor(cls_idx, dtype=torch.long)
        take = min(n_per_class, len(cls_idx))
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=rng)[:take]]
        idx.extend(cls_idx.tolist())
    return Subset(dataset, sorted(idx))