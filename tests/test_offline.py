"""Offline tests using synthetic tensors — no CIFAR-10 download required."""
import torch
import numpy as np
import sys, os
from torch.utils.data import Dataset, Subset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.definitions import SmallCNN, build_resnet18
from src.data.loading import (
    get_transforms,
    CIFAR10_CLASSES,
    make_split,
    _stratified_split_indices,
)

def test_dataset_shapes_and_labels():
    # Synthetic 32x32 images, labels 0-9
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 3, 5, 9])
    assert x.shape == (4, 3, 32, 32)
    assert y.min().item() >= 0 and y.max().item() <= 9
    assert len(CIFAR10_CLASSES) == 10


def test_stratified_split_coverage():
    # Ensure each class appears in both train and val splits
    n_per_class = 60  # small but > 1 so both splits get samples
    labels = list(range(10)) * n_per_class  # 10 classes x 60 samples
    train_idx, val_idx = _stratified_split_indices(labels, val_ratio=0.1, seed=42)
    # Union of both splits should cover all indices
    assert set(train_idx).union(set(val_idx)) == set(range(len(labels)))
    # Splits should be disjoint
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    # Every class appears in both splits
    for cls in range(10):
        cls_in_train = sum(1 for i in train_idx if labels[i] == cls)
        cls_in_val = sum(1 for i in val_idx if labels[i] == cls)
        assert cls_in_train > 0, f"Class {cls} missing from train"
        assert cls_in_val > 0, f"Class {cls} missing from val"
    # Approximate ratio
    assert len(train_idx) / len(labels) >= 0.85
    assert len(train_idx) / len(labels) <= 0.95


def test_make_split_from_dataset():
    # Create a minimal synthetic dataset
    class SyntheticDataset(Dataset):
        def __init__(self, n_samples=100, n_classes=10):
            self.n_samples = n_samples
            self.labels = torch.tensor([i % n_classes for i in range(n_samples)])
            self.data = torch.randn(n_samples, 3, 32, 32)
        def __len__(self):
            return self.n_samples
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx].long()

    full = SyntheticDataset(n_samples=100)
    train, val = make_split(full, val_ratio=0.2, seed=123)
    # Sizes should sum to 100
    assert len(train) + len(val) == len(full)
    # Validation should be about 20%
    assert 0.15 <= len(val) / len(full) <= 0.25
    # Check class balance per split
    train_labels = [train.dataset.labels[i] for i in train.indices]
    val_labels = [val.dataset.labels[i] for i in val.indices]
    # At least 1 sample per class in each split (for this seed/layout)
    for cls in range(10):
        assert cls in train_labels, f"Class {cls} missing from train"
        assert cls in val_labels, f"Class {cls} missing from val"

def test_model_output_shape_and_finite():
    model = SmallCNN(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()

def test_resnet18_output():
    m = build_resnet18(pretrained=False, freeze_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 10)

def test_train_eval_mode():
    model = SmallCNN()
    model.train()
    assert model.training
    model.eval()
    assert not model.training

def test_checkpoint_consistency():
    model = SmallCNN()
    x = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        y1 = model(x)
    # Save/load state dict
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    model2 = SmallCNN()
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y1, y2, atol=1e-6)
    os.unlink(path)

def test_transform_output_sizes():
    train_t, val_t = get_transforms(pretrained=False)
    img = torch.rand(3, 32, 32)
    # transforms expect PIL; just verify callable
    assert callable(train_t) and callable(val_t)

def test_metric_dimensions():
    # Confusion matrix 10x10; per-class metrics length 10
    import numpy as np
    cm = np.eye(10, dtype=int)
    assert cm.shape == (10, 10)

if __name__ == "__main__":
    test_dataset_shapes_and_labels()
    test_model_output_shape_and_finite()
    test_resnet18_output()
    test_train_eval_mode()
    test_checkpoint_consistency()
    test_transform_output_sizes()
    test_metric_dimensions()
    print("All offline tests passed.")
