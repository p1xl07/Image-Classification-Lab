# Image Classification Lab — CIFAR-10

Reproducible deep-learning project demonstrating a complete ML workflow on CIFAR-10.

## Problem statement
Train and compare image classifiers on CIFAR-10 (10 classes, 32×32 RGB images),
moving from a baseline CNN trained from scratch to transfer learning with
ImageNet-pretrained ResNet-18. The project emphasises reproducibility,
leakage-free model selection, and honest reporting of measured results.

## Dataset & split policy
- Source: `torchvision.datasets.CIFAR10`, downloaded into `data/` (excluded from Git).
- Official **50,000**-image training partition and official **10,000**-image test partition.
- The train/validation split is **stratified** and derived *exclusively* from the
  50,000-image training partition (`src/data/loading.py::make_split`, dependency-free
  NumPy implementation, fixed seed).
- The official test partition is **never** used for training, validation, early
  stopping, hyperparameter tuning, or checkpoint selection. It is reserved for the
  final evaluation after all model choices are frozen.
- Train augmentation (RandomCrop with padding=4, RandomHorizontalFlip) is applied
  only to training data. Validation/test transforms are deterministic.

## Environment
- Windows 11, Python 3.12.13 in `.venv-gpu` (Windows PowerShell)
- torch 2.14.1+cu132, torchvision 0.29.1+cu132
- NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM); CUDA available
- CPU fallback supported via `src/training/train.py::select_device`

## Project structure (relative paths)
- `src/data/loading.py` — CIFAR-10 loading, stratified split, augmentation
- `src/models/definitions.py` — `SmallCNN` (Exp A), `build_resnet18` (Exp B/C)
- `src/training/train.py` — training/eval loops, checkpoint helpers, subset builder
- `src/training/smoke_test.py` — one-epoch GPU smoke test
- `src/experiments/train_cnn.py` — Experiment A entry point
- `src/evaluation/` — evaluation module (metrics, confusion matrix) — *pending*
- `src/experiments/` — Experiment B/C entry points — *not yet implemented*
- `src/inference/` — Streamlit inference demo (`app.py`) for selected C checkpoint
- `tests/` — offline unit tests (synthetic tensors; no download)
- `data/`, `checkpoints/`, `*.csv` — excluded from Git

## Installation
```powershell
# Activate the GPU environment
.\.venv-gpu\Scripts\Activate.ps1

# Or call the interpreter directly
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the tests (offline — no dataset download)
```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/ -v
```
Current result: **15 passed** (9 pre-existing + 6 added for Experiment A).

## Reproducing Experiment A (CNN from scratch)

```powershell
.\.venv-gpu\Scripts\python.exe -m src.experiments.train_cnn `
    --epochs 5 `
    --batch-size 128 `
    --lr 1e-3 `
    --weight-decay 5e-4 `
    --seed 42 `
    --subset-per-class 50 `
    --val-ratio 0.1 `
    --patience 3 `
    --workers 2 `
    --checkpoint-dir checkpoints `
    --csv cifar_cnn_dev_history.csv
```

### Configuration options
| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | 10 | Maximum epochs (fixed limit when `--patience` is omitted) |
| `--batch-size` | 128 | Train/validation batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--weight-decay` | 5e-4 | Adam weight decay |
| `--seed` | 42 | Global random seed |
| `--subset-per-class` | 500 | Images per class for development runs (`0` = full training set) |
| `--val-ratio` | 0.1 | Validation fraction of the 50,000-image training partition |
| `--patience` | None | Early-stopping patience in epochs; omit for a fixed epoch count |
| `--workers` | 2 | DataLoader worker processes |
| `--checkpoint-dir` | checkpoints | Where `best_val_loss.pt` is written |
| `--csv` | training_history.csv | CSV history output path |

### Outputs
- `checkpoints/best_val_loss.pt` — best model by **validation** loss, with model
  state, optimizer state, and metadata (epoch, train/val loss, device).
- `<csv>` — configuration block plus per-epoch `train_loss`, `train_acc`,
  `val_loss`, `val_acc`, `elapsed_sec`.

## Development run results (NOT final benchmark performance)

Command: 5 epochs, `--subset-per-class 50`, batch size 128, Adam lr=1e-3,
weight decay 5e-4, seed 42, patience 3, on CUDA.

Subset sizes: train 500 / validation 500 (stratified from the 50,000-image
official training partition).

| Epoch | Train loss | Train acc | Val loss | Val acc | Elapsed (s) |
|-------|-----------|-----------|----------|---------|-------------|
| 1 | 2.3235 | 0.1520 | 2.2982 | 0.1120 | 38.39 |
| 2 | 1.8785 | 0.3280 | 2.2704 | 0.1480 | 35.44 |
| 3 | 1.6084 | 0.4480 | 2.2430 | 0.1660 | 34.79 |
| 4 | 1.4246 | 0.4700 | 2.1973 | 0.1840 | 33.88 |
| 5 | 1.2308 | 0.5680 | 2.0404 | 0.2560 | 32.70 |

- Total training time: **175.36 s** (5 epochs)
- Best epoch by validation loss: **epoch 5** (val loss 2.0404)
- Checkpoint: `checkpoints/best_val_loss.pt` (8,043,295 bytes) — reloaded
  successfully; inference on a validation sample produced a finite
  distribution (confidence 0.2107).
- No test-set evaluation was performed.

**These are development results on a 500-image subset, not CIFAR-10 benchmark
accuracy.** They are reported to verify the pipeline works end to end; no
accuracy claim is made for the full dataset.

## Status
- ✅ Environment verified, dataset pipeline implemented and validated
- ✅ Experiment A (CNN from scratch) entry point implemented + development run
- ⬜ Experiment B (frozen pretrained ResNet-18) — not implemented
- ⬜ Experiment C (fine-tuned ResNet-18) — not implemented
- ⬜ Evaluation module (metrics, confusion matrix, curves) — not implemented
- ⬜ Final test evaluation — deferred until model selection is frozen
- ✅ Streamlit inference demo implemented (`src/inference/app.py`)

## Inference Demo (Streamlit)

```powershell
.\.venv-gpu\Scripts\python.exe -m streamlit run src\inferencepp.py
```

The demo loads the selected Experiment C checkpoint (`checkpoints_full_c/best_val_loss.pt`) and predicts among the 10 CIFAR-10 classes. It uses ImageNet preprocessing (Resize 224, ImageNet normalization) and runs in evaluation mode. The checkpoint is selected by validation loss (epoch 2, val_loss=0.2681, val_acc=0.9098); the final test evaluation (90.39% accuracy, macro F1 0.9038) was performed once after selection and did not alter the model or checkpoint.

## Verified Full Results

| Experiment | Best validation accuracy | Epoch | Val loss |
|---|---:|---:|---:|
| A — CNN scratch | 78.84% | 5 | 0.6162 |
| B — Frozen ResNet-18 | 81.08% | 5 | 0.5710 |
| C — Fine-tuned ResNet-18 | 90.98% | 2 | 0.2681 |

Selected: Experiment C (lowest validation loss, epoch 2). Final official test evaluation (10,000 images, one-time): accuracy 90.39%, macro F1 0.9038. Results in `final_test_results_c.json`.

## Key constraints respected
- Only relative paths used; `.git` preserved; no commits or pushes
- Official test set never used for model/hyperparameter selection
- Pretrained ImageNet weights noted for ResNet-18 (`IMAGENET1K_V1`)
- No large hyperparameter-search framework
- No fabricated scores, charts, or screenshots
