"""Streamlit inference demo for Experiment C (Fine-tuned ResNet-18).

Loads the selected checkpoint from a configurable relative path, reconstructs
the fine-tuned architecture (pretrained ResNet-18 with layer4 + fc trainable),
and runs deterministic inference with ImageNet preprocessing (resize 224,
ImageNet normalization). Supports image upload; displays the image, predicted
CIFAR-10 class, and class probabilities sorted by confidence.

Does not train, download datasets, or use test data.
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image

from src.data.loading import CIFAR10_CLASSES, get_transforms
from src.models.definitions import build_resnet18

# ---- Config (relative, repository-root independent) ----
CHECKPOINT_PATH = "checkpoints_full_c/best_val_loss.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load once per session ----
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not path.exists() if hasattr(path, "exists") else __import__("os").path.exists(path):
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}. Expected at repo root or configure path.")
    model = build_resnet18(num_classes=10, pretrained=True, freeze_backbone=False)
    for n, p_ in model.named_parameters():
        p_.requires_grad = ("layer4" in n or "fc" in n)
    state = torch.load(str(path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model


def preprocess(image: Image.Image):
    # Same as Experiment C evaluation: resize 224, ImageNet norm
    train_t, val_t = get_transforms(pretrained=True)
    # Use val_t (deterministic) for inference
    img_t = val_t(image.convert("RGB"))
    return img_t.unsqueeze(0)  # (1, 3, 224, 224)


def main():
    st.set_page_config(page_title="CIFAR-10 — Experiment C Inference", layout="centered")
    st.title("CIFAR-10: Experiment C — Fine-tuned ResNet-18")
    st.markdown(
        "*Selected by lowest validation loss (epoch 2, val_loss=0.2681, val_acc=0.9098). "
        "This demo uses the frozen backbone + fine-tuned layer4/classifier checkpoint. "
        "It predicts among the 10 CIFAR-10 classes only.*"
    )

    # Load model
    try:
        model = load_model(CHECKPOINT_PATH)
    except Exception as e:
        st.error(f"Failed to load checkpoint `{CHECKPOINT_PATH}`. {e}")
        st.info("Place `checkpoints_full_c/best_val_loss.pt` at the repo root, or edit the CHECKPOINT_PATH in `src/inference/app.py`.")
        st.stop()

    uploaded = st.file_uploader("Upload a CIFAR-10-style image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        try:
            image = Image.open(uploaded)
        except Exception as e:
            st.error(f"Invalid image: {e}")
            st.stop()

        st.image(image, caption="Uploaded image (original)", use_container_width=True)
        tensor = preprocess(image).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(probs.argmax())
        pred_name = CIFAR10_CLASSES[pred_class]

        st.subheader(f"Prediction: **{pred_name}** (confidence {probs[pred_class]:.3f})")

        # Show sorted probabilities
        sorted_idx = np.argsort(-probs)
        st.write("Class probabilities (sorted):")
        for idx in sorted_idx:
            bar = st.progress(float(probs[idx]), text=f"{CIFAR10_CLASSES[idx]} — {probs[idx]:.4f}")

        st.markdown("---")
        st.caption(
            "Limitations: this model was trained on 32×32 CIFAR-10 images with ImageNet "
            "preprocessing. It predicts only among 10 CIFAR-10 categories. "
            "Real-world images outside these classes may yield misleading confidence. "
            "The checkpoint was selected by validation loss (not by these test results)."
        )

    else:
        st.info("Upload an image to see predictions. The model (Experiment C, fine-tuned ResNet-18) is already loaded.")
        # Show which checkpoint is loaded
        st.markdown(f"**Loaded checkpoint:** `{CHECKPOINT_PATH}` (Experiment C — best validation epoch 2, val_loss 0.2681)")

if __name__ == "__main__":
    main()
