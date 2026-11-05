"""Grad-CAM on ALL false negatives + a number that summarises where the model looks.

Grad-CAM in short: the last convolutional layer produces 512 feature maps (7 x 7).
The gradient of the "pneumonia" score with respect to each map tells how much that
map pushes the decision towards pneumonia. A weighted sum of the maps gives a heat
map that is resized to 224 x 224 and drawn over the X-ray (red = most influential).

Usage:  python -m src.gradcam --experiment weighted_aug
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from .config import EXPERIMENTS, IMAGENET_MEAN, IMAGENET_STD, RESULTS_DIR  # noqa: E402
from .dataset import make_loaders  # noqa: E402
from .evaluate import load_model  # noqa: E402

MARGIN = 0.15  # band of 15 % on each side treated as "border" (outside the lungs)


def central_fraction(cam):
    """Share of the heat map inside the central region of the image.

    The central region (15 % margin removed on each side) covers 0.7 x 0.7 = 49 %
    of the image, so evenly spread attention gives ~0.49. Higher values = attention
    on the centre (lungs); lower values = attention on borders and corners.
    """
    h, w = cam.shape
    top, left = int(h * MARGIN), int(w * MARGIN)
    centre = cam[top:h - top, left:w - left].sum()
    return float(centre / (cam.sum() + 1e-8))  # 1e-8 avoids dividing by zero


def to_rgb(tensor):
    """Undo the ImageNet normalisation so the X-ray can be displayed."""
    img = tensor.numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    return np.clip(np.array(IMAGENET_STD) * img + np.array(IMAGENET_MEAN), 0, 1)


def run(name, device, max_images=12):
    """Compute Grad-CAM for every pneumonia X-ray of the test set, save a summary
    and a figure with the false negatives (sick children predicted as healthy)."""
    model = load_model(name, device)
    # Grad-CAM needs gradients to flow through the backbone, so the layers frozen
    # during training are unfrozen here (nothing is trained, gradients are only read)
    for param in model.parameters():
        param.requires_grad = True
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])  # last convolutional block
    loaders, _ = make_loaders(EXPERIMENTS[name])

    false_negatives, fractions_fn, fractions_all = [], [], []
    for images, labels in loaders["test"]:
        images = images.to(device)
        with torch.no_grad():
            preds = model(images).argmax(1).cpu()
        sick = labels == 1  # only X-rays that really have pneumonia
        if not sick.any():
            continue
        # Heat map of the "pneumonia" class (target 1) for each sick X-ray of the batch
        maps = cam(input_tensor=images[sick], targets=[ClassifierOutputTarget(1)] * int(sick.sum()))
        for img, pred, heat in zip(images[sick].cpu(), preds[sick], maps):
            fractions_all.append(central_fraction(heat))
            if pred == 0:  # false negative: sick child predicted as healthy
                fractions_fn.append(central_fraction(heat))
                false_negatives.append((img, heat))

    summary = {
        "false_negatives": len(false_negatives),
        "central_attention_all_pneumonia": float(np.mean(fractions_all)),
        "central_attention_false_negatives": float(np.mean(fractions_fn)) if fractions_fn else None,
    }
    out = RESULTS_DIR / name
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Grid with up to 12 false negatives and their central-attention value
    shown = false_negatives[:max_images]
    if shown:
        cols = 4
        rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
        for ax, (img, heat) in zip(axes.flat, shown):
            ax.imshow(show_cam_on_image(to_rgb(img).astype(np.float32), heat, use_rgb=True))
            ax.set_title(f"centre: {central_fraction(heat):.0%}", fontsize=10)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"Grad-CAM of the false negatives ({name}): {len(false_negatives)} cases",
                     fontweight="bold")
        fig.savefig(out / "gradcam_false_negatives.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args.experiment, device)
