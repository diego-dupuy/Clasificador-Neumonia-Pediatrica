"""Train one experiment with early stopping and save the best model.

Usage:  python -m src.train --experiment weighted_aug
"""

import argparse
import copy
import json
import time

import torch
import torch.nn as nn

from .config import EXPERIMENTS, RESULTS_DIR
from .dataset import class_weights, make_loaders, set_seed
from .model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One full pass over a loader.

    With an optimizer the model learns (training); without it the model is only
    evaluated (validation). Returns the mean loss and the accuracy.
    """
    training = optimizer is not None
    model.train(training)  # some layers (e.g. BatchNorm) behave differently in each mode
    total_loss, correct, total = 0.0, 0, 0

    # Gradients are only needed when training; skipping them saves memory and time
    with torch.set_grad_enabled(training):
        for images, labels in loader:  # one batch (32 images) at a time
            images, labels = images.to(device), labels.to(device)  # move to GPU if available
            outputs = model(images)  # 1. forward pass: one score per class
            loss = criterion(outputs, labels)  # 2. how wrong the predictions are

            if training:
                optimizer.zero_grad()  # 3. clear the gradients of the previous batch
                loss.backward()  # 4. backpropagation: how each weight should change
                optimizer.step()  # 5. update the weights

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()  # class with highest score
            total += labels.size(0)
    return total_loss / total, correct / total


def train(exp, device, data_dir=None, pretrained=True, num_workers=2):
    """Train the model of one experiment and save the weights with the best validation loss."""
    set_seed()
    kwargs = {"num_workers": num_workers} | ({"data_dir": data_dir} if data_dir else {})
    loaders, train_labels = make_loaders(exp, **kwargs)
    model = build_model(pretrained).to(device)

    # Loss function. CrossEntropyLoss turns the two output scores into probabilities
    # and penalises wrong predictions (more when the model is confidently wrong).
    weights = class_weights(train_labels).to(device) if exp.class_weights else None
    if weights is not None:
        print(f"class weights [Normal, Pneumonia]: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Adam updates the weights from the gradients. Only the new last layer is trained.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=exp.learning_rate)

    out_dir = RESULTS_DIR / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []

    for epoch in range(1, exp.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d} [{time.time() - start:.0f}s]  "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Early stopping: keep a copy of the weights whenever validation improves;
        # stop if it has not improved for `patience` epochs (the model would start
        # memorising the training images instead of learning general features).
        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= exp.patience:
                print(f"early stopping: no improvement for {exp.patience} epochs")
                break

    # Restore and save the BEST weights (not the last ones) + the training history
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pth")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val loss {best_loss:.4f} -> saved {out_dir / 'model.pth'}")
    return model


if __name__ == "__main__":
    # Command-line arguments, e.g.  python -m src.train --experiment weighted
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--no-pretrained", action="store_true", help="only for quick tests")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available
    print(f"device: {device} | experiment: {args.experiment}")
    train(EXPERIMENTS[args.experiment], device, pretrained=not args.no_pretrained)
