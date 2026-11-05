"""Data pipeline: image transforms, patient-level train/validation split and loaders."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED


def set_seed(seed=SEED):
    """Fix every random number generator so results can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def eval_transform():
    """Transform for validation and test: deterministic (no randomness)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # ResNet18 expects 224 x 224
        transforms.ToTensor(),  # PIL image (0-255) -> tensor [3, 224, 224] with values 0-1
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # same scale as ImageNet
    ])


def train_transform(augmentation):
    """Transform for training.

    With data augmentation, every epoch sees a slightly different version of each
    X-ray. Corners, borders and markers no longer appear in exactly the same place,
    so the network is pushed to learn features of the lungs instead.
    The transforms are random and are applied every time an image is loaded.
    """
    if not augmentation:
        return eval_transform()
    return transforms.Compose([
        # Random crop covering 75-100 % of the image (almost square), resized to 224 x 224
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        # Small random rotation (-10 to +10 degrees): patients are never perfectly straight
        transforms.RandomRotation(10),
        # +-20 % brightness and contrast: different X-ray machines and exposures
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # No horizontal flip on purpose: it would put the heart on the wrong side
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
def patient_id(path):
    """Extract the patient code from the file name.

    Examples: person1_bacteria_1.jpeg -> person1 ; IM-0115-0001.jpeg -> IM-0115
    """
    name = Path(path).stem  # file name without extension
    if name.startswith("person"):
        return name.split("_")[0]
    return name.rsplit("-", 1)[0]


def split_train_val(dataset, val_fraction, seed=SEED):
    """Split the training images into train and validation.

    - Grouped by patient: all X-rays of one child go to the same set. Otherwise the
      model could "recognise the child" and validation would look better than it is
      (data leakage).
    - Stratified: both sets keep the same Normal / Pneumonia proportion.

    Returns the image indices of each set.
    """
    labels = np.array(dataset.targets)
    groups = [patient_id(path) for path, _ in dataset.samples]
    # n_splits = 7 for 15 %: the data is divided into 7 parts and the first is used as validation
    splitter = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True,
                                    random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return train_idx, val_idx


def class_weights(labels):
    """Inverse-frequency weights for the loss function.

    weight_c = total / (2 * images_of_class_c): the rarer class (Normal) gets a larger
    weight, so a mistake on a healthy X-ray costs more and the model stops
    favouring "pneumonia" just because it is more frequent.
    """
    counts = np.bincount(labels, minlength=2)  # [n_normal, n_pneumonia]
    weights = len(labels) / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def make_loaders(exp, data_dir=DATA_DIR, num_workers=2):
    """Create the train, validation and test DataLoaders.

    The official "val" folder of the dataset has only 16 images, which is too few
    for early stopping, so validation is taken from the "train" folder instead.
    Returns the loaders and the labels of the training images (for class weights).
    """
    # ImageFolder: one sub-folder per class; the folder name becomes the label.
    # Two views of the same training folder: with augmentation (to train) and
    # without it (to validate). The images are the same, only the transform changes.
    train_aug = datasets.ImageFolder(Path(data_dir) / "train", train_transform(exp.augmentation))
    train_plain = datasets.ImageFolder(Path(data_dir) / "train", eval_transform())
    test = datasets.ImageFolder(Path(data_dir) / "test", eval_transform())

    train_idx, val_idx = split_train_val(train_plain, exp.val_fraction)
    train_labels = np.array(train_plain.targets)[train_idx]

    generator = torch.Generator().manual_seed(SEED)  # reproducible shuffling
    loaders = {
        # shuffle=True: images in a different order every epoch, so the order is not learnt
        "train": DataLoader(Subset(train_aug, train_idx), batch_size=exp.batch_size,
                            shuffle=True, num_workers=num_workers, generator=generator),
        "val": DataLoader(Subset(train_plain, val_idx), batch_size=exp.batch_size,
                          num_workers=num_workers),
        "test": DataLoader(test, batch_size=exp.batch_size, num_workers=num_workers),
    }
    print(f"train: {len(train_idx)} images {np.bincount(train_labels).tolist()} | "
          f"val: {len(val_idx)} | test: {len(test)}  (counts = [Normal, Pneumonia])")
    return loaders, train_labels
