"""Image transforms, patient-level train/validation split and data loaders."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED


def set_seed(seed=SEED):
    """Same seed -> same split, same initial weights, same results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_transform():
    """Validation and test: no randomness, only resize and normalise."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(augmentation):
    """Training transform. With augmentation, every epoch sees a slightly different
    version of each X-ray, so the network cannot rely on fixed positions (corners,
    borders, markers) and has to learn features of the lungs themselves."""
    if not augmentation:
        return eval_transform()
    return transforms.Compose([
        # keep 75-100 % of the image and change the framing a little
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        # patients are never perfectly straight
        transforms.RandomRotation(10),
        # different machines and exposures
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # NO horizontal flip: it would put the heart on the wrong side
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def patient_id(path):
    """Patient code from the file name, e.g.
    person1_bacteria_1.jpeg -> person1 ; IM-0115-0001.jpeg -> IM-0115."""
    name = Path(path).stem
    if name.startswith("person"):
        return name.split("_")[0]
    return name.rsplit("-", 1)[0]


def split_train_val(dataset, val_fraction, seed=SEED):
    """Split by PATIENT (one child never appears in both sets) and keep the same
    Normal/Pneumonia proportion in both. Returns (train_indices, val_indices)."""
    labels = np.array(dataset.targets)
    groups = [patient_id(path) for path, _ in dataset.samples]
    splitter = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True,
                                    random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return train_idx, val_idx


def class_weights(labels):
    """Inverse-frequency weights: the rarer class counts more in the loss."""
    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(exp, data_dir=DATA_DIR, num_workers=2):
    """Train / validation / test loaders. The official 'val' folder (16 images) is
    too small to be useful, so validation is taken from 'train' instead."""
    # Two views of the same training folder: with and without augmentation
    train_aug = datasets.ImageFolder(Path(data_dir) / "train", train_transform(exp.augmentation))
    train_plain = datasets.ImageFolder(Path(data_dir) / "train", eval_transform())
    test = datasets.ImageFolder(Path(data_dir) / "test", eval_transform())

    train_idx, val_idx = split_train_val(train_plain, exp.val_fraction)
    train_labels = np.array(train_plain.targets)[train_idx]

    generator = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(Subset(train_aug, train_idx), batch_size=exp.batch_size,
                            shuffle=True, num_workers=num_workers, generator=generator),
        "val": DataLoader(Subset(train_plain, val_idx), batch_size=exp.batch_size,
                          num_workers=num_workers),
        "test": DataLoader(test, batch_size=exp.batch_size, num_workers=num_workers),
    }
    print(f"train: {len(train_idx)} images {np.bincount(train_labels).tolist()} | "
          f"val: {len(val_idx)} | test: {len(test)}  (counts = [Normal, Pneumonia])")
    return loaders, train_labels
