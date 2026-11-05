"""All project settings in one place."""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data/chest_xray")  # created by the Kaggle download in the notebook
RESULTS_DIR = Path("results")
CLASS_NAMES = ["Normal", "Pneumonia"]  # ImageFolder order: NORMAL = 0, PNEUMONIA = 1

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


@dataclass
class Experiment:
    name: str
    augmentation: bool = False  # random crops, small rotations, brightness/contrast
    class_weights: bool = False  # give more weight to the minority class (Normal)
    epochs: int = 15
    patience: int = 4  # early stopping
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_fraction: float = 0.15  # part of train used for validation


# Each experiment changes ONE thing, so the effect of each change can be measured.
EXPERIMENTS = {
    "baseline": Experiment("baseline"),
    "weighted": Experiment("weighted", class_weights=True),
    "weighted_aug": Experiment("weighted_aug", class_weights=True, augmentation=True),
}
