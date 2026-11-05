"""Project settings: paths, image parameters and the experiments to run.

Everything that might need tuning lives here, so the rest of the code never
contains "magic numbers".
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/chest_xray")  # created by the Kaggle download in the notebook
RESULTS_DIR = Path("results")  # trained models, metrics and figures are saved here

# ImageFolder assigns labels in alphabetical order of the folder names:
# NORMAL -> 0, PNEUMONIA -> 1
CLASS_NAMES = ["Normal", "Pneumonia"]

# ---------------------------------------------------------------------------
# Image parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # input size expected by ResNet18

# Mean and standard deviation of ImageNet (per RGB channel). The pretrained
# ResNet18 learnt with images normalised this way, so ours must be too.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42  # fixed seed -> same data split and same results on every run


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """Configuration of one training run."""

    name: str
    augmentation: bool = False  # random crop, rotation and brightness/contrast changes
    class_weights: bool = False  # give more weight to the minority class (Normal) in the loss
    epochs: int = 15  # maximum number of passes over the training set
    patience: int = 4  # early stopping: epochs without improvement before stopping
    batch_size: int = 32  # images processed together before each weight update
    learning_rate: float = 1e-3  # size of each weight update
    val_fraction: float = 0.15  # share of the training set used for validation


# Each experiment adds ONE change to the previous one, so the effect of every
# change can be measured on its own.
EXPERIMENTS = {
    "baseline": Experiment("baseline"),
    "weighted": Experiment("weighted", class_weights=True),
    "weighted_aug": Experiment("weighted_aug", class_weights=True, augmentation=True),
}
