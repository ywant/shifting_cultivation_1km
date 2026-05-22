"""
Data loader for the shifting cultivation classification task.

Loads 4-band PlanetScope patches (.npy, channel-first) and returns
(image_tensor, class_label) pairs for training, validation, test, and
prediction modes.

Expected .npy format: float32 array of shape (4, H, W)
  Channel 0 — Red
  Channel 1 — Green
  Channel 2 — Blue
  Channel 3 — NIR

Label mapping (from CSV 'label' column):
  'f'  → 0  Forest
  'sc' → 1  Shifting cultivation
  'sa' → 2  Secondary vegetation / agriculture
  'a'  → 3  Agriculture
  'm'  → 4  Mosaic
"""

import os
import glob
import random
import time

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils import data
from torchvision import transforms as T
import albumentations as A

from utils.preprocessing import equalize, compute_ndvi


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP = {"f": 0, "sc": 1, "sa": 2, "a": 3, "m": 4}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PATCH_SIZE = 234   # Spatial size (px) of each input patch


# ---------------------------------------------------------------------------
# Spectral index utilities
# ---------------------------------------------------------------------------

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI from Red and NIR arrays, imputing NaN with the band mean.

    Args:
        red: Red-band array, shape (H, W).
        nir: NIR-band array, shape (H, W).

    Returns:
        NDVI array scaled to [0, 255], shape (H, W), dtype float32.
    """
    np.seterr(divide="ignore", invalid="ignore")
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    ndvi = (nir - red) / (nir + red)
    ndvi = np.where(np.isnan(ndvi), np.nanmean(ndvi), ndvi)
    return ((ndvi + 1) * 255.0 / 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SCImageFolder(data.Dataset):
    """PyTorch Dataset for 4-band PlanetScope shifting cultivation patches.

    Args:
        root:       Path(s) to the directory containing .npy patch files.
                    Pass a list for multi-date imagery (paths are concatenated).
        config:     Config namespace with all data settings.
        mode:       One of 'train', 'valid', 'test', or 'pred'.
        split_list: List of file paths to include (used for train/valid splits).
    """

    def __init__(self, root, config, mode="train", split_list=None):
        self.root = root
        self.config = config
        self.mode = mode
        self.split_list = split_list or []

        # ------------------------------------------------------------------ #
        # Load label CSV once (not per-item) and build fast lookup dicts.    #
        # ------------------------------------------------------------------ #
        if mode != "pred":
            gt_df = pd.read_csv(config.classification_label_fn)
            self._path_to_label   = dict(zip(gt_df["path"], gt_df["label"]))
            self._path_to_augprob = dict(zip(gt_df["path"], gt_df["augProb"]))

        # ------------------------------------------------------------------ #
        # Collect file paths for the requested split / mode.                 #
        # ------------------------------------------------------------------ #
        self.image_paths = self._collect_paths()
        self.augmentation_prob = config.augmentation_prob if mode == "train" else 0

        print(f"[{mode}] {len(self.image_paths)} samples found.")

    def _collect_paths(self) -> list:
        """Return the list of .npy file paths for the current mode."""
        mode = self.mode
        config = self.config

        if mode in ("train", "valid"):
            # Gather files from both image-date directories
            files = (
                glob.glob(f"{config.train_path1}/*.npy")
                + glob.glob(f"{config.train_path2}/*.npy")
            )
            return [f for f in files if f in self.split_list]

        elif mode == "test":
            files = (
                glob.glob(f"{config.test_path1}/*.npy")
                + glob.glob(f"{config.test_path2}/*.npy")
            )
            gt_df = pd.read_csv(config.classification_label_fn)
            test_list = set(gt_df.loc[gt_df["useCase"] == "test", "path"].tolist())
            # Exclude any files that appear in the train/valid split
            split_set = set(self.split_list)
            return [f for f in files if f in test_list and f not in split_set]

        elif mode == "pred":
            if isinstance(self.root, list):
                # root is already a list of file paths (single-tile prediction)
                return self.root
            return glob.glob(f"{self.root}/*.npy")

        return []

    # ------------------------------------------------------------------
    # Loading and preprocessing
    # ------------------------------------------------------------------

    def _load_patch(self, path: str) -> np.ndarray:
        """Load a 4-band patch from disk.

        Pads to PATCH_SIZE if smaller, then crops to (4, PATCH_SIZE, PATCH_SIZE).
        Applies per-channel histogram equalisation.

        Args:
            path: Path to a .npy file with shape (4, H, W).

        Returns:
            Float32 array of shape (4, PATCH_SIZE, PATCH_SIZE).
        """
        image = np.load(path, mmap_mode="r").astype(np.float32)  # (4, H, W)
        _, h, w = image.shape

        # Zero-pad if patch is smaller than the expected size
        if h < PATCH_SIZE or w < PATCH_SIZE:
            pad_h = max(0, PATCH_SIZE - h)
            pad_w = max(0, PATCH_SIZE - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

        # Crop to fixed patch size
        image = image[:, :PATCH_SIZE, :PATCH_SIZE]

        # Histogram equalisation (per-channel)
        image = equalize(image)
        return image.astype(np.float32)

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random spatial and photometric augmentations.

        Only called during training when the per-sample augProb exceeds the
        configured threshold (i.e. augmentation is applied to high-confidence
        samples only under weakly supervised learning).

        Args:
            image: Channel-last float32 array of shape (H, W, C).

        Returns:
            Augmented channel-last array of the same shape.
        """
        transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_LINEAR),
            A.RandomBrightnessContrast(),
            A.GaussianBlur(),
        ])
        return transform(image=image)["image"]

    def _normalise(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet channel normalisation to the RGB channels.

        Args:
            image: Channel-last array (H, W, C).

        Returns:
            Normalised channel-last array.
        """
        transform = A.Compose([
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        return transform(image=image)["image"]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Load, preprocess, and return one sample.

        Returns:
            For train / valid / test modes:
                (image_tensor, label_tensor)
            For pred mode with pred_mode='test':
                (image_tensor, label_tensor, filepath)
            For pred mode (no labels):
                image_tensor
        """
        filepath = self.image_paths[index]

        # Load and equalise patch (channel-first: 4, H, W)
        image = self._load_patch(filepath)

        # Optionally append NDVI as a 5th channel
        if self.config.add_ndvi:
            ndvi = compute_ndvi(image[0], image[3])  # Red=ch0, NIR=ch3
            image = np.concatenate([image, ndvi[np.newaxis, :, :]], axis=0)

        # Convert to channel-last for albumentations
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        if self.mode != "pred":
            label_str = self._path_to_label.get(filepath)
            aug_prob   = self._path_to_augprob.get(filepath, 0.0)

            if label_str not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{label_str}' for file {filepath}. "
                    f"Expected one of {list(LABEL_MAP.keys())}."
                )
            label = torch.tensor(LABEL_MAP[label_str])

            # Augmentation: applied when sample augProb exceeds threshold
            if self.mode == "train" and aug_prob > self.augmentation_prob:
                image = self._apply_augmentation(image)

            if self.config.imageNetnorm:
                image = self._normalise(image)

            # Back to channel-first, scaled to [0, 255]
            image = np.transpose(image, (2, 0, 1)) * 255.0

            return image.astype(np.float32), label

        else:
            # Prediction mode — no label
            if self.config.imageNetnorm:
                image = self._normalise(image)
            image = np.transpose(image, (2, 0, 1)) * 255.0

            if self.config.pred_mode == "test":
                label_str = self._path_to_label.get(filepath)
                label = torch.tensor(LABEL_MAP[label_str])
                return image.astype(np.float32), label, filepath

            return image.astype(np.float32)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_loader(image_path, config, mode: str, split_list: list = None) -> data.DataLoader:
    """Create a DataLoader for the given mode and file paths.

    Args:
        image_path: Single path string or list of paths to image directories.
        config:     Config namespace.
        mode:       One of 'train', 'valid', 'test', or 'pred'.
        split_list: List of file paths to restrict loading to (train/valid).

    Returns:
        Configured PyTorch DataLoader.
    """
    dataset = SCImageFolder(
        root=image_path, config=config, mode=mode, split_list=split_list or []
    )

    shuffle = mode in ("train", "valid")
    batch_size = config.batch_size if mode != "pred" else config.BATCH_SIZE

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,  # set in config, e.g. 8–16
        pin_memory=True,
    )
