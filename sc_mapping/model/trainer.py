"""
Training entry point for the shifting cultivation classifier.

Loads data, initialises the Solver, and runs training or evaluation
depending on config.mode ('train' or 'test').

Usage::
    python trainer.py --config configs/sc_classification.yaml
"""

import os
import time
import json
import glob
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.solver import Solver
from data.data_loader import get_loader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, random, and PyTorch."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_valid_split(config) -> tuple[list, list]:
    """Split labelled samples into training and validation sets.

    Splitting is done at the *site* level (by 'id' column) to prevent
    spatial leakage between splits.

    Args:
        config: Namespace containing classification_label_fn, train_path1,
                train_path2, split_ratio, and (optionally) split_save_dir.

    Returns:
        train_list: List of file paths for training.
        valid_list: List of file paths for validation.
    """
    gt_df = pd.read_csv(config.classification_label_fn)
    train_subset = gt_df[gt_df["useCase"] == "train"]

    all_ids = train_subset["id"].unique()
    train_ids, valid_ids = train_test_split(
        all_ids, test_size=1 - config.split_ratio, random_state=42
    )

    train_df = train_subset[train_subset["id"].isin(train_ids)]
    valid_df = train_subset[train_subset["id"].isin(valid_ids)]

    # Sanity check: no site-level leakage
    assert set(train_df["id"]).isdisjoint(set(valid_df["id"])), (
        "ID overlap between train and validation sets — check your split logic."
    )

    train_list = train_df["path"].tolist()
    valid_list = valid_df["path"].tolist()

    print(
        f"Train: {len(train_list)} samples ({len(train_ids)} sites) | "
        f"Valid: {len(valid_list)} samples ({len(valid_ids)} sites)"
    )

    # Optionally persist the split for reproducibility
    if hasattr(config, "split_save_dir") and config.split_save_dir:
        os.makedirs(config.split_save_dir, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M")
        split_path = os.path.join(
            config.split_save_dir,
            f"split_{timestr}_ratio{int(config.split_ratio * 10)}.json",
        )
        with open(split_path, "w") as f:
            json.dump(
                {
                    "training": train_list,
                    "validation": valid_list,
                    "num_train": len(train_list),
                    "num_valid": len(valid_list),
                    "train_ratio": config.split_ratio,
                },
                f,
            )
        print(f"Split saved to {split_path}")

    return train_list, valid_list


def get_sample_shape(loader, config) -> tuple:
    """Retrieve the shape of a single sample from a DataLoader.

    Args:
        loader: Any DataLoader (train, valid, or test).
        config: Config namespace (used to determine batch layout).

    Returns:
        Tuple representing the image tensor shape, e.g. (B, C, H, W).
    """
    dataiter = iter(loader)
    if config.conf_score:
        images, _, _ = next(dataiter)
    else:
        images, _ = next(dataiter)
    return images.shape


def Trainer(config, launch_wandb=True):
    """Build data loaders, initialise the Solver, and run training or testing.

    Args:
        config:       Argparse / config namespace with all hyperparameters.
        launch_wandb: Whether to start a W&B run (default True).
    """
    set_seed(42)
    print(config)
    start = time.time()

    # ------------------------------------------------------------------ #
    # Build train / valid split                                            #
    # ------------------------------------------------------------------ #
    if config.mergeTrainValid:
        train_list, valid_list = make_train_valid_split(config)
    else:
        train_list = valid_list = []

    # ------------------------------------------------------------------ #
    # Data loaders                                                         #
    # ------------------------------------------------------------------ #
    train_path = [config.train_path1, config.train_path2]
    val_path   = [config.valid_path1, config.valid_path2]
    test_path  = [config.test_path1,  config.test_path2]

    train_loader = get_loader(
        image_path=train_path, config=config,
        split_list=train_list, mode="train",
    )
    valid_loader = get_loader(
        image_path=val_path, config=config,
        split_list=valid_list, mode="valid",
    )
    test_loader = get_loader(
        image_path=test_path, config=config,
        split_list=train_list + valid_list, mode="test",
    )

    # ------------------------------------------------------------------ #
    # Infer patch size from a real batch                                   #
    # ------------------------------------------------------------------ #
    ref_loader = train_loader if config.mode == "train" else test_loader
    patch_size = get_sample_shape(ref_loader, config)

    # ------------------------------------------------------------------ #
    # Solver                                                               #
    # ------------------------------------------------------------------ #
    solver = Solver(config, train_loader, valid_loader, test_loader,
                    patch_size, launch_wandb=launch_wandb)

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()
    else:
        raise ValueError(f"Unknown mode: {config.mode}. Expected 'train' or 'test'.")

    print(f"Total runtime: {time.time() - start:.1f}s")
