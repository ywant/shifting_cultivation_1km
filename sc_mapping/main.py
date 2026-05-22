"""
Main entry point for the shifting cultivation classifier.

All hyperparameters are set in the YAML config file for full reproducibility.
Command-line arguments can override individual values.

Example — training::

    python main.py --config configs/sc_classification.yaml

Example — test set evaluation::

    python main.py --config configs/sc_classification.yaml --mode test \\
                   --model_path saved_models/bestF1.pkl

Example — override a single value::

    python main.py --config configs/sc_classification.yaml --lr 0.0001
"""

import argparse
import os
import sys
import yaml
from torch.backends import cudnn

from model.trainer import Trainer


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file and return it as a flat dict.

    Nested blocks (e.g. the 'wandb' block) are flattened with a double-
    underscore separator so they become ordinary namespace attributes, e.g.
    ``wandb__project``.  The special 'wandb.log' key is also exposed as the
    top-level ``use_wandb`` flag consumed by the Trainer.

    Args:
        path: Path to the YAML file.

    Returns:
        Flat dict of config key-value pairs.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}__{sub_key}"] = sub_value
        else:
            flat[key] = value

    # Expose wandb.log as the simple use_wandb flag expected by Trainer
    if "wandb__log" in flat:
        flat.setdefault("use_wandb", flat["wandb__log"])

    return flat


def merge_config(args: argparse.Namespace, yaml_cfg: dict) -> argparse.Namespace:
    """Merge YAML config into an argparse namespace.

    YAML values fill in any attribute that is still None or missing; explicit
    command-line arguments are never overwritten.

    Args:
        args:     Parsed argparse namespace.
        yaml_cfg: Flat dict from load_yaml_config().

    Returns:
        Updated argparse namespace.
    """
    for key, value in yaml_cfg.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def validate_config(config: argparse.Namespace):
    """Verify that required fields are present and internally consistent.

    Raises:
        SystemExit: On missing fields or conflicting settings.
    """
    required = [
        "model_path", "result_path", "checkpoint_dir",
        "train_path1", "train_path2",
        "valid_path1", "valid_path2",
        "test_path1",  "test_path2",
        "classification_label_fn",
    ]
    missing = [k for k in required if not getattr(config, k, None)]
    if missing:
        print(f"[ERROR] Missing required config fields: {missing}")
        print("        Set them in your YAML config or via CLI arguments.")
        sys.exit(1)

    if config.mode not in ("train", "test"):
        print(f"[ERROR] --mode must be 'train' or 'test', got '{config.mode}'.")
        sys.exit(1)

    if config.mode == "test" and not os.path.isfile(config.model_path):
        print(
            f"[ERROR] --mode test requires a model checkpoint file at "
            f"--model_path.\n        Got: {config.model_path}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: argparse.Namespace):
    # Deterministic cuDNN for reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True

    os.makedirs(config.model_path,     exist_ok=True)
    os.makedirs(config.result_path,    exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Shifting Cultivation Classifier — EfficientNet-B1")
    print("=" * 60)
    for k, v in sorted(vars(config).items()):
        print(f"  {k:<35s}: {v}")
    print("=" * 60)

    Trainer(config, launch_wandb=getattr(config, "use_wandb", False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / evaluate the shifting cultivation EfficientNet-B1 classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Config file (primary way to set parameters)                         #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. CLI arguments override YAML values.",
    )

    # ------------------------------------------------------------------ #
    # Commonly overridden at the command line                             #
    # Everything else should live in the YAML file.                       #
    # ------------------------------------------------------------------ #
    parser.add_argument("--mode", type=str, default=None,
                        choices=["train", "test"])
    parser.add_argument("--model_path",     type=str, default=None,
                        help="Checkpoint directory (train) or .pkl file (test).")
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--num_epochs",     type=int,   default=None)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--loss_func",      type=str,   default=None)
    parser.add_argument("--use_wandb",      type=int,   default=None,
                        help="1 = log to W&B, 0 = skip.")

    # All other parameters are read from the YAML config.
    # Add individual overrides here as needed.

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load and merge YAML config                                           #
    # ------------------------------------------------------------------ #
    if args.config:
        if not os.path.isfile(args.config):
            print(f"[ERROR] Config file not found: {args.config}")
            sys.exit(1)
        yaml_cfg = load_yaml_config(args.config)
        args = merge_config(args, yaml_cfg)

    # Parse sample_counts from comma-separated string if needed
    if isinstance(getattr(args, "sample_counts", None), str):
        args.sample_counts = [int(x) for x in args.sample_counts.split(",")]

    # Set safe defaults for optional flags not present in older configs
    for flag, default in [
        ("add_input2", 0), ("add_outputs", 0), ("test_vis", 0),
        ("notree", 0), ("pred_mode", "pred"), ("earlystop", 0),
        ("patience", 50), ("conf_score", 0), ("norm_wei", 0),
        ("saveImages", 0), ("imageCallbackDir", "./visionCallbacks"),
        ("image_callback_freq", 10), ("model_suf", ""),
        ("split_save_dir", None), ("num_workers", 8),
        ("mergeTrainValid", 1), ("split_ratio", 0.8),
    ]:
        if not hasattr(args, flag) or getattr(args, flag) is None:
            setattr(args, flag, default)

    validate_config(args)
    main(args)
