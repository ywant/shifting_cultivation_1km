"""
Inference entry point for the shifting cultivation classifier.

Loads the trained EfficientNet-B1, runs patch-based prediction with
test-time augmentation over NICFI PlanetScope GeoTIFF basemap tiles,
and writes class-prediction and confidence GeoTIFFs.

See model/predictor.py for full documentation of the prediction pipeline,
including two-date fusion and output raster formats.

Usage::

    python predict.py --config configs/sc_prediction.yaml

    # Predict a specific year:
    python predict.py --config configs/sc_prediction.yaml \\
                      --input_dir data/nicfi/2020-06_2020-11/ \\
                      --output_dir results/predictions/2020/

    # Use a specific checkpoint:
    python predict.py --config configs/sc_prediction.yaml \\
                      --model_path saved_models/bestF1.pkl
"""

import argparse
import os
import sys
import yaml
import logging
import torch

from model.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str) -> argparse.Namespace:
    """Load a YAML config file into a flat argparse Namespace.

    Nested YAML blocks are flattened with '__' separators.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}__{sk}"] = sv
        else:
            flat[k] = v
    return argparse.Namespace(**flat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shifting cultivation inference over NICFI GeoTIFF tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to configs/sc_prediction.yaml")
    parser.add_argument("--input_dir",  type=str, default=None,
                        help="Override input_image_dir (e.g. for a specific year).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output_dir.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model_path.")
    parser.add_argument("--gpu",        type=int, default=0,
                        help="CUDA device index.")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        log.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # CLI overrides
    if args.input_dir:
        config.input_image_dir = [args.input_dir]
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_path:
        config.model_path = args.model_path

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        log.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        torch.cuda.empty_cache()

    # Required field check
    for field in ["model_path", "input_image_dir", "output_dir",
                  "input_size", "BATCH_SIZE", "STRIDE", "output_ch"]:
        if not getattr(config, field, None):
            log.error(f"Missing required config field: '{field}'")
            sys.exit(1)

    os.makedirs(config.output_dir, exist_ok=True)

    predictor = Predictor(config)
    predictor.predict_all()
