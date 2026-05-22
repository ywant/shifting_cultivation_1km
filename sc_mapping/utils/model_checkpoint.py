"""
Lightweight model checkpoint manager.

Saves the best model weights observed during training, tracked separately
for each metric (e.g. best validation loss, best F1 score).  Periodic
epoch snapshots are also supported.

This is a minimal reimplementation of the checkpoint logic needed for the
shifting cultivation classifier.  It has no dependencies beyond PyTorch
and the standard library.
"""

import os
import copy
import logging
import torch

log = logging.getLogger(__name__)


class ModelCheckpoint:
    """Track and save the best model weights for one or more metrics.

    One ``.pt`` file is written per metric, named ``best_{metric_name}.pt``.
    The checkpoint also records which epoch each best value was achieved.

    Args:
        checkpoint_dir:  Directory where checkpoint files are written.
        model_name:      Identifier string used in log messages.
        selection_stage: Stage whose metrics are used for model selection
                         (typically ``'val'``; ``'train'`` is ignored).
        run_config:      Arbitrary config object stored alongside weights
                         for reference (optional).
        resume:          Unused; kept for API compatibility.
    """

    # Metrics where *higher* is better.  All others use *lower is better*.
    _HIGHER_IS_BETTER = {"r2", "f1", "acc", "accuracy", "sens", "sensitivity"}

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        selection_stage: str,
        run_config=None,
        resume: bool = False,
    ):
        self.checkpoint_dir   = checkpoint_dir
        self.model_name       = model_name
        self.selection_stage  = selection_stage
        self.run_config       = run_config

        os.makedirs(checkpoint_dir, exist_ok=True)

        # best_{metric_name} → best scalar value seen so far
        self._best_values: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        """Path of the most recently written checkpoint file."""
        return self._last_saved_path if hasattr(self, "_last_saved_path") else ""

    def save_best_models_under_current_metrics(
        self, model: torch.nn.Module, metrics_holder: dict
    ):
        """Save model weights whenever a metric improves.

        Args:
            model:          The model whose ``state_dict`` is saved.
            metrics_holder: Dict with keys ``'stage'``, ``'epoch'``, and
                            ``'current_metrics'`` (a dict of metric_name → value).
        """
        stage   = metrics_holder["stage"]
        epoch   = metrics_holder["epoch"]
        metrics = metrics_holder["current_metrics"]

        # Only use the designated selection stage for saving
        if stage != self.selection_stage:
            return

        state_dict = copy.deepcopy(model.state_dict())

        for metric_name, current_value in metrics.items():
            higher_is_better = any(
                token in metric_name.lower()
                for token in self._HIGHER_IS_BETTER
            )
            prev_best = self._best_values.get(metric_name)

            improved = (
                prev_best is None
                or (higher_is_better and current_value > prev_best)
                or (not higher_is_better and current_value < prev_best)
            )

            if improved:
                self._best_values[metric_name] = current_value
                save_path = self._metric_path(metric_name)
                torch.save(
                    {
                        "epoch":       epoch,
                        "state_dict":  state_dict,
                        "metric_name": metric_name,
                        "metric_value": current_value,
                        "run_config":  self.run_config,
                    },
                    save_path,
                )
                self._last_saved_path = save_path
                log.info(
                    f"  ✓ [{metric_name}] {prev_best} → {current_value:.4f} "
                    f"(epoch {epoch}) saved to {save_path}"
                )

    def load_best(
        self,
        model: torch.nn.Module,
        metric_name: str,
        device: torch.device = None,
    ) -> int:
        """Load the best weights for a given metric into the model.

        Args:
            model:       Model to load weights into.
            metric_name: Metric identifier (e.g. ``'val_f1'``).
            device:      Device to map tensors onto (defaults to CPU).

        Returns:
            Epoch at which the loaded checkpoint was saved.

        Raises:
            FileNotFoundError: If no checkpoint exists for metric_name.
        """
        path = self._metric_path(metric_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No checkpoint found for metric '{metric_name}' at {path}."
            )
        ckpt = torch.load(path, map_location=device or "cpu")
        model.load_state_dict(ckpt["state_dict"])
        log.info(
            f"Loaded checkpoint for '{metric_name}' "
            f"(value={ckpt['metric_value']:.4f}, epoch={ckpt['epoch']}) "
            f"from {path}"
        )
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metric_path(self, metric_name: str) -> str:
        """Return the file path for a given metric's checkpoint."""
        safe_name = metric_name.replace("/", "_")
        return os.path.join(self.checkpoint_dir, f"best_{safe_name}.pt")
