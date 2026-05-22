"""
Optional Weights & Biases logging wrapper.

W&B logging is entirely optional — the training pipeline runs normally
when W&B is not installed or when ``config.use_wandb`` is set to ``0``.
Readers reproducing results from the paper do not need a W&B account.

To enable logging::

    pip install wandb
    wandb login

Then set ``use_wandb: 1`` in your config file.
"""

import logging

log = logging.getLogger(__name__)


class Wandb:
    """Thin wrapper around the wandb API with graceful fallback.

    All methods are no-ops when wandb is unavailable or disabled.
    """

    _enabled: bool = False

    @classmethod
    def launch(cls, config, enabled: int = 1):
        """Initialise a W&B run if wandb is installed and enabled.

        Args:
            config:  Config namespace.  Used to populate the W&B run config.
            enabled: 1 to enable logging, 0 to skip.
        """
        if not enabled:
            log.info("W&B logging disabled (use_wandb=0).")
            return

        try:
            import wandb  # noqa: F401
        except ImportError:
            log.warning(
                "wandb not installed — skipping W&B logging. "
                "Install with: pip install wandb"
            )
            return

        import wandb

        project = getattr(config, "wandb__project", "shifting-cultivation")
        name    = getattr(config, "wandb__name",    getattr(config, "model_type", None))

        wandb.init(
            project=project,
            name=name,
            config=vars(config),
            reinit=True,
        )
        cls._enabled = True
        log.info(f"W&B run initialised: project='{project}', name='{name}'")

    @classmethod
    def add_file(cls, filepath: str):
        """Save a file artifact to the current W&B run.

        Args:
            filepath: Local path of the file to upload.
        """
        if not cls._enabled or not filepath:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.save(filepath)
        except Exception as exc:
            log.warning(f"W&B file upload failed ({filepath}): {exc}")
