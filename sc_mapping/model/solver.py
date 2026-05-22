"""
Solver for shifting cultivation land-use classification.

Trains, validates, and tests an EfficientNet-B1 classifier that maps
PlanetScope satellite patches to one of five land-use classes:
    0 - Forest (f)
    1 - Shifting cultivation (sc)
    2 - Secondary vegetation / agriculture (sa)
    3 - Agriculture (a)
    4 - Mosaic (m)

Supports two training modes:
    'classic' : standard supervised training
    'wsl'     : weakly supervised learning with per-sample confidence weights
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary
import torchvision.models as models
import torchvision.models.efficientnet as efficientnet
import wandb
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score
)

from utils.wandb_utils import Wandb
from utils.model_checkpoint import ModelCheckpoint

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute focal loss between logits and ground-truth labels.

    Focal loss down-weights easy examples so training focuses on hard ones.
    Reference: Lin et al., 2017 (https://arxiv.org/abs/1708.02002).

    Args:
        logits: Float tensor of shape [batch, num_classes].
        labels: Float tensor of shape [batch, num_classes].
        alpha:  Optional float tensor of shape [batch] for per-sample weighting.
        gamma:  Focusing parameter (default 2).

    Returns:
        Scalar focal loss value.
    """
    bc_loss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )
    modulator = (
        1.0 if gamma == 0.0
        else torch.exp(
            -gamma * labels * logits
            - gamma * torch.log(1 + torch.exp(-logits))
        )
    )
    loss = modulator * bc_loss
    if alpha is not None:
        loss = alpha * loss
    return torch.sum(loss) / torch.sum(labels)


class Loss(nn.Module):
    """Class-balanced loss wrapper.

    Supports focal loss, cross-entropy, binary cross-entropy, and
    softmax binary cross-entropy, optionally with class-balanced weighting.

    Reference: Cui et al., CVPR 2019.

    Args:
        loss_type:         One of 'focal_loss', 'cross_entropy',
                           'binary_cross_entropy', 'softmax_binary_cross_entropy'.
        beta:              Class-balance hyperparameter (default 0.999).
        fl_gamma:          Focal loss focusing parameter (default 2).
        samples_per_class: List of sample counts per class. Required when
                           class_balanced=True.
        class_balanced:    Whether to apply class-balanced weighting.
    """

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma: float = 2,
        samples_per_class=None,
        class_balanced: bool = False,
    ):
        super().__init__()
        if class_balanced and samples_per_class is None:
            raise ValueError(
                "samples_per_class must be provided when class_balanced=True."
            )
        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced loss.

        Args:
            logits: Float tensor of shape [batch, num_classes].
            labels: Int tensor of shape [batch].

        Returns:
            Scalar loss tensor.
        """
        num_classes = 5  # Fixed for the SC classification task
        batch_size = logits.size(0)
        labels_one_hot = labels

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0).repeat(batch_size, 1)
                weights = (weights * labels_one_hot).sum(1).unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            return focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            return F.cross_entropy(input=logits, target=labels_one_hot, weight=None)
        elif self.loss_type == "binary_cross_entropy":
            return F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, weight=weights
            )
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            return F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver:
    """Manages training, validation, and testing of the SC classifier.

    Args:
        config:       Namespace of hyperparameters and paths (see main.py).
        train_loader: DataLoader for the training split.
        valid_loader: DataLoader for the validation split.
        test_loader:  DataLoader for the test split.
        patch_size:   Tuple describing the input tensor shape, e.g. (B, C, H, W).
        launch_wandb: Whether to initialise a Weights & Biases run (default True).
    """

    def __init__(self, config, train_loader, valid_loader, test_loader,
                 patch_size, launch_wandb=True):
        torch.cuda.empty_cache()

        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.patch_size = patch_size

        # Model configuration
        self.model_type = config.model_type
        self.train_mode = config.train_mode
        self.pretrained = config.pretrained
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.nnlevel = config.nnlevel

        # Training hyperparameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weightDecay = config.weightDecay
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr_decay_rate = config.lr_decay_rate
        self.lr_decay_frequency = config.lr_decay_frequency
        self.min_lr = config.min_lr
        self.augmentation_prob = config.augmentation_prob
        self.t = config.t

        # Confidence-weighted loss options
        self.conf_score = config.conf_score
        self.norm_wei = config.norm_wei

        # Logging and checkpointing
        self.log_step = config.log_step
        self.val_step = config.val_step
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.checkpoint_dir = config.checkpoint_dir
        self.image_callback_freq = config.image_callback_freq
        self.saveImages = config.saveImages
        self.suf = config.model_suf

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up loss criterion
        reduction = "none" if self.conf_score else "mean"
        self.loss_func = config.loss_func
        if self.loss_func == "BCE":
            self.criterion = nn.BCELoss(reduction=reduction)
        elif self.loss_func == "wBCE":
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([config.positiveWeight]).to(self.device)
            )
        elif self.loss_func == "focal":
            self.criterion = Loss(loss_type="focal_loss")
        elif self.loss_func == "BCE2":
            self.criterion = Loss(loss_type="binary_cross_entropy")
        elif self.loss_func == "focal_balance":
            self.criterion = Loss(
                loss_type="focal_loss",
                samples_per_class=config.sample_counts,
                class_balanced=True,
            )
        elif self.loss_func == "MCE":
            self.criterion = Loss(loss_type="cross_entropy")
        else:
            raise ValueError(f"Unknown loss function: {self.loss_func}")

        # Build model and set up paths
        self.build_model()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        timestr = time.strftime("%Y%m%d-%H%M")
        ps = self.patch_size[-1]
        self.nnet_path_dir = os.path.join(
            self.model_path,
            f"classification_{timestr}-{self.model_type}"
            f"-Epo{self.num_epochs}-patch_{ps}"
            f"-Loss_{self.loss_func}-_{self.suf}",
        )
        print(f"Model checkpoint directory: {self.nnet_path_dir}")

        if self.mode == "train":
            os.makedirs(self.nnet_path_dir, exist_ok=True)

        if self.saveImages:
            self.image_callback_dir = os.path.join(
                config.imageCallbackDir, f"model_{timestr}-{self.model_type}"
            )
            os.makedirs(self.image_callback_dir, exist_ok=True)

        self._checkpoint = ModelCheckpoint(
            self.checkpoint_dir,
            self.config.model_type,
            self.config.mode,
            run_config=self.config,
            resume=0,
        )

        if self.mode == "train" and launch_wandb:
            Wandb.launch(config, 1)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self):
        """Build EfficientNet-B1 with a 4-channel input and multi-class head.

        The first convolutional layer is modified to accept 4-band PlanetScope
        imagery (R, G, B, NIR). The classifier head is replaced with a
        four-layer fully-connected network producing `output_ch` logits.
        Pre-trained ImageNet weights are loaded for all layers except the
        modified input conv and the new classifier head.
        """
        self.nnet = models.efficientnet_b1(
            weights=efficientnet.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        # Adapt input layer for 4-channel imagery
        self.nnet.features[0] = nn.Conv2d(
            in_channels=4, out_channels=32,
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        num_features = self.nnet.classifier[1].in_features
        self.nnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 640),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(640, 320),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(320, 64),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(64, self.output_ch),
        )

        # Optimise all parameters
        params_to_update = [p for p in self.nnet.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay
        )
        self.nnet.to(self.device)
        self.print_network(self.nnet, self.model_type)

    def print_network(self, model, name):
        """Print a torchinfo summary of the model."""
        num_params = sum(p.numel() for p in model.parameters())
        print(f"--- Model: {name} ---")
        model_stats = summary(
            model,
            input_size=self.patch_size,
            col_names=("input_size", "output_size", "num_params"),
        )
        print(model_stats)
        print(f"Total parameters: {num_params:,}")
        self.summary_str = str(model_stats)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _weighted_loss(self, loss, sp_wei):
        """Apply per-sample confidence weights to a batch loss.

        Args:
            loss:   Unreduced loss tensor of shape [batch].
            sp_wei: Per-sample weight tensor of shape [batch].

        Returns:
            Scalar weighted loss.
        """
        if self.norm_wei:
            return (loss * sp_wei).sum() / sp_wei.sum()
        return torch.mean(loss * sp_wei)

    def _unpack_batch(self, data):
        """Unpack a data batch according to the training configuration.

        Returns:
            images, GT, sp_wei (sp_wei is None when conf_score=False)
        """
        if self.conf_score:
            images, GT, sp_wei = data
            sp_wei = sp_wei.to(self.device)
        else:
            images, GT = data
            sp_wei = None
        return images.to(self.device), GT.to(self.device), sp_wei

    def _compute_loss(self, pred, GT):
        """Compute classification loss from logits and integer class labels.

        Applies sigmoid for BCE-based losses or cross-entropy for MCE.

        Returns:
            loss, pred  (pred is updated in-place for BCE to include sigmoid)
        """
        if self.loss_func == "BCE":
            pred = torch.sigmoid(pred)
            GT = GT.to(torch.float32)
            loss = self.criterion(pred, GT)
        elif self.loss_func == "wBCE":
            GT = GT.to(torch.float32)
            loss = self.criterion(pred, GT)
            pred = torch.sigmoid(pred)
        elif self.loss_func == "MCE":
            GT = GT.to(torch.int64)
            loss = F.cross_entropy(pred, GT)
        else:
            raise ValueError(f"Unexpected loss function in _compute_loss: {self.loss_func}")
        return loss, pred

    # ------------------------------------------------------------------
    # Train / valid / test loops
    # ------------------------------------------------------------------

    def train_epoch(self, epoch, lr):
        """Run one training epoch.

        Args:
            epoch: Current epoch index (0-based).
            lr:    Current learning rate (used for decay logging only).

        Returns:
            Tuple of (loss, f1, sensitivity) averages for the epoch.
        """
        self.nnet.train(True)
        self.stage = "train"

        batch_time = AverageMeter()
        losses = AverageMeter()
        f1_meter = AverageMeter()
        acc_meter = AverageMeter()
        sens_meter = AverageMeter()
        prec_meter = AverageMeter()

        end = time.time()
        epoch_start = time.time()

        for i, data in enumerate(self.train_loader):
            images, GT, sp_wei = self._unpack_batch(data)

            pred = self.nnet(images).squeeze()
            loss, pred = self._compute_loss(pred, GT)

            if self.conf_score:
                loss = self._weighted_loss(loss, sp_wei)
                losses.update(loss.item(), sp_wei.sum().item())
            else:
                losses.update(loss.item(), images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_np = pred.cpu().detach().numpy()
            GT_np = GT.cpu().detach().numpy()
            pred_labels = np.argmax(pred_np, axis=1)

            f1_meter.update(f1_score(GT_np, pred_labels, average="macro", zero_division=1))
            acc_meter.update(accuracy_score(GT_np, pred_labels))
            sens_meter.update(recall_score(GT_np, pred_labels, average="macro", zero_division=1))
            prec_meter.update(precision_score(GT_np, pred_labels, average="macro", zero_division=1))

            batch_time.update(time.time() - end)
            end = time.time()

            if self.saveImages and i == 0 and epoch % self.image_callback_freq == 0:
                self.image_callback(
                    epoch,
                    images.cpu().detach().numpy(),
                    pred_labels,
                    GT_np,
                    stage="_train",
                )

        print(
            f"Epoch [{epoch+1}/{self.num_epochs}] "
            f"Time [{time.time()-epoch_start:.1f}s / {batch_time.avg:.2f}s/it] "
            f"Train — Loss: {losses.avg:.4f}  Acc: {acc_meter.avg:.4f}  "
            f"F1: {f1_meter.avg:.4f}  Sens: {sens_meter.avg:.4f}  Prec: {prec_meter.avg:.4f}"
        )

        # Learning rate decay
        if epoch % self.lr_decay_frequency == 0:
            new_lr = self.lr * (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
            if new_lr > self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                print(f"  LR decayed to {new_lr:.6f}")

        self.finalize_epoch(epoch, losses.avg, f1_meter.avg, sens_meter.avg)
        return losses.avg, f1_meter.avg, sens_meter.avg

    def valid_epoch(self, epoch):
        """Run one validation epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Tuple of (loss, f1, sensitivity) averages for the epoch.
        """
        self.nnet.eval()
        self.stage = "val"

        batch_time = AverageMeter()
        val_losses = AverageMeter()
        val_f1 = AverageMeter()
        val_acc = AverageMeter()
        val_sens = AverageMeter()
        val_prec = AverageMeter()

        end = time.time()
        epoch_start = time.time()

        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                images, GT, sp_wei = self._unpack_batch(data)

                pred = self.nnet(images).squeeze()
                loss, pred = self._compute_loss(pred, GT)

                if self.conf_score:
                    loss = self._weighted_loss(loss, sp_wei)
                    val_losses.update(loss.item(), sp_wei.sum().item())
                else:
                    val_losses.update(loss.item(), images.size(0))

                pred_np = pred.cpu().numpy()
                GT_np = GT.cpu().numpy()
                pred_labels = np.argmax(pred_np, axis=1)

                val_f1.update(f1_score(GT_np, pred_labels, average="macro", zero_division=1), images.size(0))
                val_acc.update(accuracy_score(GT_np, pred_labels), images.size(0))
                val_sens.update(recall_score(GT_np, pred_labels, average="macro", zero_division=1), images.size(0))
                val_prec.update(precision_score(GT_np, pred_labels, average="macro", zero_division=1), images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if self.saveImages and i == 0 and epoch % self.image_callback_freq == 0:
                    self.image_callback(
                        epoch,
                        images.cpu().numpy(),
                        pred_labels,
                        GT_np,
                        stage="_valid",
                    )

        print(
            f"Epoch [{epoch+1}/{self.num_epochs}] "
            f"Time [{time.time()-epoch_start:.1f}s / {batch_time.avg:.2f}s/it] "
            f"Val   — Loss: {val_losses.avg:.4f}  Acc: {val_acc.avg:.4f}  "
            f"F1: {val_f1.avg:.4f}  Sens: {val_sens.avg:.4f}  Prec: {val_prec.avg:.4f}"
        )

        self.finalize_epoch(epoch, val_losses.avg, val_f1.avg, val_sens.avg)
        return val_losses.avg, val_f1.avg, val_sens.avg

    def test(self):
        """Evaluate the model on the held-out test set.

        Loads model weights from config.model_path and reports per-class
        and overall F1, accuracy, recall, and precision.

        Returns:
            Tuple of (labels, predictions, f1, accuracy, recall, precision).
        """
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.config.model_path}"
            )
        self.nnet.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        print(f"Loaded weights from {self.config.model_path}")

        self.nnet.eval()

        lb_list, pd_list = [], []
        with torch.no_grad():
            for data in self.test_loader:
                images, GT, sp_wei = self._unpack_batch(data)
                pred = torch.sigmoid(self.nnet(images).squeeze())
                lb_list.extend(GT.cpu().numpy())
                pd_list.extend(pred.cpu().numpy())

        labels = np.array(lb_list)
        preds = (np.array(pd_list) > 0.5).astype(int)

        print(f"Test set size: {len(labels)}")
        print(f"  F1        : {f1_score(labels, preds, zero_division=1):.4f}")
        print(f"  Accuracy  : {accuracy_score(labels, preds):.4f}")
        print(f"  Recall    : {recall_score(labels, preds, zero_division=1):.4f}")
        print(f"  Precision : {precision_score(labels, preds, zero_division=1):.4f}")

        return (
            labels, preds,
            f1_score(labels, preds, zero_division=1),
            accuracy_score(labels, preds),
            recall_score(labels, preds, zero_division=1),
            precision_score(labels, preds, zero_division=1),
        )

    def train(self):
        """Full training loop with early stopping and checkpoint saving.

        Saves three checkpoints:
            bestLoss.pkl  — lowest validation loss
            bestF1.pkl    — highest validation macro-F1
            epoch_N.pkl   — snapshot every 30 epochs
        """
        nnet_path_loss = os.path.join(self.nnet_path_dir, "bestLoss.pkl")
        nnet_path_f1 = os.path.join(self.nnet_path_dir, "bestF1.pkl")

        lr = self.lr
        best_loss = float("inf")
        best_f1 = -float("inf")

        early_stopper = EarlyStopper(patience=self.config.patience) if self.config.earlystop else None

        for epoch in range(self.num_epochs):
            print("-" * 80)
            train_loss, train_f1, _ = self.train_epoch(epoch, lr)
            val_loss, val_f1, _ = self.valid_epoch(epoch)

            # Early stopping check
            if early_stopper is not None and early_stopper.early_stop(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

            # Save best-loss checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.nnet.state_dict(), nnet_path_loss)
                if not os.path.isfile(nnet_path_loss.replace("pkl", "log")):
                    with open(nnet_path_loss.replace("pkl", "log"), "w") as f:
                        f.write(self.summary_str)
                print(f"  ✓ New best loss: {best_loss:.4f} → saved to {nnet_path_loss}")

            # Save best-F1 checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.nnet.state_dict(), nnet_path_f1)
                print(f"  ✓ New best F1:   {best_f1:.4f} → saved to {nnet_path_f1}")

            # Periodic snapshot
            if (epoch + 1) % 30 == 0:
                snap_path = nnet_path_loss.replace("bestLoss.pkl", f"epoch_{epoch+1}.pkl")
                torch.save(self.nnet.state_dict(), snap_path)
                print(f"  Snapshot saved at epoch {epoch+1}: {snap_path}")

    # ------------------------------------------------------------------
    # Callbacks and logging
    # ------------------------------------------------------------------

    def image_callback(self, epoch, img, pred, lab, stage="_train"):
        """Save a 3×3 grid of sample images with predicted and true class labels.

        Args:
            epoch: Current epoch (used in filename).
            img:   Numpy array of images, shape [B, C, H, W].
            pred:  Predicted class indices, shape [B].
            lab:   True class indices, shape [B].
            stage: String suffix for the saved file ('_train' or '_valid').
        """
        plt.figure(figsize=(20, 20))
        for j in range(3):
            for i in range(3):
                idx = 3 * j + i
                plt.subplot(3, 3, idx + 1)
                plt.imshow(np.transpose(img[idx], (1, 2, 0))[:, :, :3])
                plt.title(f"True: {int(lab[idx])}  Pred: {int(pred[idx])}")
        plt.tight_layout()
        fig_path = os.path.join(
            self.image_callback_dir, f"Epoch{epoch}{stage}.jpg"
        )
        plt.savefig(fig_path)
        plt.close("all")

    def finalize_epoch(self, epoch, loss, f1, sens):
        """Log metrics to Weights & Biases and update the model checkpoint.

        Args:
            epoch: Current epoch index.
            loss:  Average loss for the epoch.
            f1:    Macro-averaged F1 score.
            sens:  Macro-averaged sensitivity (recall).
        """
        metrics = get_metrics_classification(
            {"loss": loss, "f1": f1, "sens": sens}, self.stage
        )
        wandb.log(metrics, step=epoch)
        wandb.config.update({"model_name": self.config.model_type}, allow_val_change=True)

        self._checkpoint.save_best_models_under_current_metrics(
            self.nnet,
            {"epoch": epoch, "stage": self.stage, "current_metrics": metrics},
        )
        Wandb.add_file(self._checkpoint.checkpoint_path)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Stop training when validation loss has not improved for `patience` epochs.

    Args:
        patience:  Number of epochs to wait without improvement (default 50).
        min_delta: Minimum improvement threshold (default 0).
    """

    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track the running average of a metric over a sequence of updates.

    Example::
        meter = AverageMeter()
        meter.update(0.5, n=32)
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = div0(self.sum, self.count)


def get_metrics_classification(metrics: dict, stage: str) -> dict:
    """Format classification metrics for W&B logging.

    Args:
        metrics: Dict with keys 'loss', 'f1', 'sens'.
        stage:   One of 'train' or 'val'.

    Returns:
        Dict with stage-prefixed keys.
    """
    return {
        f"{stage}_loss": metrics["loss"],
        f"{stage}_f1":   metrics["f1"],
        f"{stage}_sens": metrics["sens"],
    }


def div0(x, y):
    """Divide x by y, returning 0 on ZeroDivisionError."""
    try:
        return x / y
    except ZeroDivisionError:
        return 0.0
