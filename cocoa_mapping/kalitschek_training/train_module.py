"""
This code is adjusted from https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
"""

from typing import Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy, Precision, Recall, MeanMetric, F1Score
import lightning as L
from omegaconf import DictConfig

from cocoa_mapping.kalitschek_training.loss import DiceLoss
from cocoa_mapping.models.model_utils import model_contains_unfrozen_encoder


class TrainingModule(L.LightningModule):
    """Training module to distill Kalitschek et al. cocoa mapping."""

    def __init__(self, config: DictConfig, model: torch.nn.Module):
        super().__init__()
        # Save everything needed to reproduce the run (shows in checkpoints and WandB via the logger)
        self.save_hyperparameters(ignore=["model"])  # stores config; keep raw model object out of hparams
        self.config = config
        self.model = model

        # --- Loss ---
        self.loss_fn = DiceLoss(ignore_index=3)

        # --- Metrics ---
        self.train_acc = Accuracy(ignore_index=3, task="binary")
        self.train_prec = Precision(ignore_index=3, task="binary")
        self.train_rec = Recall(ignore_index=3, task="binary")
        self.train_f1 = F1Score(ignore_index=3, task="binary")
        self.train_loss = MeanMetric()

        self.val_acc = Accuracy(ignore_index=3, task="binary")
        self.val_prec = Precision(ignore_index=3, task="binary")
        self.val_rec = Recall(ignore_index=3, task="binary")
        self.val_f1 = F1Score(ignore_index=3, task="binary")
        self.val_loss = MeanMetric()

    # ------ core Lightning hooks ------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # for inference / .predict
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            preds = preds.flatten()
            y_flat = y.squeeze(1).flatten()

            self.train_acc.update(preds, y_flat)
            self.train_prec.update(preds, y_flat)
            self.train_rec.update(preds, y_flat)
            self.train_f1.update(preds, y_flat)
            self.train_loss.update(loss.detach())

        # let Lightning log per-step loss to the progress bar, epoch aggregation below
        self.log("train/step_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        # epoch-wise logging (Lightning will reset when we reset the Metric objects)
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_prec.compute(), on_epoch=True)
        self.log("train/recall", self.train_rec.compute(), on_epoch=True)
        self.log('train/f1', self.train_f1.compute(), on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss.compute(), on_epoch=True)

        # reset epoch metrics
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_rec.reset()
        self.train_f1.reset()
        self.train_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        preds = preds.flatten()
        y_flat = y.squeeze(1).flatten()

        self.val_acc.update(preds, y_flat)
        self.val_prec.update(preds, y_flat)
        self.val_rec.update(preds, y_flat)
        self.val_f1.update(preds, y_flat)
        self.val_loss.update(loss.detach())

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc", self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_prec.compute(), on_epoch=True)
        self.log("val/recall", self.val_rec.compute(), on_epoch=True)
        self.log('val/f1', self.val_f1.compute(), on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss.compute(), on_epoch=True)

        self.val_acc.reset()
        self.val_prec.reset()
        self.val_rec.reset()
        self.val_f1.reset()
        self.val_loss.reset()

    def configure_optimizers(self) -> dict:
        if not model_contains_unfrozen_encoder(self.model):
            opt = Adam(self.model.parameters(), **self.config.optimizer)
        else:
            encoder_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder.")]
            non_encoder_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
            assert len(encoder_params) > 0 and len(non_encoder_params) > 0, "Expect to find both encoder and non-encoder parameters"
            opt = Adam([
                {"params": encoder_params, **self.config.encoder_optimizer},
                {"params": non_encoder_params, **self.config.optimizer}
            ])

        sch = MultiStepLR(opt, **self.config.scheduler)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
            },
        }
