from omegaconf import DictConfig
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from lightning import LightningModule
from torchmetrics import MeanMetric

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.finetuning.loss import MeanOverSamplesCE
from cocoa_mapping.kalitschek_training.loss import DiceLoss
from cocoa_mapping.models.model_utils import model_contains_unfrozen_encoder


class FinetuningLightningModule(LightningModule):
    def __init__(self,
                 model: AbstractTorchModel,
                 config: DictConfig
                 ):
        """Initialize the FinetuningLightningModule.

        Args:
            model: The model to finetune.
            config: The configuration.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # do not log large paths repeatedly
        self.config = config
        self.model = model

        if config.loss == 'dice':
            self.loss_fn = DiceLoss(ignore_index=3)
        elif config.loss == 'ce':
            self.loss_fn = MeanOverSamplesCE(ignore_index=3)

        # --- Metrics ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        # Update samples counter and skipped counter
        self.train_samples_counter += len(batch['mask'])
        self.train_samples_skipped += int(torch.all(batch['mask'] == 3, dim=(1, 2)).sum().item())
        self.log('s', self.train_samples_skipped, on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log('t', self.train_samples_counter, on_step=True, on_epoch=False, logger=False, prog_bar=True)

        # Compute loss and confusion counts
        loss, (tp, fp, tn, fn) = self._compute_loss_and_counts(batch)

        # Accumulate
        self.train_tp += tp
        self.train_fp += fp
        self.train_tn += tn
        self.train_fn += fn
        self.train_loss.update(loss.item())
        self.log('train/loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        # Update samples counter and skipped counter
        self.val_samples_counter += len(batch['mask'])
        self.val_samples_skipped += int(torch.all(batch['mask'] == 3, dim=(1, 2)).sum().item())
        self.log('s', self.val_samples_skipped, on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log('t', self.val_samples_counter, on_step=True, on_epoch=False, logger=False, prog_bar=True)

        # Compute loss and confusion counts
        loss, (tp, fp, tn, fn) = self._compute_loss_and_counts(batch)
        self.val_tp += tp
        self.val_fp += fp
        self.val_tn += tn
        self.val_fn += fn
        self.val_loss.update(loss.item())
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _compute_loss_and_counts(self, batch) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
        images: torch.Tensor = batch['image']  # (B, C, H, W)
        masks: torch.Tensor = batch['mask']    # (B, H, W) with values {0,1,3}
        logits: torch.Tensor = self.forward(images)
        loss = self.loss_fn(logits, masks.long())

        # Predictions using threshold on class-1 probability
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)[:, 1, :, :]
            preds = (probs >= 0.5).long()
            tp, fp, tn, fn = self._compute_normalized_confusion_matrix(preds, masks)
        return loss, (tp.item(), fp.item(), tn.item(), fn.item())

    @staticmethod
    def _compute_normalized_confusion_matrix(preds: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        labeled = masks != 3
        if labeled.sum() == 0:
            return (torch.tensor(0.0, device=preds.device),) * 4
        pred_pos = preds == 1
        pred_neg = preds == 0
        true_pos = masks == 1
        true_neg = masks == 0

        # Per-polygon weighting: count proportion of correct pixels within labeled area
        # Note: Our dataset provides a single geometry per sample; treat all labeled pixels as one region
        region = labeled
        region_size = region.sum(dim=(1, 2)).clamp_min(1)
        tp = (((pred_pos & true_pos) & region).sum(dim=(1, 2)).float() / region_size).sum()
        fp = (((pred_pos & true_neg) & region).sum(dim=(1, 2)).float() / region_size).sum()
        tn = (((pred_neg & true_neg) & region).sum(dim=(1, 2)).float() / region_size).sum()
        fn = (((pred_neg & true_pos) & region).sum(dim=(1, 2)).float() / region_size).sum()
        return tp, fp, tn, fn

    def on_train_epoch_start(self):
        self._reset_counters(split='train')
        self.train_loss.reset()
        self.train_samples_counter = 0
        self.train_samples_skipped = 0

    def on_validation_epoch_start(self):
        self._reset_counters(split='val')
        self.val_loss.reset()
        self.val_samples_counter = 0
        self.val_samples_skipped = 0

    def on_train_epoch_end(self):
        metrics = self._compute_metrics_from_counts(self.train_tp, self.train_fp, self.train_tn, self.train_fn)
        for k, v in metrics.items():
            self.log(f'train/{k}', v, on_step=False, on_epoch=True, prog_bar=(k in ['recall', 'precision']))
        self.log('train/loss_epoch', self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self._compute_metrics_from_counts(self.val_tp, self.val_fp, self.val_tn, self.val_fn)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, prog_bar=(k in ['recall', 'precision', 'loss']))
        self.log('val/loss_epoch', self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

    @staticmethod
    def _compute_metrics_from_counts(tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        denom = (tp + fp + tn + fn)
        accuracy = (tp + tn) / denom if denom > 0 else 0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }

    def _reset_counters(self, split: str):
        setattr(self, f'{split}_tp', 0)
        setattr(self, f'{split}_fp', 0)
        setattr(self, f'{split}_tn', 0)
        setattr(self, f'{split}_fn', 0)

    def configure_optimizers(self):
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
        scheduler = MultiStepLR(opt, **self.config.scheduler)

        # Add a warm-up scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: min(1, (epoch + 1) / self.config.warmup.epochs))

        # Combine warm-up and main scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[self.config.warmup.epochs]
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
