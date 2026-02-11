import torch
import torch.nn.functional as F


class MeanOverSamplesCE(torch.nn.Module):
    """Cross-entropy loss normalized or weighted by the number of valid pixels per sample."""

    def __init__(self, ignore_index: int = 3):
        """Initialize the MeanOverSamplesCE loss.

        Args:
            ignore_index: The index to ignore in the loss calculation.
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss averaged over samples."""
        loss_per_pixel = F.cross_entropy(logits, masks, reduction='none', ignore_index=self.ignore_index)  # (B, H, W)

        # Average loss per sample (ignoring index=self.ignore_index)
        valid_mask = (masks != self.ignore_index).float()  # (B, H, W)
        loss_per_sample = (loss_per_pixel * valid_mask).sum(dim=(1, 2)) / valid_mask.sum(dim=(1, 2)).clamp(min=1)
        return loss_per_sample.mean()  # Mean over samples
