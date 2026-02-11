import numpy as np
import torch

from cocoa_mapping.finetuning.finetuning_dataset import FinetuningDataset
from cocoa_mapping.kalitschek_training.dataset import H5Dataset

PRETRAIN_SAMPLE_IDX = -1
"""Pretraining samples are pulled randomly"""


class CombinedFinetunePretrainDataset(torch.utils.data.Dataset):
    """A dataset that mixes finetuning and pretraining samples in a fixed ratio."""

    def __init__(self, finetune_dataset: FinetuningDataset,
                 pretrain_dataset: H5Dataset,
                 pretrain_ratio_pct: float,
                 seed: int = 42,
                 shuffle: bool = False):
        """Initializes the CombinedFinetunePretrainDataset.

        Args:
            finetune_dataset: The finetune dataset.
            pretrain_dataset: The pretrain dataset. Is expected to have much more samples than the finetune dataset.
            pretrain_ratio_pct: The percentage of pretrain samples in each 100-sample block.
            seed: The seed for the random number generator, used for shuffling pretrain samples and the 100-sample pattern.
            shuffle: Whether to shuffle the combined dataset indices. Normally not needed as the DataLoader will handle shuffling during training.
        """
        if pretrain_ratio_pct <= 0 or pretrain_ratio_pct >= 100:
            raise ValueError("pretrain_ratio_pct must be greater than 0 and less than 100")

        self.finetune_dataset = finetune_dataset
        self.pretrain_dataset = pretrain_dataset
        self.pretrain_ratio_pct = pretrain_ratio_pct
        np.random.seed(seed)  # Will hopefully ensure that the indices reset will be the same for each worker.

        # Calculate the number of finetune and pretrain samples
        self.total_num_finetune_samples = len(finetune_dataset)
        self.num_pretrain_samples_target = round(
            self.total_num_finetune_samples / (100 - pretrain_ratio_pct) * pretrain_ratio_pct
        )
        self.total_mixed_length = self.total_num_finetune_samples + self.num_pretrain_samples_target

        # Build a mapping from the combined dataset index to the finetune or pretrain dataset index
        self.idx_to_dataset_idx = np.concatenate([
            np.arange(self.total_num_finetune_samples, dtype=np.int32),
            np.full(self.num_pretrain_samples_target, fill_value=PRETRAIN_SAMPLE_IDX, dtype=np.int32),
        ])
        assert len(self.idx_to_dataset_idx) == self.total_mixed_length

        # Shuffle if requested
        if shuffle:
            self.idx_to_dataset_idx = np.random.permutation(self.idx_to_dataset_idx)

        # Pretraining sample randomization state
        self._reset_pretrain_indices()

    def _reset_pretrain_indices(self):
        """Reshuffle the list of remaining pretraining indices."""
        # If we would use for it for validation, we may want to keep seed constant. But now, we only use it for training.
        self.remaining_pretrain_indices = np.random.permutation(len(self.pretrain_dataset)).tolist()

    def _draw_pretrain_index(self):
        """Pop one unused pretraining index; reshuffle if depleted."""
        if not self.remaining_pretrain_indices:
            self._reset_pretrain_indices()
        return self.remaining_pretrain_indices.pop()

    def __len__(self):
        return self.total_mixed_length

    def __getitem__(self, idx):
        dataset_index = self.idx_to_dataset_idx[idx]

        if dataset_index == PRETRAIN_SAMPLE_IDX:
            pretrain_index = self._draw_pretrain_index()
            image, label = self.pretrain_dataset[pretrain_index]
            return {
                'image': image,
                'mask': label,
            }
        else:
            return self.finetune_dataset[dataset_index]
