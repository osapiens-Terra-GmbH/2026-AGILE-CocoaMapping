"""
This code is adjusted from https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
"""

import os
import torch.utils.data
from typing import Iterable
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra

from cocoa_mapping.kalitschek_training.dataset import H5Dataset


class KalitschekTrainingDataModule(L.LightningDataModule):
    def __init__(self, cfg: OmegaConf, data_dir: Path | str):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.preprocessor = hydra.utils.instantiate(cfg.preprocessor)

        # Values will be set in setup method
        self.train_dataset = None
        self.train_sampler = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.n_train_samples = None

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = H5Dataset(mode="train", data_dir=self.data_dir, preprocessor=self.preprocessor)
            self.train_sampler = None
            if self.cfg.loader.train.get("num_samples", None) and self.cfg.trainer.get("overfit_batches", 0) == 0:
                self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset, num_samples=self.cfg.loader.train.num_samples)
            self.val_dataset = H5Dataset(mode="val",
                                         data_dir=self.data_dir,
                                         preprocessor=self.preprocessor,
                                         n_samples=self.cfg.loader.val.get("num_samples", None))
        elif stage == 'predict':
            raise NotImplementedError
        elif stage == 'test':
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.loader.train.batch_size,
                          num_workers=self.cfg.loader.train.num_workers or os.cpu_count(),
                          sampler=self.train_sampler,
                          shuffle=(self.train_sampler is None and self.cfg.trainer.get("overfit_batches", 0) == 0))

    def val_dataloader(self) -> Iterable:
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.loader.val.batch_size,
                          num_workers=self.cfg.loader.val.num_workers or os.cpu_count(),
                          shuffle=False)
