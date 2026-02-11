import os
from typing import Callable
from geopandas import gpd
from omegaconf import OmegaConf

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cocoa_mapping.finetuning.combined_dataset import CombinedFinetunePretrainDataset
from cocoa_mapping.kalitschek_training.dataset import H5Dataset
from cocoa_mapping.finetuning.finetuning_dataset import FinetuningDataset


class FinetuningDataModule(LightningDataModule):
    def __init__(self,
                 train_gdf: gpd.GeoDataFrame,
                 val_gdf: gpd.GeoDataFrame,
                 config: OmegaConf,
                 transform: Callable):
        """Initialize the FinetuningDataModule.

        Args:
            train_gdf: The geo dataframe to use for the training dataset.
            val_gdf: The geo dataframe to use for the validation dataset.
            config: The configuration.
            transform: The transform to apply to the data.
        """
        super().__init__()
        self.train_gdf = train_gdf
        self.val_gdf = val_gdf
        self.cfg = config
        self.transform = transform

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = self._get_train_dataset()
            self.val_dataset = FinetuningDataset(gdf=self.val_gdf, transform=self.transform, config=self.cfg)
        elif stage == 'predict':
            raise NotImplementedError
        elif stage == 'test':
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def _get_train_dataset(self) -> FinetuningDataset:
        finetune_dataset = FinetuningDataset(gdf=self.train_gdf, transform=self.transform, config=self.cfg)
        if self.cfg.pretraining_samples.ratio_pct == 0:
            return finetune_dataset

        pretraining_samples_path = self.cfg.pretraining_samples[self.cfg.image_type].training_data_file
        mode = os.path.basename(pretraining_samples_path).replace(".hdf5", "")
        pretrain_dataset = H5Dataset(mode=mode,
                                     data_dir=os.path.dirname(pretraining_samples_path),
                                     preprocessor=self.transform)
        return CombinedFinetunePretrainDataset(finetune_dataset=finetune_dataset,
                                               pretrain_dataset=pretrain_dataset,
                                               pretrain_ratio_pct=self.cfg.pretraining_ratio_pct,
                                               seed=self.cfg.seed)

    def train_dataloader(self) -> DataLoader:
        cfg_num_workers = self.cfg.loader.train.num_workers
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.loader.train.batch_size,
                          num_workers=cfg_num_workers if cfg_num_workers is not None else os.cpu_count(),
                          persistent_workers=self.cfg.loader.train.persistent_workers,
                          pin_memory=True,
                          shuffle=(self.cfg.trainer.get("overfit_batches", 0) == 0))

    def val_dataloader(self) -> DataLoader:
        cfg_num_workers = self.cfg.loader.train.num_workers
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.loader.train.batch_size,
                          num_workers=cfg_num_workers if cfg_num_workers is not None else os.cpu_count(),
                          persistent_workers=self.cfg.loader.train.persistent_workers,
                          pin_memory=True,
                          shuffle=False)
