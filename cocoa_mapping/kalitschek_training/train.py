"""
This code is adjusted from https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
"""

import argparse
import logging
import os
import shlex
import sys

import hydra
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from cocoa_mapping.kalitschek_training.train_module import TrainingModule
from cocoa_mapping.kalitschek_training.training_data_module import KalitschekTrainingDataModule
from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.models.model_utils import save_model_and_preprocessor, upload_model
from cocoa_mapping.paths import Paths
from cocoa_mapping.utils.general_utils import load_env_file, remove_system_keyword_arguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=Paths.KALITSCHEK_TRAINING_CONFIGS_DIR.value, config_name="config")
def main(cfg: OmegaConf):
    model: AbstractTorchModel = hydra.utils.instantiate(cfg.model)
    preprocessor: AbstractPreprocessor = hydra.utils.instantiate(cfg.preprocessor)

    training_module = TrainingModule(config=cfg, model=model)
    data_module = KalitschekTrainingDataModule(cfg=cfg, data_dir=cfg.paths.training_data_dir)  # training_data_dir is set in the main function
    data_module.setup(stage='fit')

    profiler = SimpleProfiler(extended=True)
    wandb_config = {
        **OmegaConf.to_container(cfg, resolve=True),
        "model_version": model.api_version,
        "preprocessor_version": preprocessor.api_version,
    }
    wandb_logger = loggers.WandbLogger(project=cfg.exp.project,
                                       name=cfg.exp.name,
                                       config=wandb_config) if cfg.logging else None

    # Model checkpointing
    checkpoints_dir = os.path.join(cfg.paths.checkpoints_dir or Paths.CHECKPOINTS_DIR.value, cfg.exp.project, cfg.exp.name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch:03d}-{val/f1:.4f}',
        monitor="val/f1",
        mode="max",
        save_top_k=10,
        save_last=True
    )

    trainer = L.Trainer(
        profiler=profiler,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        **cfg.trainer,
    )
    trainer.fit(model=training_module, datamodule=data_module)

    # Save the best model
    if trainer.global_rank == 0:
        model_dir = os.path.join(cfg.paths.models_dir or Paths.MODELS_DIR.value, cfg.exp.name)
        best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=False)
        training_module.load_state_dict(best_ckpt["state_dict"], strict=False)
        save_model_and_preprocessor(model=training_module.model,
                                    preprocessor=preprocessor,
                                    model_dir=model_dir)

        if cfg.logging and cfg.upload_model:
            upload_model(model_dir, description=f"Cocoa Mapping model")


def _check_if_training_data_exists(training_data_dir: str) -> bool:
    """Check if the training data exists in the given directory."""
    for split_file in ("train.hdf5", "val.hdf5"):
        if not os.path.exists(os.path.join(training_data_dir, split_file)):
            logger.info(f"Training data file {split_file} does not exist in {training_data_dir}.")
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name', type=str, default=None,
        help=(
            "Name of the run. You can provide this as an argparse argument or using hydra notation as exp.name='...'. "
            "Use argparse notation when running from AWS SageMaker script (it does not support hydra notation)."
        )
    )
    parser.add_argument(
        '-a', '--aef', type=int, choices=[0, 1], default=0,
        help="Whether to use AEF training data. Affects default training data directory and s3 path for training data."
    )
    parser.add_argument(
        '-ha', '--hydra-args', type=str, default=None,
        help="Alternative way for providing hydra arguments via argparse. Use it when running from AWS SageMaker script (it does not support hydra notation)."
    )
    parser.add_argument(
        '-td', '--training-data-dir', type=str, default=None,
        help="Data directory. If not provided, the default path will be used."
    )

    # Important: allow Hydra overrides like exp.name=foo, training.epochs=20, etc.
    args, _unknown = parser.parse_known_args()
    load_env_file()

    if args.name:
        # Inject name into Hydra config
        sys.argv.append(f"exp.name={args.name}")
        # Remove the name argument from the sys.argv
        remove_system_keyword_arguments('-n', '--name')

    if args.hydra_args:
        sys.argv.extend(shlex.split(args.hydra_args))
        remove_system_keyword_arguments('-ha', '--hydra-args')

    # Handle aef
    aef_mode = bool(args.aef)
    remove_system_keyword_arguments('-a', '--aef')

    # training data dir is set by default
    training_data_dir = args.training_data_dir
    if training_data_dir is None:
        training_data_dir = Paths.KALITSCHEK_TRAINING_DEFAULT_DATA_DIR.value if aef_mode == 0 \
            else Paths.AEF_KALITSCHEK_TRAINING_DEFAULT_DATA_DIR.value
    sys.argv.append(f"paths.training_data_dir={training_data_dir}")
    remove_system_keyword_arguments('-td', '--training-data-dir')

    # When running on aws and not in copy mode, we need to download the training data from S3
    if not _check_if_training_data_exists(training_data_dir):
        raise FileNotFoundError(f"Training data files do not exist in {training_data_dir}.")

    main()
