import argparse
import os
import logging
import shlex
import sys

import geopandas as gpd
import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import OmegaConf
import torch

from cocoa_mapping.utils.training_utils import get_device
from cocoa_mapping.finetuning.finetune_utils import assemble_datasets, prepare_training_dataset
from cocoa_mapping.finetuning.finetuning_data_module import FinetuningDataModule
from cocoa_mapping.finetuning.finetuning_train_module import FinetuningLightningModule
from cocoa_mapping.models.model_utils import model_contains_unfrozen_encoder, save_model_and_preprocessor, download_model_if_not_exist, upload_model
from cocoa_mapping.models.models_preprocessors_registry import load_preprocessor, load_model
from cocoa_mapping.paths import Paths
from cocoa_mapping.utils.general_utils import load_env_file, remove_system_keyword_arguments
from cocoa_mapping.utils.logging_utils import get_annotation_distribution


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=Paths.FINETUNING_CONFIGS_DIR.value, config_name="config")
def finetune(config: OmegaConf):
    L.seed_everything(config.seed, workers=True)

    # Load datasets. First, we need to assemble the dataset and download data.
    # We sometimes combine multiple datasets, each of them having local path and mirrored s3 path (so we can run on AWS or locally)
    train_val_gdf = assemble_datasets(dataset_config=config.dataset, image_type=config.image_type, data_col='path')

    # Prepare the dataset for finetuning - ensure that data exists (or filter in debug mode), that all are in the came crs, etc.
    train_val_gdf = prepare_training_dataset(train_val_gdf, do_filter_on_existing_data=config.debug, data_col='path')

    # Split the annotations into train and val
    train_gdf, val_gdf = split_annotation_to_train_val_test(train_val_gdf, config)

    # Load model and preprocessor
    model, preprocessor = load_model_preprocessor(config)

    # Setup all modules
    training_module = FinetuningLightningModule(config=config, model=model)
    data_module = FinetuningDataModule(train_gdf=train_gdf,
                                       val_gdf=val_gdf,
                                       config=config,
                                       transform=preprocessor)
    data_module.setup(stage='fit')

    profiler = SimpleProfiler(extended=True)
    wandb_config = {
        **OmegaConf.to_container(config, resolve=True),
        "model_version": model.api_version,
        "preprocessor_version": preprocessor.api_version,
        'val_distribution': get_annotation_distribution(val_gdf),
        'train_distribution': get_annotation_distribution(train_gdf),
    }
    wandb_logger = loggers.WandbLogger(project=config.exp.project,
                                       name=config.exp.name,
                                       config=wandb_config) if config.logging else None

    # Model checkpointing
    checkpoints_dir = os.path.join(config.paths.checkpoints_dir or Paths.CHECKPOINTS_DIR.value, config.exp.project, config.exp.name)
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
        **config.trainer,
    )
    trainer.fit(model=training_module, datamodule=data_module)

    if trainer.global_rank == 0:
        logger.info("Saving best model")
        model_dir = os.path.join(config.paths.models_dir or Paths.MODELS_DIR.value, config.exp.name)
        best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=False)
        training_module.load_state_dict(best_ckpt["state_dict"], strict=False)
        save_model_and_preprocessor(model=training_module.model,
                                    preprocessor=preprocessor,
                                    model_dir=model_dir)

        if config.logging and config.upload_model:
            upload_model(model_dir, description=f"Cocoa Mapping model | Finetuned")


def load_model_preprocessor(config: OmegaConf):
    """Load the model and preprocessor based on the configuration..
    Freeze the encoder if needed.

    Args:
        config: The configuration.

    Returns:
        model: The model.
        preprocessor: The preprocessor.
    """
    if config.pretrained_model.name:
        # Get model and preprocessor
        logger.info(f"Loading pretrained model {config.pretrained_model.name} from {config.pretrained_model.project}")
        model_path = download_model_if_not_exist(model_name=config.pretrained_model.name,
                                                 project_name=config.pretrained_model.project,
                                                 models_dir=config.paths.models_dir or Paths.MODELS_DIR.value)
        preprocessor = load_preprocessor(model_path)
        model = load_model(model_path).to(get_device())

        # Freeze encoder if needed
        if config.pretrained_model.freeze_encoder and model_contains_unfrozen_encoder(model):
            logger.info("Freezing encoder")
            for param in model.encoder.parameters():
                param.requires_grad = False

        return model, preprocessor
    else:
        logger.info("Initializing model from config")
        return hydra.utils.instantiate(config.model.model), hydra.utils.instantiate(config.model.preprocessor)


def split_annotation_to_train_val_test(train_val_gdf: gpd.GeoDataFrame, config: OmegaConf) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split the annotations into train, validation, and test sets.

    Args:
        train_val_gdf: The training and validation annotations.
        config: The configuration.

    Returns:
        train_gdf: The training annotations.
        val_gdf: The validation annotations.
    """
    # We mainly used sklearn in this study for finetuning, as it was enough for our purposes.
    # However, for other use cases, we recommend to use more advanced sampling methods.
    if config.split.method != 'sklearn':
        raise NotImplementedError(f"Split method {config.split.method} is not implemented yet. Only sklearn is supported for now.")

    train_gdf, val_gdf = train_test_split(train_val_gdf, test_size=config.split.val, random_state=config.seed)

    # Log train and val distribution
    os.makedirs(Paths.TEMP_DIR.value, exist_ok=True)
    train_gdf.to_file(os.path.join(Paths.TEMP_DIR.value, 'train_gdf.geojson'))
    val_gdf.to_file(os.path.join(Paths.TEMP_DIR.value, 'val_gdf.geojson'))
    return train_gdf, val_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model.")
    parser.add_argument(
        '-n', '--name', type=str, default=None,
        help=(
            "Name of the run. You can provide this as an argparse argument or using hydra notation as exp.name='...'. "
            "Use argparse notation when running from AWS SageMaker script (it does not support hydra notation)."
        )
    )
    parser.add_argument(
        '-ha', '--hydra-args', type=str, default=None,
        help="Alternative way for providing hydra arguments via argparse. Use it when running from AWS SageMaker script (it does not support hydra notation)."
    )
    # Debug mode
    parser.add_argument(
        '-d', '--debug', type=int, default=0, choices=[0, 1],
        help="Debug mode."
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

    sys.argv.append(f"debug={'true' if args.debug else 'false'}")
    remove_system_keyword_arguments('-d', '--debug')

    finetune()
