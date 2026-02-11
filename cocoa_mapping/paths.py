from enum import Enum
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Paths(Enum):
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
    ENV_FILE = os.path.join(ROOT_DIR, ".env")

    # Auxiliary data
    AUXILIARY_DATA_DIR = os.path.join(ROOT_DIR, "auxiliary_data")
    WORLD_BOUNDARIES = os.path.join(AUXILIARY_DATA_DIR, "world_administrative_boundaries_countries.geojson")
    SENTINEL_2_GRID = os.path.join(AUXILIARY_DATA_DIR, "sentinel_2_grid.geojson")

    # If we are on aws, either use DATA_DIR if set or create a new one in the root directory.
    DATA_DIR = os.getenv('DATA_DIR', os.path.join(ROOT_DIR, 'data'))
    # In case the data directory read-only, set OUTPUT_DATA_DIR. Otherwise, use the data directory.
    OUTPUT_DATA_DIR = os.getenv('OUTPUT_DATA_DIR', DATA_DIR)

    # We save everything in data directory for this project so that when we sync data with S3, all data is saved in the same place.
    MODELS_DIR = os.path.join(OUTPUT_DATA_DIR, "models")
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DATA_DIR, "checkpoints")
    PRETAINED_MODELS_DIR = os.path.join(OUTPUT_DATA_DIR, "pretrained_models")
    CACHE_DIR = os.path.join(OUTPUT_DATA_DIR, "cache")
    TEMP_DIR = os.path.join(OUTPUT_DATA_DIR, "temp")
    TEST_INPUT_DIR = os.path.join(OUTPUT_DATA_DIR, "test_input")
    TEST_RESULTS_DIR = os.path.join(OUTPUT_DATA_DIR, "test_results")

    # Kalitschek training
    KALITSCHEK_TRAINING_CONFIGS_DIR = os.path.join(ROOT_DIR, "cocoa_mapping", "kalitschek_training", "configs")
    KALITSCHEK_TRAINING_DEFAULT_DATA_DIR = os.path.join(DATA_DIR, "training_data")
    KALITSCHEK_PROBS = os.path.join(DATA_DIR, "kalitschek_probs.tif")

    # Finetuning
    FINETUNING_CONFIGS_DIR = os.path.join(ROOT_DIR, "cocoa_mapping", "finetuning", "configs")
    NIGERIA_TRAINING_DATA_DIR = os.path.join(DATA_DIR, "nigeria_training_data")

    # AEFs
    AEF_KALITSCHEK_TRAINING_DEFAULT_DATA_DIR = os.path.join(DATA_DIR, "training_data_aef")
    AEF_NIGERIA_TRAINING_DATA_DIR = os.path.join(DATA_DIR, "nigeria_training_data_aef")
