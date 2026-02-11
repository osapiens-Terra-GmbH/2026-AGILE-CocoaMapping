# Cocoa Mapping Project
General workflow and layout follow the experiments from *Generalizability of Foundation Models: A Case Study on Cocoa Mapping Across Countries Using Sparse Labels*. The repository is intended primarily as a reference for the paper; running the pipelines requires substantial setup, including configuring a PostGIS database, uploading tables, and downloading large volumes of imagery.

## Setup
- Run the targets in the Makefile first (e.g., environment setup, linting, tests). It captures the main commands needed to install dependencies and validate changes.
- Make sure you have a PostGIS database and W&B set up.
- Create a `.env` based on `.env.template` before running any pipeline steps.

## Repository Structure
- `cocoa_mapping/paths.py`: central place where paths to all datasets are defined.
- `cocoa_mapping/image_downloader` and `cocoa_mapping/aef_embeddings_downloader`: source download functions for Sentinel-2 imagery and AEF embeddings. The implemenation of AEF downloader is not provided because the experiments used cached AlphaEarth embeddings. To work with AEF, pre-download embeddings, cache them, and adapt the downloader to your cache layout.
- `cocoa_mapping/training_data`: builds source training data. It downloads Sentinel-2 and AEF images via the modules above, converts cocoa probability maps from Kalitschek et al. (2023) into pseudo-labels (expected at `data/kalitschek_probs.tif`), and writes HDF5 files with image/label pairs for source-region training. The probability maps can be downloaded from https://www.research-collection.ethz.ch/entities/researchdata/fa059526-e934-4fff-80b9-7d212827b76a and are published under CC BY 4.0.
- `cocoa_mapping/kalitschek_training`: trains on the source region. Uses models from `cocoa_mapping/models` and Hydra configs to choose architectures and hyperparameters. Metrics and best models are logged to Weights & Biases; set `WANDB_ENTITY` in your `.env`. If you rely on `cocoa_mapping/models/canopy_height_pretrained`, run `cocoa_mapping/models/canopy_height_pretrained/scripts/get_pretrained_weights_and_train_stats.py` beforehand to fetch weights and stats.
- `cocoa_mapping/finetuning_data`: downloads data for fine-tuning on the target region (Nigeria) using the same download utilities as source training. The `cocoa_mapping/finetuning_data/compute_and_upload_clusters.py` script performs clustering, downloads imagery, and uploads the dataset (with a `cluster_id` column) to the provided `--output-table-name`.
- `cocoa_mapping/finetuning`: fine-tunes source-trained models on the target region; mirrors the source training flow.
- `cocoa_mapping/finetuning/configs/dataset`: datasets used by finetuning. Each config points to a `table_name` (label masks; matches the `--output-table-name` above) and a `training_data_dir` containing subdirectories named by `cluster_id`, each holding GeoTIFF imagery inputs.
- `cocoa_mapping/evaluation`: evaluates models. Requires evaluation datasets to live in a PostgreSQL database. Set `GEO_DB_HOST`, `GEO_DB_NAME`, `GEO_DB_USER`, `GEO_DB_PASSWORD`, and `GEO_DB_PORT` as in `.env.template`, and upload datasets to the table names specified in `cocoa_mapping/evaluation/evaluation_setups.py`. Uses image chunkers from `cocoa_mapping/image_chunker`.
- `cocoa_mapping/utils`: shared utilities used across the codebase.


## Database Setup and Tables
- Create a PostgreSQL database with PostGIS enabled, then create the schema used by the code:
  - `CREATE EXTENSION postgis;`
  - `CREATE SCHEMA IF NOT EXISTS cocoa_data;`
  - Set `GEO_DB_HOST`, `GEO_DB_NAME`, `GEO_DB_USER`, `GEO_DB_PASSWORD`, and `GEO_DB_PORT` in `.env` (see `.env.template`), then load it via `cocoa_mapping.utils.general_utils.load_env_file`.
- Conventions expected by `cocoa_mapping.utils.db_utils.get_full_table`:
  - Geometry column named `geometry` stored as WKB with SRID 4326; We assume CRS of `EPSG:4326` for all vector datasets.
- Evaluation tables (used by `cocoa_mapping/evaluation/evaluation_setups.py` and `evaluation_utils.py`), all in schema `cocoa_data`:
  - `sentinel_2_test_tiles`: polygons of Sentinel-2 tiles with columns `Name` (tile code), `country`, and `geometry`. If missing, the code falls back to `auxiliary_data/sentinel_2_test_tiles.geojson`.
  - Country datasets for evaluation with the names defined in `cocoa_mapping/evaluation/evaluation_setups.py` for each country, e.g.  `cote_divoire_unified_dataset_v1`, `ghana_unified_dataset_v1`, `nigeria_unified_dataset_v2`, `cameroon_unified_dataset_v1` for `thesis` setup. Required columns: `geometry` (Point or Polygon), `year` (int), `cocoa` (bool), `label` (string). Extra columns are ignored. The table names should be the same as in the evaluation setup 
- Fine-tuning / training tables (see `cocoa_mapping/finetuning/configs/dataset/nigeria.yaml` and `finetune_utils.py`):
  - `cluster_id` is generated automatically by `cocoa_mapping/finetuning_data/compute_and_upload_clusters.py` (users should not set it manually). The script writes two tables: annotations with `cluster_id` and a cluster-geometry table of bounding boxes.
  - The resulting training val samples table, e.g. `nigeria_train_val_samples_v2`, must include `cluster_id`, `label`, `geometry`, and `year`. The downloader assumes cluster folders are named `cluster_<cluster_id>` inside the training data directory configured in `cocoa_mapping/finetuning/configs/dataset`.
  - Cluster geometry tables (e.g., `nigeria_train_val_clusters_v2`) contain `cluster_id`, `geometry` (polygons), and `year`. These feed the imagery download scripts in `cocoa_mapping/finetuning_data/dataset_downloaders`.
- Example workflow to seed evaluation tables from local annotations:
  1. Prepare a GeoJSON with `geometry`, `year`, `label`, and `cocoa` columns.
  2. Upload evaluation datasets and test tiles with `geopandas.GeoDataFrame.to_postgis(table_name, engine, schema="cocoa_data", if_exists="replace")` (or use `upload_table_to_db`), keeping column names/types as above.

## Fine-tuning Data Download Workflow
1) Add cluster IDs and upload tables (required before any fine-tuning downloads), for example:
   - `python -m cocoa_mapping.finetuning_data.compute_and_upload_clusters --annotations-geojson <path_to_annotations.geojson> --output-table-name nigeria_train_val_samples_v2 --output-cluster-table-name nigeria_train_val_clusters_v2 --schema cocoa_data`
   - This script computes clusters, adds `cluster_id` to your annotations, and uploads both tables to PostGIS. It requires the `.env` database settings loaded via `load_env_file`.
2) Download Sentinel-2 data for each cluster, for example:
   - `python -m cocoa_mapping.finetuning_data.dataset_downloaders.download_sentinel_2_data_for_cluster_table --cluster-table nigeria_train_val_clusters_v2 --output-dir <local_output_dir> --schema cocoa_data --num-scenes 5`
   - Adjust `--num-scenes`, `--max-processes`, and `--download-workers-per-process` as needed. A `clusters.geojson` manifest is written under `<local_output_dir>`.
3) (Optional, if AEF support is implemented locally) Download AEF data, for example:
   - `python -m cocoa_mapping.finetuning_data.dataset_downloaders.download_data_for_cluster_table --cluster-table nigeria_train_val_clusters_v2 --output-dir <local_output_dir_aef> --schema cocoa_data --image-type aef`
4) Point finetuning configs to the downloaded data, for example:
   - In `cocoa_mapping/finetuning/configs/dataset/nigeria.yaml`, set `training_data_dir` values to the folders containing `cluster_<cluster_id>` subdirectories produced by the downloaders. The training code uses `cluster_id` to map rows to these directories.
