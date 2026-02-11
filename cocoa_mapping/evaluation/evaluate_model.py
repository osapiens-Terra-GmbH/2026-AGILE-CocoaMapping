import argparse
import os
from typing import Literal, Optional

import geopandas as gpd
import pandas as pd
import shutil
from shapely import box
import wandb
import logging

from cocoa_mapping.evaluation.evaluation_setups import ALL_SUPPORTED_COUNTRIES, EVALUATION_SETUPS_REGISTRY, get_evaluation_setup
from cocoa_mapping.evaluation.evaluation_metrics import ALL_METRICS_TYPES, EvaluationMetrics
from cocoa_mapping.evaluation.evaluation_utils import default_imagery_dir, get_country_test_dataset, get_two_closest, test_probs
from cocoa_mapping.aef_embeddings_downloader.aef_embeddings_downloader import download_aef_for_sentinel_2_tile
from cocoa_mapping.evaluation.prediction_utils import predict_multipolygon_paths, predict_paths
from cocoa_mapping.image_downloader.image_downloader import download_and_consolidate_tile
from cocoa_mapping.models.models_preprocessors_registry import load_model, load_model_preprocessor_configs_for_logging, load_preprocessor
from cocoa_mapping.paths import Paths
from cocoa_mapping.utils.logging_utils import flatten_wandb_metrics, get_annotation_distribution
from cocoa_mapping.utils.general_utils import load_env_file
from cocoa_mapping.utils.training_utils import get_device
from cocoa_mapping.models.model_utils import download_model_if_not_exist


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_model(run_name: str,
                   model_name: str,
                   countries: list[str] | str,
                   aef_mode: bool,
                   prob_threshold: float = 0.5,
                   metrics_types: str | list[str] | Literal['all'] = 'all',
                   debug: bool = False,
                   predict_num_scenes: int = 1,
                   num_scenes: int = 5,
                   batch_size: int = 256,
                   delete_input: bool = False,
                   evaluation_setup: str = 'thesis',
                   run_project: str = 'cocoa-mapping-test',
                   model_project: Optional[str] = None,
                   same_year: Optional[int] = None,
                   predict_full_tiles: bool = True):
    """Evaluate a model on a given set of countries and dataset versions.

    Args:
        run_name: The name of the run.
        model_name: The name of the model to evaluate, i.e. wandb run name to which the model was uploaded.
        countries: The countries to evaluate.
        aef_mode: Whether to use AEF model for prediction.
        prob_threshold: The probability threshold for prediction.
        metrics_types: The metrics types to evaluate. If 'all', all metrics will be evaluated.
        debug: Whether to run in debug mode.
        predict_num_scenes: The number of scenes to predict.
        num_scenes: The number of scenes to download. More scenes increase probability that cloud free patch will be found.
        batch_size: The batch size for prediction.
        delete_input: Whether to delete input images after prediction.
        evaluation_setup: The evaluation setup to use, e.g. 'thesis', 'cameroon'.  This determines which tables to use for the evaluation.
        run_project: The project name for this run.
        model_project: The project name to which the model was uploaded. If not provided, all projects will be searched till the model is found.
        same_year: If set, the script will assume all annotations are from the given year.
        predict_full_tiles: Whether to use full files for prediction.
            If set, the script will use the full files for prediction, predictions will take longer but it will produce output for the whole tile.
            If not set, the script will only predict the samples.
    """
    if isinstance(countries, str):
        countries = [countries]

    # Validate countries
    for country in countries:
        if not get_evaluation_setup(evaluation_setup).support_country(country):
            raise ValueError(
                f"Evaluation setup {evaluation_setup} does not support country {country}. Please choose from: {get_evaluation_setup(evaluation_setup).countries.keys()}")

    # Get model and preprocessor
    model_path = download_model_if_not_exist(model_name=model_name,
                                             project_name=model_project,
                                             models_dir=Paths.MODELS_DIR.value)
    model = load_model(model_path).to(get_device()).eval()
    preprocessor = load_preprocessor(model_path)

    # Get config
    config = {
        'model': model_name,
        'model_project': model_project,
        "aef": aef_mode,
        **load_model_preprocessor_configs_for_logging(model_path),
        'prob_threshold': prob_threshold,
        'debug': debug,
        'predict_num_scenes': predict_num_scenes,
        'num_scenes': num_scenes,
        'evaluation_setup': evaluation_setup,
        'same_year': same_year,
    }

    # Get some settings
    wandb.init(project=run_project,
               name=run_name,
               reinit=True,
               config=config
               )

    # Create directories
    output_dir = f"{Paths.TEST_RESULTS_DIR.value}/{run_name}"
    input_dir = Paths.TEST_INPUT_DIR.value
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    # Iterate over countries and compute metrics for each country
    for country in countries:
        country_test_samples, current_dataset_version = get_country_test_dataset(country, eval_setup_name=evaluation_setup)
        config[f'{country}_dist'] = get_annotation_distribution(country_test_samples)
        logger.info(f"Found {len(country_test_samples)} samples for {country} {current_dataset_version}")

        # Set same year if provided
        if same_year is not None:
            country_test_samples['year'] = same_year

        all_metrics: list[EvaluationMetrics] = []
        fn_gdfs = []
        fp_gdfs = []
        total_ignored_samples = 0
        n_tiles = len(country_test_samples.tile_name.unique())
        # Test each tile for each year
        for tile_i, (tile_name, tile_samples) in enumerate(country_test_samples.groupby('tile_name')):
            # Skip at second tile if debugging
            if debug and tile_i == 1:
                continue

            # Test each year for this tile
            n_years = len(tile_samples.year.unique())
            for year_i, (year, current_samples) in enumerate(tile_samples.groupby('year')):
                # Stop at second year if debugging
                if debug and year_i == 1:
                    break

                logger.info(f'Processing {year} ({year_i + 1}/{n_years}) for tile {tile_name} ({tile_i + 1}/{n_tiles} for {country})')

                # Reduce number of samples if debugging
                if debug:
                    current_samples = get_two_closest(current_samples)

                # Download images for this year and this tile
                logger.info(f"Downloading {year} images for {tile_name} in {country}...")
                current_input_dir = default_imagery_dir(country, tile_name, year)
                image_paths = download_images(grid_code=tile_name,
                                              aef_mode=aef_mode,
                                              output_dir=current_input_dir,
                                              year=year,
                                              delete_input=delete_input,
                                              num_scenes=num_scenes)

                # Predict images for this year
                prediction_path = f"{output_dir}/{country}/predictions/{tile_name}/{year}"
                prediction_path = f"{prediction_path}.tif" if predict_full_tiles else prediction_path
                logger.info(f"Predicting tile {tile_name} {year} to {prediction_path}...")
                if predict_full_tiles:
                    predict_paths(model=model,
                                  preprocessor=preprocessor,
                                  image_paths=image_paths,
                                  dataset_type='tif',
                                  image_type='aef' if aef_mode else 'sentinel_2',
                                  output_path=prediction_path,
                                  predict_num_scenes=predict_num_scenes,
                                  polygon=box(*current_samples.total_bounds),
                                  batch_size=batch_size
                                  )
                else:
                    prediction_paths = predict_multipolygon_paths(model=model,
                                                                  preprocessor=preprocessor,
                                                                  gdf=current_samples,
                                                                  image_paths=image_paths,
                                                                  dataset_type='tif',
                                                                  output_dir=prediction_path,
                                                                  image_type='aef' if aef_mode else 'sentinel_2',
                                                                  predict_num_scenes=predict_num_scenes,
                                                                  batch_size=batch_size,
                                                                  )
                    current_samples['cocoa_prediction'] = prediction_paths

                # Test predictions
                metrics, fn_rows, fp_rows, ignored_samples = test_probs(annotations=current_samples,
                                                                        probs_path=prediction_path if predict_full_tiles else None,
                                                                        probs_path_col='cocoa_prediction' if not predict_full_tiles else None,
                                                                        threshold=prob_threshold,
                                                                        metrics_types=metrics_types)

                # Save false negatives and false positives
                if not fn_rows.empty:
                    fn_gdfs.append(fn_rows)
                if not fp_rows.empty:
                    fp_gdfs.append(fp_rows)
                all_metrics.append(metrics)
                total_ignored_samples += ignored_samples

                # Delete input images
                if delete_input:
                    shutil.rmtree(current_input_dir, ignore_errors=True)

        # Save false negatives and false positives
        if fn_gdfs:
            fn_gdfs = gpd.GeoDataFrame(pd.concat(fn_gdfs), geometry='geometry', crs=fn_gdfs[0].crs)
            fn_gdfs.to_file(f"{output_dir}/{country}/fn_rows.geojson")
        if fp_gdfs:
            fp_gdfs = gpd.GeoDataFrame(pd.concat(fp_gdfs), geometry='geometry', crs=fp_gdfs[0].crs)
            fp_gdfs.to_file(f"{output_dir}/{country}/fp_rows.geojson")

        # Merge & log metrics for this country
        total_metrics = sum(all_metrics).compute_metrics()
        total_metrics = {f"{country} {current_dataset_version}/{key}": value for key, value in total_metrics.items()}
        wandb.log(flatten_wandb_metrics(total_metrics))
        wandb.log({'ignored_samples': total_ignored_samples})


def download_images(grid_code: str,
                    aef_mode: bool,
                    output_dir: str,
                    year: int,
                    delete_input: bool | None,
                    num_scenes: int | None,
                    mininterval: float = 0.1) -> list[str]:
    """Download images for a tile and year.

    Args:
        grid_code: The grid code of the tile.
        aef_mode: Whether to use AEF model for prediction.
        output_dir: The directory to save the images.
        year: The year of the images.
        delete_input: Whether to delete the input images after prediction.
        num_scenes: The number of scenes to download, if sentinel-2 mode is used.
        mininterval: The minimum interval to show progress.

    Returns:
        List of paths to the downloaded images.
    """
    if not aef_mode:
        return download_and_consolidate_tile(grid_code=grid_code,
                                             time_interval=f'{year}-01-01/{year}-12-31',
                                             num_scenes=num_scenes,
                                             output_type='tif',
                                             output_dir=output_dir,
                                             tqdm_mininterval=mininterval)
    else:
        image_path = download_aef_for_sentinel_2_tile(grid_code=grid_code,
                                                      year=year,
                                                      output_path=os.path.join(output_dir, f"aef_{year}.tif"),
                                                      delete_input=delete_input,
                                                      use_progress_callback=True,
                                                      max_download_workers=8)
        return [image_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a given set of countries and dataset versions."
    )
    # Required positional arguments
    parser.add_argument("-r", "--run-name", type=str, required=True,
                        help="The name of the run.")
    parser.add_argument("-m", "--model-name", type=str, required=True,
                        help="The name of the model to evaluate, i.e. wandb run name to which the model was uploaded.")
    parser.add_argument("-c", "--countries", nargs="+", type=str, required=True, choices=['all'] + list(ALL_SUPPORTED_COUNTRIES),
                        help="The countries to evaluate. If 'all', all countries will be evaluated.")

    # Optional arguments
    parser.add_argument("-a", "--aef", type=int, choices=[0, 1], default=0,
                        help="Whether to use AEF model for prediction.")
    parser.add_argument("-p", "--prob-threshold", type=float, default=0.5,
                        help="The probability threshold for prediction. (default: 0.5)")
    parser.add_argument("-d", "--debug", type=int, choices=[0, 1], default=0,
                        help="Whether to run in debug mode.")
    parser.add_argument("-mt", "--metrics-types", type=str, default='all', choices=['all'] + ALL_METRICS_TYPES,
                        help="The metrics types to evaluate. If 'all', all metrics will be evaluated.")
    parser.add_argument("-es", "--evaluation-setup", type=str, default="thesis", choices=EVALUATION_SETUPS_REGISTRY.keys(),
                        help="The evaluation setup to use, e.g. 'thesis', 'cameroon'.  This determines which tables to use for the evaluation.")
    parser.add_argument("-rp", "--run-project", type=str, default="cocoa-mapping-test",
                        help="The project name for this run. (default: cocoa-mapping-test)")
    parser.add_argument("-mp", "--model-project", type=str, default=None,
                        help="The project name to which the model was uploaded. If not provided, all projects will be searched till the model is found.")

    # Prediction arguments
    parser.add_argument("-pn", "--predict-num-scenes", type=int, default=1,
                        help="The number of scenes to predict. (default: 1)")
    parser.add_argument("-n", "--num-scenes", type=int, default=5,
                        help="The number of scenes to download. (default: 5)")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="The batch size for prediction. (default: 256)")
    parser.add_argument("-di", "--delete-input", type=int, choices=[0, 1], default=0,
                        help="Whether to delete input images after prediction.")

    # Special experiments
    parser.add_argument('-sy', '--same-year', type=int, default=None,
                        help="If set, the script will assume all annotations are from the given year.")

    # Optimization
    parser.add_argument("-pf", "--predict-full-tiles", type=int, choices=[0, 1], default=1,
                        help=("Whether to use full files for prediction. "
                              "If set, the script will use the full files for prediction, predictions will take longer but it will produce output for the whole tile "
                              "If not set, the script will only predict the samples"
                              )
                        )

    args = parser.parse_args()

    if 'all' in args.countries:
        countries = [*get_evaluation_setup(args.evaluation_setup).countries.keys()]
    else:
        countries = args.countries

    load_env_file()

    evaluate_model(run_name=args.run_name,
                   model_name=args.model_name,
                   countries=countries,
                   aef_mode=bool(args.aef),
                   prob_threshold=args.prob_threshold,
                   debug=bool(args.debug),
                   metrics_types=args.metrics_types,
                   evaluation_setup=args.evaluation_setup,
                   run_project=args.run_project,
                   model_project=args.model_project,
                   predict_num_scenes=args.predict_num_scenes,
                   num_scenes=args.num_scenes,
                   batch_size=args.batch_size,
                   delete_input=bool(args.delete_input),
                   same_year=args.same_year,
                   predict_full_tiles=bool(args.predict_full_tiles)
                   )
