import os
import shutil
from typing import Literal
import logging

from pystac import Item
from rasterio.enums import Resampling
from shapely import Polygon

from cocoa_mapping.image_downloader.aws_stac_api_utils import gew_aws_items, validate_grid_code, choose_scenes_for_tile
from cocoa_mapping.image_downloader.bands import SENTINEL_2_BANDS
from cocoa_mapping.image_downloader.consolidation_utils import consolidate_items
from cocoa_mapping.image_downloader.imagery_download_utils import download_items

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_and_consolidate_tile(grid_code: str,
                                  time_interval: str,
                                  num_scenes: int,
                                  output_dir: str,
                                  output_type: Literal["tif", "hdf5"] = "tif",
                                  bands: list[str] = SENTINEL_2_BANDS,
                                  use_progress_callback: bool = True,
                                  max_download_workers: int = 8,
                                  max_consolidate_workers: int | None = None,
                                  tqdm_mininterval: float = 0.1,
                                  ) -> list[str]:
    """Download and consolidate Sentinel-2 scenes for a given grid tile.

    If the output type is 'hdf5', the file will have the following structure:
     - 'image' dataset will have shape (len(bands), *ref_shape) and dtype np.uint16.
     - 'crs' string attribute will be the crs of the reference band.
     - 'transform' 6-tuple attribute will be the transform of the reference band. Use Affine.from_gdal() to convert to Affine.

    Args:
        grid_code: The grid code of the tile to download.
        time_interval: The time interval to download the scenes from, in the format "YYYY-MM-DD/YYYY-MM-DD".
        num_scenes: The number of scenes required per each part of the tile. It will download the stac items that together cover the tile num_scenes times.
        output_dir: The directory to save the downloaded scenes to. The output files will be named after the stac item id, with the suffix .tif or .hdf5 depending on the output_type.
        output_type: The type of the output file. Should be either 'tif' or 'hdf5'.
        bands: The bands to download.
        use_progress_callback: Whether to use a progress callback.
        max_download_workers: The maximum threads to use for the download.
        max_consolidate_workers: The maximum number of workers to use for the consolidation.
            If None, will use the number of CPUs or the number of items, whichever is smaller.
            If 0, will use the current process only.
        tqdm_mininterval: The minimum interval to update the progress bar. Is useful for aws training, where every update will be on new line (so we need to update less often).

    Returns:
        List of paths to the output files.
    """
    grid_code = validate_grid_code(grid_code)
    items = gew_aws_items(
        time_interval=time_interval,
        collection="sentinel-2-l2a",
        grid_codes=[grid_code],
    )
    selected_scenes = choose_scenes_for_tile(items, grid_code=grid_code, num_scenes=num_scenes)
    return download_and_consolidate_items(
        stac_items=selected_scenes,
        output_paths=output_dir,
        output_type=output_type,
        polygon=None,
        bands=bands,
        use_progress_callback=use_progress_callback,
        max_download_workers=max_download_workers,
        max_consolidate_workers=max_consolidate_workers,
        tqdm_mininterval=tqdm_mininterval
    )


def download_and_consolidate_items(stac_items: Item | list[Item],
                                   output_paths: str | list[str],
                                   output_type: Literal["tif", "hdf5"] = "tif",
                                   bands: list[str] = SENTINEL_2_BANDS,
                                   ref_band: str = "B08",
                                   polygon: Polygon | None = None,
                                   use_progress_callback: bool = True,
                                   max_download_workers: int = 8,
                                   max_consolidate_workers: int | None = None,
                                   resampling: Resampling = Resampling.bilinear,
                                   keep_bands: bool = False,
                                   tqdm_mininterval: float = 0.1,
                                   ) -> list[str]:  # Default in tqdm
    """Download and consolidate stac items from AWS.

    If the output type is 'hdf5', the file will have the following structure:
     - 'image' dataset will have shape (len(bands), *ref_shape) and dtype np.uint16.
     - 'crs' string attribute will be the crs of the reference band.
     - 'transform' 6-tuple attribute will be the transform of the reference band. Use Affine.from_gdal() to convert to Affine.

    Args:
        stac_items: The stac items to download and consolidate.
        output_paths: The paths to the output hdf5 or tif files. Should either match the number of stac items or be a single directory path.
            If single path but multiple stac items are provided, it will be interpreted as a directory and the output files will be named after the stac item id.
        output_type: The type of the output file. Should be either 'tif' or 'hdf5'.
        bands: The bands to download and consolidate.
        ref_band: The reference band to use as base for consolidation.
        polygon: If provided, the cropped images will be downloaded to the polygon extent.
        use_progress_callback: Whether to use a progress callback.
        max_download_workers: The maximum number of workers to use if use_multithreading is True.
        max_consolidate_workers: The maximum number of workers to use for the consolidation.
            If None, will use the number of CPUs or the number of items, whichever is smaller.
            If 0, will use the current process only.
        resampling: The resampling method to use for upsampling the numeric bands.
        keep_bands: Whether to keep the bands after consolidation.
        tqdm_mininterval: The minimum interval to update the progress bar.

    Returns:
        The path to the output files.
    """
    stac_items, output_paths = _validate_args(stac_items, output_paths, output_type)

    # Get temporatiry output dirs
    bands_output_dirs = [os.path.join(os.path.dirname(output_path), f'{stac_item.id}_bands')
                         for output_path, stac_item in zip(output_paths, stac_items)]

    # Filter out items that have already been downloaded.
    all_output_paths = output_paths.copy()
    for stac_item, output_path, band_dir in zip(stac_items.copy(), all_output_paths, bands_output_dirs.copy()):
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            stac_items.remove(stac_item)
            output_paths.remove(output_path)
            bands_output_dirs.remove(band_dir)

    if not stac_items:
        return all_output_paths

    try:
        download_items(stac_items=stac_items,
                       output_dirs=bands_output_dirs,
                       polygon=polygon,
                       bands=bands,
                       use_progress_callback=use_progress_callback,
                       max_workers=max_download_workers,
                       tqdm_mininterval=tqdm_mininterval
                       )
        consolidate_items(band_dirs=bands_output_dirs,
                          output_paths=all_output_paths,
                          output_type=output_type,
                          bands=bands,
                          ref_band=ref_band,
                          resampling=resampling,
                          use_progress_callback=use_progress_callback,
                          max_consolidate_workers=max_consolidate_workers,
                          tqdm_mininterval=tqdm_mininterval)
    except:
        logger.error(f"Download or consolidation failed.", exc_info=True)
        raise
    finally:
        if not keep_bands:
            _delete_directories(bands_output_dirs)

    return all_output_paths


def _validate_args(stac_items: Item | list[Item], output_paths: str | list[str], output_type: Literal["tif", "hdf5"]) -> tuple[list[Item], list[str]]:
    """Validate stac items and output paths.

    If single output path is provided for multiple items, it will be interpreted as a directory and the output files will be named after the stac item id.
    If multiple output paths are provided for multiple items, they should match in length.

    Returns:
        Validated stac items and output paths.
    """
    if isinstance(stac_items, Item):
        stac_items = [stac_items]

    if len(stac_items) == 0:
        logger.warning("No stac items provided.")
        return []

    if isinstance(output_paths, str):
        output_paths = [output_paths]

    if len(stac_items) > 1 and len(output_paths) == 1:
        output_dir = output_paths[0]
        suffix = '.tif' if output_type == 'tif' else '.hdf5'
        output_paths = [os.path.join(output_dir, f"{stac_item.id}{suffix}") for stac_item in stac_items]

    return stac_items, output_paths


def _delete_directories(dirs: list[str]):
    """Delete the given directories."""
    for dir_ in dirs:
        if os.path.exists(dir_):
            shutil.rmtree(dir_, ignore_errors=True)
