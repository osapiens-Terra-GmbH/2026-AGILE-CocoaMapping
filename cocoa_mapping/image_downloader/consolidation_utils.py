import os
import uuid
from typing import Literal, Sequence
import logging

import h5py
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from cocoa_mapping.image_downloader.bands import SENTINEL_2_BANDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def consolidate_items(band_dirs: list[str],
                      output_paths: list[str] | str,
                      output_type: Literal["tif", "hdf5"],
                      bands: list[str] = SENTINEL_2_BANDS,
                      ref_band: str = "B08",
                      resampling: Resampling = Resampling.bilinear,
                      use_progress_callback: bool = True,
                      max_consolidate_workers: int | None = None,
                      tqdm_mininterval: float = 0.1
                      ) -> list[str]:
    """Consolidate items (represented by band directories) into tif or hdf5 files.

    If the output type is "hdf5", it will have following structure:
    - image: (bands, height, width) uint16 array
    - attrs:
        - crs: The CRS of the image.
        - transform: The affine transform of the image as 6-tuple.

    Args:
        band_dirs: The paths to the band directories.
        output_paths: The paths to the output files. Should be the same length as band_dirs.
            If a single path is provided, it will be interpreted as a directory and the output files will be named after the band directories.
        output_type: The type of the output file. Should be either 'tif' or 'hdf5'.
        bands: The bands to consolidate.
        ref_band: The reference band to use as base for consolidation.
        resampling: The resampling method to use for upsampling the numeric bands.
        use_progress_callback: Whether to use a progress callback.
        max_consolidate_workers: The maximum number of workers to use for the consolidation.
            If 0, will use the current process only.
            If None, will use the number of CPUs or the number of items, whichever is smaller.
        tqdm_mininterval: The minimum interval to update the progress bar. Is useful for aws training, where every update will be on new line (so we need to update less often).

    Returns:
        output_paths: The paths to the output files.
    """
    # If a single path is provided, it will be interpreted as a directory and the output files will be named after the band directories.
    if isinstance(output_paths, str) or not isinstance(output_paths, Sequence):
        extension = '.tif' if output_type == 'tif' else '.hdf5'
        output_paths = [os.path.join(output_paths, f"{os.path.basename(band_dir)}{extension}") for band_dir in band_dirs]

    assert len(band_dirs) == len(output_paths), "Number of band directories and output paths must match."
    assert ref_band in bands, f"{ref_band} is not in {bands}"

    # Filter the output paths that already exist.
    left_output_paths, left_band_dirs = _filter_existing_output_paths(output_paths, band_dirs)
    if len(left_output_paths) == 0:
        return output_paths

    # Pick if running progress bar per band or per item.
    if use_progress_callback:
        progress_bar_type = 'per_band' if len(left_band_dirs) == 1 else 'per_item'
    else:
        progress_bar_type = 'none'

    # Compute number of workers for the consolidation.
    if max_consolidate_workers is None:
        max_consolidate_workers = os.cpu_count()

    # Single worker consolidation.
    if max_consolidate_workers == 0 or len(left_band_dirs) == 1:
        for bands_dir, output_path in tqdm(zip(left_band_dirs, left_output_paths),
                                           desc="Consolidating bands",
                                           total=len(left_band_dirs),
                                           disable=progress_bar_type != 'per_item',
                                           mininterval=tqdm_mininterval):
            try:
                consolidate_bands(input_path=bands_dir,
                                  output_path=output_path,
                                  output_type=output_type,
                                  bands=bands,
                                  ref_band=ref_band,
                                  resampling=resampling,
                                  progress_bar=progress_bar_type == 'per_band'
                                  )
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.error(f"Consolidation failed for {bands_dir} to {output_path}.", exc_info=True)

    # Multiple workers consolidation.
    else:
        # Figure out number of workers to use.
        if max_consolidate_workers is None:
            max_consolidate_workers = os.cpu_count()
        workers = min(max_consolidate_workers, len(left_band_dirs))

        # Run the consolidation in parallel.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = (executor.submit(consolidate_bands,
                                       input_path=bands_dir,
                                       output_path=output_path,
                                       output_type=output_type,
                                       bands=bands,
                                       ref_band=ref_band,
                                       resampling=resampling,
                                       progress_bar=progress_bar_type == 'per_band')
                       for bands_dir, output_path in zip(left_band_dirs, left_output_paths))

            for future in tqdm(as_completed(futures),
                               desc="Consolidating bands",
                               total=len(left_band_dirs),
                               disable=progress_bar_type != 'per_item',
                               mininterval=tqdm_mininterval):
                try:
                    future.result()
                except KeyboardInterrupt:
                    logger.error(f"Keyboard interrupt. Shutting down the executor.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except:
                    logger.error(f"Consolidation failed for one of the items.", exc_info=True)
    return output_paths


def _filter_existing_output_paths(output_paths: list[str], band_dirs: list[str]) -> tuple[list[str], list[str]]:
    """Filter the output paths that already exist.
    Returns the filtered output paths and the corresponding band directories.
    """
    new_items = [(b, o) for b, o in zip(band_dirs, output_paths) if not (os.path.exists(o) and os.path.getsize(o) > 0)]
    new_band_dirs = [b for b, _ in new_items]
    new_output_paths = [o for _, o in new_items]
    return new_output_paths, new_band_dirs


def consolidate_bands(input_path: str,
                      output_path: str,
                      output_type: Literal["tif", "hdf5"] = "tif",
                      bands: list[str] = SENTINEL_2_BANDS,
                      ref_band: str = "B08",
                      resampling: Resampling = Resampling.bilinear,
                      progress_bar: bool = False) -> str:
    """Consolidate bands into tif file.

    If the output type is "hdf5", it will have following structure:
    - image: (bands, height, width) uint16 array
    - attrs:
        - crs: The CRS of the image.
        - transform: The affine transform of the image as 6-tuple.

    Args:
        input_path: The path to the input bands downloaded from AWS.
        output_path: The path to the output tif file.
        output_type: The type of the output file, either "tif" or "hdf5".
        bands: The bands to consolidate.
        ref_band: The reference band to use as base for consolidation.
        resampling: The resampling method to use for upsampling the numeric bands.
        progress_bar: Whether to show a progress bar.

    Returns:
        The path to the output tif file.
    """
    assert ref_band in bands, f"{ref_band} is not in {bands}"

    ref_band_path = os.path.join(input_path, f"{ref_band}.tif")
    if not os.path.exists(ref_band_path):
        ref_band_path = os.path.join(input_path, f"{ref_band}.jp2")
        assert os.path.exists(ref_band_path), f"Neither {ref_band}.tif nor {ref_band}.jp2 exists in {input_path}"

    with rasterio.open(ref_band_path) as src:
        ref_profile = src.profile
        ref_shape = src.shape
        crs = src.crs
        transform = src.transform

    output_profile = ref_profile.copy()
    output_profile.update({
        "count": len(bands),
        "dtype": np.uint16,
        "compress": None,
    })

    # Create output tif. Start with a temp file and rename it to the output file.
    temp_output_path = f"{output_path}.temp-{uuid.uuid4()}"
    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
    try:
        # Choose the file opener and kwargs based on the output type
        file_opener = h5py.File if output_type == "hdf5" else rasterio.open
        kwargs = {} if output_type == "hdf5" else output_profile

        # Create the output file
        with file_opener(temp_output_path, "w", **kwargs) as dst:
            if output_type == "hdf5":
                image_dataset = dst.create_dataset("image", shape=(len(bands), *ref_shape), dtype=np.uint16)
                dst.attrs["crs"] = crs.to_string()
                dst.attrs["transform"] = transform.to_gdal()  # Store as 6-tuple
                dst.attrs["nodata"] = ref_profile["nodata"]

            # Write each band to the output file
            iterator = tqdm(bands, desc="Consolidating bands") if progress_bar else bands
            for i, band in enumerate(iterator):
                band_path = os.path.join(input_path, f"{band}.tif")
                if not os.path.exists(band_path):
                    band_path = os.path.join(input_path, f"{band}.jp2")
                    if not os.path.exists(band_path):
                        raise ValueError(f"Neither {band}.tif nor {band}.jp2 exists in {input_path}")

                with rasterio.open(band_path) as src:
                    res = src.res[0]
                    dtype = src.dtypes[0]
                    assert dtype in ["uint16", "uint8"]
                    if res == 10:
                        image = src.read(1)
                    else:
                        image = src.read(
                            1,
                            out_shape=ref_shape,
                            resampling=resampling if band != "SCL" else Resampling.nearest
                        )

                if output_type == "hdf5":
                    image_dataset[i] = image.astype(np.uint16)
                else:
                    dst.write(image, i + 1)
                del image  # Delete the image to allow gb to collect it while the next band is being processed.

    except Exception as e:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise e

    # Rename the temp file to the output file. Delete the output file if it exists.
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output_path, output_path)

    return output_path
