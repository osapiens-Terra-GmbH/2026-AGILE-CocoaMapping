from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from typing import TypedDict


from pystac import Asset, Item
import rasterio
import logging

from shapely import Polygon
from tqdm import tqdm
from rasterio.mask import mask
import rasterio.session

from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs
from cocoa_mapping.image_downloader.general_download_utils import download_file_streaming, download_file_from_s3, TransferProgressCallback

from cocoa_mapping.image_downloader.bands import SENTINEL_2_BANDS, SENTINEL_2_BANDS_TO_ASSET_NAMES, \
    SENTINEL_2_BANDS_TO_RESOLUTION


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_items(stac_items: Item | list[Item],
                   output_dirs: str | list[str],
                   polygon: Polygon | None = None,
                   bands: list[str] = SENTINEL_2_BANDS,
                   use_progress_callback: bool = True,
                   max_workers: int = 8,
                   tqdm_mininterval: float = 0.1,
                   ) -> list[str]:
    """Download bands from a stac items from AWS.
    We want to download them together to distribute the download tasks evenly among workers.

    If you want to download a single item, you can still provide single item and single output directory.

    Args:
        stac_items: The stac item(s) to download the bands from.
        output_dirs: The directory(ies) to download the bands to. Should either match the number of stac items or be a single directory.
            If multiple items but only one output directory is provided, the bands will be downloaded to a subdirectory for each item, named after the item id.
        polygon: If provided, the cropped images will be downloaded using vsis3 protocol.
        bands: The names of the bands to download, e.g. ['B02', 'B03', 'B04']
        use_progress_callback: Whether to use a progress callback.
        max_workers: The number of threads to use
        tqdm_mininterval: The minimum interval to update the tqdm progress bar. Default is 0.1 seconds.

    Returns:
        The path to the output directory with the downloaded bands if a single item is provided.
        A list of paths to the output directories with the downloaded bands if multiple items are provided.
    """
    stac_items, output_dirs = _validate_args(stac_items=stac_items, output_dirs=output_dirs)

    # Gather all bands that need to be downloaded
    band_infos = _accumulate_band_infos(stac_items=stac_items, output_dirs=output_dirs, bands=bands)
    if len(band_infos) == 0:
        return output_dirs

    # Pick progress bar type
    if use_progress_callback:
        # If polygon is provided, we cannot use callback as we do not know the size of the cropped file in advance.
        progress_bar_type = 'callback' if polygon is None else 'tqdm'
    else:
        progress_bar_type = 'none'

    # Set up the progress callback.
    paths = [band_info["band_href"] for band_info in band_infos]
    progress_callback = TransferProgressCallback(paths=paths, mininterval=tqdm_mininterval) if progress_bar_type == 'callback' else None

    # Download the bands.
    n_downloaded = 0
    n_total = len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = (executor.submit(_download_band,
                                   stac_item=band_info["stac_item"],
                                   band=band_info["band"],
                                   output_dir=band_info["output_dir"],
                                   polygon=polygon,
                                   use_progress_callback=use_progress_callback and progress_bar_type == 'callback',
                                   progress_callback=progress_callback
                                   )
                   for band_info in band_infos)

        for future in tqdm(as_completed(futures),
                           desc="Downloading bands",
                           total=len(band_infos),
                           disable=progress_bar_type != 'tqdm',
                           mininterval=tqdm_mininterval):
            try:
                future.result()
                n_downloaded += 1
                if progress_callback:
                    progress_callback.update_desc(f"Downloaded {n_downloaded}/{n_total}")
            except KeyboardInterrupt:
                logger.error(f"Keyboard interrupt. Shutting down the executor.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception:
                logger.error(f"Download for one of the items failed.", exc_info=True)

    # Close the progress callback.
    if progress_callback:
        progress_callback.close()

    return output_dirs


def _validate_args(stac_items: Item | list[Item],
                   output_dirs: str | list[str]) -> tuple[list[Item], list[str]]:
    """Make sure stac_items & output_dirs are lists and match in length."""
    if isinstance(stac_items, Item):
        stac_items = [stac_items]
    if isinstance(output_dirs, str):
        output_dirs = [output_dirs]
    if len(stac_items) > 1 and len(output_dirs) == 1:
        # If single output dir is provided for multiple items, use stac item id as subdirectory.
        output_dir = output_dirs[0]
        output_dirs = [os.path.join(output_dir, stac_item.id) for stac_item in stac_items]
    return stac_items, output_dirs


class BandInfo(TypedDict):
    band: str
    band_href: str
    output_dir: str
    stac_item: Item


def _accumulate_band_infos(stac_items: list[Item],
                           output_dirs: list[str],
                           bands: list[str]) -> list[BandInfo]:
    """Gather all bands that need to be downloaded.
    Outputs list of dicts with the following keys:
        "band": The band to download, e.g. 'B02'.
        "band_href": The href of the band, e.g. 'https://storage.googleapis.com/sentinel-2-l2a/30/Q/Z/2020/B02.jp2'.
        "output_dir": The directory to save the downloaded band to.
        "stac_item": The stac item to download the band from.
    """
    band_infos: list[BandInfo] = []
    for stac_item, item_dir in zip(stac_items, output_dirs):
        # Create the item directory and save the stac item there.
        os.makedirs(item_dir, exist_ok=True)
        with open(os.path.join(item_dir, "stac_item.json"), "w") as f:
            json.dump(stac_item.to_dict(), f)

        # Get the output paths for the bands and check if they exist and are not empty.
        output_paths = [_get_output_path(stac_item, band, item_dir) for band in bands]
        bands_to_download = [band for band, output_path in zip(bands, output_paths) if not os.path.exists(output_path) or os.path.getsize(output_path) == 0]
        band_infos.extend({
            "band": band,
            "band_href": _get_band_href(stac_item, band),
            "output_dir": item_dir,
            "stac_item": stac_item,
        } for band in bands_to_download)
    return band_infos


def _download_band(stac_item: Item,
                   band: str,
                   output_dir: str,
                   polygon: Polygon | None = None,
                   use_progress_callback: bool = True,
                   progress_callback: TransferProgressCallback = None,
                   max_attempts: int = 10
                   ) -> str:
    """Download a band from a stac item from AWS.

    Args:
        stac_item: The stac item to download the band from.
        band: The band to download, e.g. 'B02'.
        output_dir: The directory to save the downloaded band to.
        polygon: If provided, the cropped image will be downloaded using vsis3 protocol.
        use_progress_callback: Whether to use a progress callback.
        progress_callback: The progress callback to use.

    Returns:
        The path to the downloaded band.
    """
    # Check if the band is already downloaded
    asset_name = SENTINEL_2_BANDS_TO_ASSET_NAMES[band]
    asset = stac_item.assets[asset_name]
    output_path = _get_output_path(stac_item, band, output_dir)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    # Case I: If polygon is provided, we need to use vsis3 protocol to download the cropped image.
    for attempt in range(max_attempts):
        try:
            os.makedirs(output_dir, exist_ok=True)
            if polygon is not None:
                vsis3_link = to_vsis3_uri(asset, "sentinel-2-l2a")
                requester_pays = bool(int(os.getenv("S3_REQUESTER_PAYS", "0")))
                session = rasterio.session.AWSSession(requester_pays=requester_pays)
                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                    GDAL_HTTP_MAX_RETRY='5',
                    GDAL_HTTP_RETRY_DELAY='1',
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
                    session=session
                ):
                    with rasterio.open(vsis3_link, 'r') as src:
                        polygon_local = transform_geom_to_crs(polygon, 'EPSG:4326', src.crs)
                        out_image, out_transform = mask(src, [polygon_local], crop=True, filled=False, all_touched=True)
                        with rasterio.open(output_path, "w", **{
                            **src.profile,
                            "driver": "GTiff",
                            "transform": out_transform,
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                        }) as dst:
                            dst.write(out_image)
                    return output_path

            # Case II: If s3 link, use s3 download
            if asset.href.startswith("s3://"):
                path = download_file_from_s3(s3_path=asset.href,
                                             local_path=output_path,
                                             use_progress_callback=use_progress_callback,
                                             progress_callback=progress_callback
                                             )
                if path is None:
                    raise ValueError(f"Failed to download {asset.href} to {output_path}")
                return output_path

            # Case III: Otherwise, use http download
            download_file_streaming(url=asset.href,
                                    output_path=output_path,
                                    use_progress_callback=use_progress_callback,
                                    progress_callback=progress_callback
                                    )
            return output_path
        except:
            if attempt == max_attempts - 1:
                logger.error(f"Final attempt ({attempt + 1}) failed. Giving up.", exc_info=True)
                raise
            waiting_time = min(2 ** (attempt + 1), 60)  # 2, 4, 8, 16s seconds after 1st, 2nd etc. attempt, but cap at 60 seconds
            logger.error(f"Download for {asset.href} failed. Attempt {attempt + 1} of {max_attempts}. Waiting {waiting_time} seconds before retrying...", exc_info=True)
            time.sleep(waiting_time)

    raise RuntimeError(f"Download for {asset.href} failed after {max_attempts} attempts.")


def to_vsis3_uri(asset_obj: Asset, collection_id: str) -> str:
    """Convert an asset's HTTP/S3 href into a `/ vsis3/...` URI.

    This utility takes a STAC `Asset` object and rewrites its href to a GDAL-compatible
    `/vsis3 /` path, based on the STAC collection it belongs to.

    For Sentinel collections with `https: // ...amazonaws.com/...` style URLs, the HTTPS
    prefix is stripped and the domain is parsed to retrieve the bucket name.
    For Landsat Collection 2 assets, the embedded alternate S3 link is used directly.

    Args:
        asset_obj(Asset): STAC asset whose `href` or `extra_fields` will be used.
        collection_id(str): Collection identifier(e.g. ``"sentinel-2-l2a"``,
            ``"sentinel-1-grd"``, ``"landsat-c2l2-sr"``).

    Returns:
        str: Rewritten URI using the ``/ vsis3/<bucket > / < path >`` scheme.

    Raises:
        ValueError: If the collection is not supported.
    """

    if collection_id == "sentinel-2-l2a" or collection_id == "sentinel-1-grd":
        url = asset_obj.href

        if url.endswith(".jp2"):
            assert url.startswith("s3://"), f"The jp2 asset {url} is not a s3 link"
            band_name = url.rsplit("/", 1)[1].rsplit(".", 1)[0]
            resolution = SENTINEL_2_BANDS_TO_RESOLUTION[band_name]
            band_dir = f"R{resolution}m"
            if band_dir not in url:
                url = url.replace(band_name, f"{band_dir}/{band_name}")
            return url.replace(f"s3://", "/vsis3/")

        if url.startswith("https://"):
            url = url[len("https://"):]
            parts = url.split("/", 1)
            domain = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            bucket = domain.split(".")[0]  # everything before `.s3`
            return f"/vsis3/{bucket}/{path}"

        if url.startswith("s3://"):
            return url.replace("s3://", "/vsis3/")

    if collection_id == "sentinel-2-l1c":
        url = asset_obj.href
        assert url.startswith("s3://"), "The href is not a s3 link"
        assert url.endswith(".jp2"), "The href is not a jp2 link as expected from sentinel-2-l1c collection"
        return url.replace(f"s3://", "/vsis3/")

    if collection_id == "landsat-c2l2-sr":
        s3_uri = asset_obj.extra_fields['alternate']['s3']['href']
        return s3_uri.replace("s3://", "/vsis3/")

    raise ValueError(f"Unsupported collection_id: {collection_id}")


def _get_band_href(stac_item: Item, band: str) -> str:
    """Get the href of a band from a stac item."""
    asset_name = SENTINEL_2_BANDS_TO_ASSET_NAMES[band]
    return stac_item.assets[asset_name].href


def _get_output_path(stac_item: Item, band: str, output_dir: str) -> str:
    """Compute the output path for a band from a stac item.
    It is <output_dir>/<band>.<extension>, where extension is the extension of the band href."""
    band_href = _get_band_href(stac_item, band)  # Need it to get the extension
    extension = band_href.rsplit(".", 1)[-1]
    return os.path.join(output_dir, f"{band}.{extension}")
