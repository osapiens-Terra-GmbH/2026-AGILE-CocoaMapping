import geopandas as gpd


def download_aef_for_sentinel_2_tile(
    grid_code: str,
    year: int,
    output_path: str,
    delete_input: bool = True,
    use_progress_callback: bool = True,
    max_download_workers: int = 8,
    overwrite: bool = False
) -> str:
    """Download AlphaEarth embeddings for a Sentinel-2 MGRS tile and export a single GeoTIFF mosaic.

    Args:
        grid_code: Sentinel-2 MGRS grid code (e.g. ``29NQJ``).
        year: Year to download.
        output_path: Path to save the downloaded AEF mosaic.
        delete_input: If ``True``, delete raw AlphaEarth tiles cached locally after cropping.
        use_progress_callback: Whether to show a tqdm progress bar while downloading tiles.
        max_download_workers: Number of workers to use for downloading AEF tiles.
        overwrite: If ``True``, overwrite the existing output path if it exists.

    Returns:
        Path to the AEF mosaic.

    Raises:
        ValueError: If the grid code is invalid or not present in the Sentinel-2 grid.
        RuntimeError: If no AlphaEarth tiles were downloaded for the requested year.
    """
    raise NotImplementedError((
        "In the study, we used cached AlphaEarth embeddings that we pre-downloaded. "
        "If you want to experiment with AEF embeddings, you can pre-download the embeddings, cache them somewhere, and implement this function based on how you cached them."
    ))


def download_aef_data(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    max_threads_number: int,
) -> gpd.GeoDataFrame:
    """
    Download AEF data.

    This function:
    1. Downloads AEF embeddings for each geometry in parallel
    2. Saves GeoDataFrame with tiff_file column pointing to exported files
    3. Returns the GeoDataFrame with the tiff_file column

    Args:
        gdf: GeoDataFrame row containing 'geometry' and 'id' columns
        output_dir: Directory to save the downloaded data and annotations
        max_threads_number: Maximum number of threads for parallel download.

    Returns:
        GeoDataFrame with the tiff_file column
    """
    raise NotImplementedError((
        "In the study, we used cached AlphaEarth embeddings that we pre-downloaded. "
        "If you want to experiment with AEF embeddings, you can pre-download the embeddings, cache them somewhere, and implement this function based on how you cached them."
    ))
