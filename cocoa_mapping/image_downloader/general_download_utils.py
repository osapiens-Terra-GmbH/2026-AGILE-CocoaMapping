from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from typing import Callable, Literal
import uuid
import logging

import boto3
from botocore.utils import ClientError
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_file_streaming(url: str,
                            output_path: str,
                            use_progress_callback: bool = True,
                            progress_callback: Callable[[int], None] | None = None,
                            chunk_size: int | None = 8 * 1024 * 1024):
    """
    Downloads a file with streaming and optional progress callback.
    Uses a temporary filename and renames on success; cleans up on failure.

    Note: Setting chunk_size to None sometimes leads to the whole file being downloaded as one chunk, even if it's many GBs large.

    Parameters:
        url(str): URL of the file to download.
        output_path(str): Final path where the file should be saved.
        use_progress_callback(bool): Whether to use a progress callback.
        progress_callback(callable, optional): If provided, it will be called with bytes downloaded. Useful for progress bars.
        chunk_size(int, optional): Chunk size in bytes. None means it will read data as it arrives in whatever size the chunks are received.
            Default is 8MB which is a good value for large files.
    """
    if use_progress_callback and progress_callback is None:
        progress_callback = TransferProgressCallback(paths=[url])

    temp_path = f"{output_path}.{uuid.uuid4().hex}.part"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if use_progress_callback:
                        progress_callback(len(chunk))

        os.replace(temp_path, output_path)  # atomic rename
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Always remove the temp file


def download_file_from_s3(s3_path: str,
                          local_path: str,
                          verbose: bool = True,
                          raise_error: bool = False,
                          use_progress_callback: bool = False,
                          progress_callback: "TransferProgressCallback" = None) -> str | None:
    """Download the file from the specified S3 location to the local path.

    Args:
        s3_path (str): The S3 path to the file (e.g., 's3://bucket/key').
        local_path (str): The local path to save the downloaded file.
        verbose (bool): Whether to log detailed errors.
        raise_error (bool): If true, raise an error if the download fails instead of returning None.
        use_progress_callback (bool): If true, use the progress callback to print the progress of the download.
        progress_callback (TransferProgressCallback): If provided, this callback will be used. Useful if you run the function in parallel and want them to report to the same progress bar.

    Returns:
        The local path of the downloaded file or None if the download failed and raise_error is False.

    Raises:
        ClientError: If the download fails and raise_error is True.
    """
    bucket, key = get_bucket_and_file_key(s3_path)

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

    s3_client = boto3.client('s3')
    try:
        progress_callback = (progress_callback or TransferProgressCallback(s3_path)) if use_progress_callback else None
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path, Callback=progress_callback)
    except ClientError:
        if raise_error:
            raise
        if verbose:
            logger.error(f"Failed to download {bucket}/{key}", exc_info=True)
        return None
    logger.debug(f"Downloaded {bucket}/{key} to {local_path}")
    return local_path


def get_bucket_and_file_key(s3_path: str) -> tuple[str, str]:
    """Extract the bucket and file key from the full S3 path."""
    assert s3_path.startswith('s3://'), f"Invalid S3 path: {s3_path}"

    bucket = s3_path.split('/')[2]
    file_key = '/'.join(s3_path.split('/')[3:])
    return bucket, file_key


class TransferProgressCallback:
    """This is a callback function that is used to print the progress for upload/download.
    It supports uploading local paths and downloading from S3 or HTTP.
    """

    def __init__(self,
                 paths: str | list[str] | None,
                 **tqdm_kwargs):
        """Initialize the callback.

        Args:
            paths (str | list[str] | None): The path(s) to the file(s) being uploaded/downloaded.
                If not given, we can not track the progress in percentage.
            **tqdm_kwargs: Additional keyword arguments to pass to tqdm.
        """
        self._lock = threading.Lock()
        if paths:
            paths = [paths] if isinstance(paths, str) else paths
            assert len(paths) > 0, "No paths provided"
            path_type = _determine_path_type(paths[0])
            size = self._get_paths_size(paths, path_type=path_type)

            word = 'Downloading' if path_type in ['s3', 'http'] else 'Uploading'
            paths_str = ', '.join([os.path.basename(path) for path in paths])
            desc = f"{word} {_truncate_text(paths_str, 50)}"
        else:
            desc = 'File Transfer'
            size = None

        self._pbar = tqdm(**{
            "total": size or None,
            "unit": 'B',
            "unit_scale": True,
            "desc": desc,
            **tqdm_kwargs
        })

    def __call__(self, bytes_amount: int):
        """Update the progress bar with the given number of bytes."""
        # We use lock as this is could be called from multiple threads.
        with self._lock:
            self._pbar.update(bytes_amount)

    def _get_paths_size(self, paths: list[str], path_type: Literal['s3', 'http', 'local']):
        """Get the total size of all the paths."""
        # For local or single path, do not use a thread pool executor.
        if path_type == 'local' or len(paths) == 1:
            return sum([get_path_size(path, path_type) for path in paths], 0)

        # If we have multiple paths and they are not local, use a thread pool executor.
        max_workers = min(8, len(paths))  # Do not use more workers than paths to get the size
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_path_size,
                                       path=path,
                                       path_type=path_type
                                       ) for path in paths]
            try:
                return sum([future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating total size")], 0)
            except Exception as e:
                logger.error("Error calculating total size. Shutting down executor.", exc_info=True)
                executor.shutdown(wait=False, cancel_futures=True)
                raise e

    def update_desc(self, desc: str):
        """Update the description of the progress bar."""
        with self._lock:
            self._pbar.set_description(desc, refresh=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self._pbar:
            self._pbar.close()


def _determine_path_type(path: str) -> Literal['s3', 'http', 'local']:
    """Determine if this is s3, http, or local path based on the prefix."""
    if path.startswith('s3://'):
        return 's3'
    if path.startswith('http'):
        return 'http'
    if os.path.exists(path):
        return 'local'
    raise ValueError(f"Unknown file type: {path}. Looks like local path, but it does not exist.")


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length and add ellipsis if it is longer."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_path_size(path: str, path_type: Literal['s3', 'http', 'local']) -> int:
    """Get the size of a path in bytes.

    Args:
        path: The path to get the size of.
        path_type: The type of the path.

    Returns:
        int: The size of the path in bytes or 0 if could not be determined.
    """
    if path_type == 's3':
        return get_s3_file_size(path)
    if path_type == 'http':
        return get_http_path_size(path)
    if path_type == 'local':
        return get_local_path_size(path)
    raise ValueError(f"Unknown path type: {path_type}")


def get_s3_file_size(s3_path: str, verbose: bool = True, raise_error: bool = True) -> int | None:
    """Get the size of a file in S3 in bytes.

    Args:
        s3_path: The S3 path to get the size of.
        verbose: Whether to log errors.
        raise_error: Whether to raise an error if the size could not be determined.

    Returns:
        int: The size of the file in S3 or None if the size could not be determined and raise_error is False.

    Raises:
        ClientError: If the size could not be determined and raise_error is True.
    """
    assert not s3_path.endswith('/'), f"S3 path is a directory: {s3_path}. Remove the '/' at the end of the path."

    try:
        bucket, key = get_bucket_and_file_key(s3_path)
        s3_client = boto3.client('s3')
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response['ContentLength']
    except ClientError:
        if raise_error:
            raise
        if verbose:
            logger.error(f"Error getting size of file: {s3_path}", exc_info=True)
        return None


def get_local_path_size(path: str, _may_be_symlink: bool = True) -> int:
    """Get the size of a directory or file or symlink in bytes.

    Args:
        path: The local path to get the size of.
        _may_be_symlink (bool): Private parameter (do not use). If false, we assume the path is not a symlink.
            Used to avoid infinite recursion, e.g. for circular symlinks.

    Returns:
        int: The size of the path in bytes or 0 if could not be determined.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    if os.path.isdir(path):
        return get_dir_size(path)
    if os.path.islink(path):
        assert not _may_be_symlink, f"Arrived at symlink after replacing symlinks. Please check the path: {path}"
        return get_local_path_size(os.path.realpath(path), _may_be_symlink=False)

    logger.warning(f"Unknown file type: {path}. Will not be counted in the size.")
    return 0


def get_dir_size(path: str) -> int:
    """Get the size of a directory in bytes.

    Args:
        path: The directorypath to get the size of.

    Returns:
        int: The size of the directory in bytes or 0 if no items inside could not be determined.
    """
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
            else:
                logger.warning(f"Unknown file type: {entry.path}. Will not be counted in the size.")
    return total


def get_http_path_size(path: str) -> int:
    """Get the size of a HTTP(s) file in bytes using the HEAD request.

    Args:
        path: The HTTP(s) path to get the size of.

    Returns:
        int: The size of the HTTP(s) file in bytes or 0 if could not be determined (not all servers support HEAD requests)
    """
    try:
        response = requests.head(path, timeout=10)  # 10 seconds is more than enough for a HEAD request.
        response.raise_for_status()
        return int(response.headers['Content-Length'])
    except Exception:
        logger.warning(f"Could not get the size of the HTTP path: {path}. Will not be counted in the size.")
        return 0
