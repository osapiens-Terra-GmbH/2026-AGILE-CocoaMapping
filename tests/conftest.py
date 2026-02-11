import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def sentinel_tif_path(tmp_path) -> str:
    """Create a synthetic Sentinel-2 GeoTIFF used by tests."""
    path = tmp_path / "sentinel.tif"
    transform = from_origin(0, 40, 10, 10)
    data = np.stack(
        [
            np.full((4, 4), fill_value=100, dtype=np.int16),
            np.full((4, 4), fill_value=200, dtype=np.int16),
            np.full((4, 4), fill_value=300, dtype=np.int16),
            np.array(
                [
                    [4, 4, 8, 8],
                    [4, 3, 3, 4],
                    [4, 4, 4, 4],
                    [1, 1, 4, 4],
                ],
                dtype=np.uint8,
            ),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=4,
        dtype=data.dtype,
        crs="EPSG:32630",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data.astype(dst.profile["dtype"]))
    return str(path.resolve())


@pytest.fixture
def aef_tif_path(tmp_path) -> str:
    """Create a synthetic AEF GeoTIFF with nodata regions."""
    path = tmp_path / "aef.tif"
    transform = from_origin(0, 40, 10, 10)
    data = np.stack(
        [
            np.array(
                [
                    [0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [0, 0, 2, 2],
                    [0, 2, 2, 0],
                    [0, 2, 2, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype=data.dtype,
        crs="EPSG:32630",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data.astype(dst.profile["dtype"]))
    return str(path.resolve())


@pytest.fixture
def kalitschek_tif_path(tmp_path) -> str:
    """Create a synthetic Kalitschek label GeoTIFF with background value 3."""
    path = tmp_path / "kalitschek.tif"
    transform = from_origin(0, 40, 10, 10)
    data = np.array(
        [
            [3, 3, 1, 1],
            [3, 1, 1, 3],
            [3, 2, 2, 3],
            [3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32630",
        transform=transform,
        nodata=3,
    ) as dst:
        dst.write(data[np.newaxis, :, :])
    return str(path.resolve())
