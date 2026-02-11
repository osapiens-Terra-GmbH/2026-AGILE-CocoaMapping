import math
from itertools import cycle, repeat
from typing import Literal, Any

import numpy as np
from affine import Affine
from rasterio.transform import array_bounds
from shapely.geometry.base import BaseGeometry


def compute_combined_transform(transforms: list[Affine],
                               heights: list[int],
                               widths: list[int],
                               mode: Literal['union', 'intersection'] = 'union') -> tuple[Affine, int, int]:
    """Compute the combined transform, height, and width of multiple tif or hdf5 files.
    Basically, computes what is the extend of all provided images together

    Args:
        transforms: The transforms of the images.
        heights: The heights of the images.
        widths: The widths of the images.
        mode: The mode to use for the computation.
            'union': The combined transform is the union of all transforms.
            'intersection': The combined transform is the intersection of all transforms.

    Returns:
        A tuple of (transform, height, width).
    """
    assert len(transforms) == len(heights) == len(widths), \
        f"Expected the same number of transforms, heights, and widths, but got {len(transforms)}, {len(heights)}, and {len(widths)}"

    min_x, min_y, max_x, max_y = None, None, None, None
    x_scale, y_scale = None, None
    for transform, height, width in zip(transforms, heights, widths):
        # Extract the scale from the transform
        if x_scale is None:
            # We do not care about the sign of the y_scale (normally negative), as we will fit it directly to Affine.scale
            x_scale, y_scale = transform.a, transform.e
        else:
            assert x_scale == transform.a and y_scale == transform.e, f"Expected images with same resolution and orientation, but {x_scale} != {transform.a} or {y_scale} != {transform.e}"

        # Extract the bounds of the current image
        curr_min_x, curr_min_y, curr_max_x, curr_max_y = array_bounds(height, width, transform)

        # Choose selection of bounds
        min_sel = min if mode == 'union' else max
        max_sel = max if mode == 'union' else min

        # Update the bounds of the overall image
        min_x = min_sel(min_x, curr_min_x) if min_x is not None else curr_min_x
        min_y = min_sel(min_y, curr_min_y) if min_y is not None else curr_min_y
        max_x = max_sel(max_x, curr_max_x) if max_x is not None else curr_max_x
        max_y = max_sel(max_y, curr_max_y) if max_y is not None else curr_max_y

    # Validate results
    if mode == 'intersection' and (max_x <= min_x or max_y <= min_y):
        raise ValueError('Seems like the images do not overlap.')
    assert max_x > min_x and max_y > min_y, 'Should not happen in union mode.'

    # Compute the output transform
    # First, figure out what is origin. Origin is the point is the point that scale is directed from.
    origin_x = min_x if x_scale > 0 else max_x
    origin_y = min_y if y_scale > 0 else max_y

    # Define transform as move to origin and then scale
    combined_transform = Affine.translation(origin_x, origin_y) * Affine.scale(x_scale, y_scale)

    # Compute n_rows and n_cols
    min_col, min_row, max_col, max_row = get_roi_bounds(bounds=(min_x, min_y, max_x, max_y),
                                                        transform=combined_transform,
                                                        pixel_rounding='round',  # We do not expect fractional pixels here anyway
                                                        )
    assert min_col == 0 and min_row == 0, f"Expected min_col and min_row to be 0, but got {min_col} and {min_row}"
    height, width = max_row, max_col  # max values are exclusive
    return combined_transform, height, width


def get_roi_bounds(bounds: BaseGeometry | tuple[float, float, float, float],
                   transform: Affine | tuple[float, float, float, float, float, float],
                   buffer: int = 0,
                   pixel_rounding: Literal['tight', 'loose', 'round'] = 'loose',
                   clip_to_image: bool = False,
                   img_rows: int = None,
                   img_cols: int = None,
                   ) -> tuple[int, int, int, int]:
    """Get pixel bounds for the region of interest from polygon.

    Args:
        bounds: The bounds of the region of interest, either a shapely geometry, a BoundingBox object or a tuple of 4 floats, in the same CRS as the image.
            If a shapely geometry is provided, the bounds will be extracted from it.
            If tuple is provided, it expected to be as (min_x, min_y, max_x, max_y).
        transform: The transform of the image, either an Affine object or a gdal-style tuple of 6 elements.
        buffer: The number of pixels to add to the bounds.
        pixel_rounding: The rounding mode for the pixel bounds.
            'tight': Make sure that output bounds are fully within the input bounds.
            'loose': Make sure that output bounds fully cover the input bounds. Corresponds all_touched parameter in rasterio.mask.mask.
            'round': Round the bounds to the nearest integer. Corresponds to nearest interpolation.
        clip_to_image: Whether to clip the bounds to the image dimensions.
        img_cols: The number of columns in the image. Required if clip_to_image is True.
        img_rows: The number of rows in the image. Required if clip_to_image is True.

    Returns:
        (min_col, min_row, max_col, max_row) The pixel bounds of the polygon in the image. The max values are exclusive.
    """
    if isinstance(bounds, BaseGeometry):
        bounds = bounds.bounds

    if not isinstance(transform, Affine):
        transform = Affine.from_gdal(*transform)

    min_x_geo, min_y_geo, max_x_geo, max_y_geo = bounds

    # Figure out origin and opposite corner
    # Origin is the point that scale is directed from
    origin_x = min_x_geo if transform.a > 0 else max_x_geo
    origin_y = min_y_geo if transform.e > 0 else max_y_geo

    # Opposite corner is the point that scale points to
    opposite_x = max_x_geo if transform.a > 0 else min_x_geo
    opposite_y = max_y_geo if transform.e > 0 else min_y_geo

    # Convert origin and opposite corner to pixel coordinates
    min_col, min_row = ~transform * (origin_x, origin_y)
    max_col, max_row = ~transform * (opposite_x, opposite_y)
    assert max_col >= min_col and max_row >= min_row, f"The implementation ensures that this does not happen"

    # Convert to integers
    if pixel_rounding == 'tight':
        min_col, max_col = math.ceil(min_col), math.floor(max_col)
        min_row, max_row = math.ceil(min_row), math.floor(max_row)
    elif pixel_rounding == 'loose':
        min_col, max_col = math.floor(min_col), math.ceil(max_col)
        min_row, max_row = math.floor(min_row), math.ceil(max_row)
    elif pixel_rounding == 'round':
        min_col, max_col = round(min_col), round(max_col)
        min_row, max_row = round(min_row), round(max_row)
    else:
        raise ValueError(f"Invalid pixel_rounding: {pixel_rounding}")

    # Add buffer to the bounds.
    min_col, min_row = min_col - buffer, min_row - buffer
    max_col, max_row = max_col + buffer, max_row + buffer

    # If requested, clip the bounds to the image dimensions.
    if clip_to_image:
        assert img_cols is not None and img_rows is not None, "img_cols and img_rows must be provided if clip_to_image is True"
        min_col = np.clip(min_col, 0, img_cols)
        max_col = np.clip(max_col, 0, img_cols)
        min_row = np.clip(min_row, 0, img_rows)
        max_row = np.clip(max_row, 0, img_rows)

    return min_col, min_row, max_col, max_row


def zip_longest_recycle(*iterables,
                        allow_empty: bool = False,
                        fallback_value: Any = None):
    """
    Iterate over multiple iterables in parallel, stopping when the longest is exhausted
    while reusing the values of the shorter iterables.

    Like itertools.zip_longest, but when an iterable runs out of values, it begins
    reusing (cycling through) the values it has already produced instead of filling
    with a static value.

    Args:
        *iterables: One or more iterable objects to be zipped together.
        allow_empty: If False (default), iteration stops immediately if any iterable yields no values at all.
            If True, such an iterable contributes `fallback_value` for every step instead.
        fallback_value: The value to use for an iterable that is completely empty (i.e., yields no values) when `allow_empty` is True.

    Example:
        >>> a = [1, 2]
        >>> b = ['x', 'y', 'z', 'w']
        >>> for pair in zip_longest_recycle(a, b):
        ...     print(pair)
        (1, 'x')
        (2, 'y')
        (1, 'z')
        (2, 'w')

    Once all iterables have been exhausted at least once, iteration stops.

    Yields:
        Tuples containing one element from each iterable, where shorter iterables
        recycle their earlier values once exhausted.

    See Also:
        itertools.zip_longest
        itertools.cycle
    """
    iterators = [iter(it) for it in iterables]
    caches = [[] for _ in iterables]
    exhausted = [False] * len(iterables)

    while not all(exhausted):
        row = []
        for i, it in enumerate(iterators):
            if not exhausted[i]:
                try:
                    val = next(it)
                    caches[i].append(val)
                    row.append(val)
                except StopIteration:
                    exhausted[i] = True
                    if caches[i]:
                        iterators[i] = cycle(caches[i])
                        row.append(next(iterators[i]))
                    elif allow_empty:
                        iterators[i] = repeat(fallback_value)
                        row.append(fallback_value)
                    else:
                        return
            else:
                row.append(next(iterators[i]))
        if all(exhausted):
            return
        yield tuple(row)
