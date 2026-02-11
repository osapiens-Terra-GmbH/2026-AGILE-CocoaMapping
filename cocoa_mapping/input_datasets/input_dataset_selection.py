from typing import Literal, Sequence

from cocoa_mapping.input_datasets.single_scene_datasets import Sentinel2InputDataset, AEFInputDataset
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.multi_scenes_datasets import Sentinel2MultiScenes
from cocoa_mapping.input_datasets.multi_scenes_datasets import AEFMultiScenes


INPUT_DATASET_REGISTRY: dict[str, dict[Literal['single', 'multi'], type[InputDataset]]] = {
    'sentinel_2': {
        'single': Sentinel2InputDataset,
        'multi': Sentinel2MultiScenes,
    },
    'aef': {
        'single': AEFInputDataset,
        'multi': AEFMultiScenes,
    },
}
"""Dictionary of input dataset types to their single and multiple scene classes."""


def get_input_dataset(image_paths: Sequence[str] | str,
                      image_type: Literal['sentinel_2', 'aef'],
                      dataset_type: Literal['hdf5', 'tif'],
                      dataset_selection: Literal['best', 'random'] = 'best',
                      n_scenes: int = 1,
                      min_coverage: float = 0.5,
                      **kwargs: dict
                      ) -> InputDataset:
    """Get the input dataset.

    Args:
        image_paths: The paths to the input files. 
            If single path (str), it will be used as a single scene input dataset. Otherwise, multiple scenes (list[str]).
        image_type: The type of the input files, either 'sentinel_2' or 'aef'.
        dataset_type: The type of the input files, either 'hdf5' or 'tif'.
        dataset_selection: If multiple scenes provided, the method to select the datasets, either 'best' or 'random'.
        n_scenes: If multiple scenes provided, the number of scenes to output for each patch. 
        min_coverage: If multiple scenes provided, the minimum coverage for the patch
        kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        The input dataset.
    """
    # Validate inputs
    if image_type not in INPUT_DATASET_REGISTRY:
        raise ValueError(f"Invalid image type: {image_type}")
    if image_type == 'aef' and dataset_type != 'tif':
        raise ValueError(f"Only TIF is supported for AEF")

    # Get the dataset
    single_scene = isinstance(image_paths, str)
    if single_scene:
        class_ = INPUT_DATASET_REGISTRY[image_type]['single']
        return class_(path=image_paths,
                      dataset_type=dataset_type,
                      **kwargs)
    else:
        class_ = INPUT_DATASET_REGISTRY[image_type]['multi']
        return class_(paths=image_paths,
                      dataset_type=dataset_type,
                      n_scenes=n_scenes,
                      dataset_selection=dataset_selection,
                      min_coverage=min_coverage,
                      **kwargs)
