from typing import Dict, Tuple, List
import numpy as np
from shapely.geometry import MultiPolygon
import json
import hashlib
import cv2

metric_indx = dict({
    "all": "All",
    "0": "Buildings",
    "1": "Misc",
    "2": "Road",
    "3": "Track",
    "4": "Trees",
    "5": "Crops",
    "6": "Waterway",
    "7": "Standing water",
    "8": "Vehicle Large",
    "9": "Vehicle Small",
    "10": "Negative"
})

def generate_unique_config_hash(config):

    # Convert to json
    config_json = json.dumps(config, sort_keys=True)

    # Compute the SHA-256 hash of the JSON string
    hash_object = hashlib.sha256(config_json.encode())
    unique_filename = hash_object.hexdigest()

    return unique_filename

class FilterConfig3D:
    def __init__(self, kernel: Tuple[int, int, int], stride: Tuple[int, int, int]):
        self.kernel = kernel
        self.stride = stride

    def to_dict(self):
        return {
            "kernel": self.kernel,
            "stride": self.stride
        }

class BandGroup:
    def __init__(self, bands: List[int], filter_config: FilterConfig3D = None,
                 strategy: str = None):
        self.bands = np.array(bands)
        self.filter_config = filter_config
        self.strategy_name = strategy

        if strategy == 'max':
            self.strategy = lambda x: np.max(x, axis=0 if self.filter_config is None else None)
        elif strategy == 'mean':
            self.strategy = lambda x: np.mean(x, axis=0 if self.filter_config is None else None)
        elif strategy == 'min':
            self.strategy = lambda x: np.min(x, axis=0 if self.filter_config is None else None)
        elif strategy == 'sum':
            self.strategy = lambda x: np.sum(x, axis=0 if self.filter_config is None else None)
        else:
            self.strategy = None
    def to_dict(self):
        filter_config = self.filter_config.to_dict() if self.filter_config else None
        band_group = {
            "bands": self.bands.tolist()
        }
        if self.filter_config is not None:
            band_group.update({"filter_config": filter_config})
        if self.strategy_name is not None:
            band_group.update({"strategy": self.strategy_name})
        return band_group

def array_3d_merge(arr, config: FilterConfig3D):
    kernel_shape = config.kernel
    stride = config.stride
    func = config.strategy

    # Get array shape and kernel shape
    arr_shape = arr.shape
    kernel_shape = np.array(kernel_shape)
    stride = np.array(stride)

    # Calculate output shape
    output_shape = ((arr_shape - kernel_shape) // stride) + 1

    # Initialize an array to store the results
    results = np.zeros(output_shape)

    # Iterate over the array with the specified stride
    for i in range(0, arr_shape[0] - kernel_shape[0] + 1, stride[0]):
        for j in range(0, arr_shape[1] - kernel_shape[1] + 1, stride[1]):
            for k in range(0, arr_shape[2] - kernel_shape[2] + 1, stride[2]):
                # Extract the subarray within the sliding window
                subarray = arr[i:i+kernel_shape[0], j:j+kernel_shape[1], k:k+kernel_shape[2]]

                # Apply the lambda function to the subarray and store the result
                results[i//stride[0], j//stride[1], k//stride[2]] = func(subarray)

    return results


def extract_mask_values_using_polygons(mask: np.ndarray,
                                       polygons: MultiPolygon):
    """ Return numpy mask for given polygons.
        polygons should already be converted to image coordinates.
        non values are given -1.
    """
    # Mark the values to extract with a 1.
    mark_value = 1
    marked_mask = np.zeros(mask.shape, dtype=np.int8)
    cv2.fillPoly(marked_mask, polygons, mark_value)

    # Extract the values from the main mask using the marked mask
    extracted_values_mask = np.full(marked_mask.shape, -1, dtype=np.float32)
    for index, element in np.ndenumerate(marked_mask):
        if element == mark_value:
            extracted_values_mask[index] = mask[index]
    return extracted_values_mask

def mask_for_polygons(
        im_size: Tuple[int, int], polygons: MultiPolygon) -> np.ndarray:
    """ Return numpy mask for given polygons.
    polygons should already be converted to image coordinates.
    """
    img_mask = np.zeros(im_size)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask
