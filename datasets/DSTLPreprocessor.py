# Written by Luke Mason
# Date: 2020-06-03
# Purpose: To create a dataloader for the DSTL dataset.
import sys
import time
import csv
import datetime
import logging
import os
from itertools import islice
from pathlib import Path
from typing import Dict, Tuple, List
import sys
import cv2
import numpy as np
import rasterio
import shapely.affinity
import shapely.geometry
import shapely.wkt
import torch
from base import BaseDataSet, BaseDataLoader
from shapely.geometry import MultiPolygon
from utils import (array_3d_merge, FilterConfig3D, BandGroup,
                   mask_for_polygons,
                   palette, generate_unique_config_hash)
import seaborn as sns
import matplotlib.pyplot as plt
from utils import metric_indx
epsilon = sys.float_info.epsilon

class DSTLPreprocessor:

    def __init__(self,
                 data,
                 root,
                 training_band_groups: Tuple[BandGroup, BandGroup, BandGroup],
                 training_classes: List[int],
                 img_ref_scale: str,
                 patch_size: int,
                 overlap_pixels: float,
                 align_images: bool,
                 interpolation_method: int,
                 add_negative_class: bool,
                 save_dir: str,
                 name: str,
                 start_time: str,
                 train_indices: List[int] or None = None,
                 val_indices: List[int] or None = None
                 ):
        """Constructor, initialiser.

        Args:
            training_classes (List[int]): The class labels to train on.
            training_band_groups (Tuple[BandGroup, BandGroup, BandGroup]): The colour spectrum
            bands to train on.
            align_images (bool): Align the images.
            interpolation_method (int): The interpolation method to use.
        """
        if img_ref_scale not in ['RGB', 'P', 'M', 'A']:
            raise ValueError(f"Unknown image reference scale: {img_ref_scale}")

        if len(training_band_groups) != 3:
            raise ValueError("Band groups must be 3")

        for group in training_band_groups:
            if group.strategy is None and len(group.bands) != 1:
                raise ValueError("Number of bands in a group must be 1 if "
                                 "there is no merge strategy for the group")

        self.num_classes = len(training_classes)
        if add_negative_class:
            self.num_classes += 1

        path = os.path.join(name, start_time)
        self.run_dir = os.path.join(save_dir, "run_info", path)
        os.makedirs(self.run_dir, exist_ok=True)

        # Preprocessing
        self.train_indices = train_indices
        self.val_indices = val_indices

        self._train_files = []
        self._val_files = []
        self.train_files = []
        self.val_files = []

        # Attributes
        self.img_ref_scale = img_ref_scale
        self.training_classes = training_classes
        self.training_band_groups = training_band_groups
        self.patch_size = patch_size
        self.overlap_pixels = overlap_pixels
        self.align_images = align_images
        self.interpolation_method = interpolation_method
        self.add_negative_class = add_negative_class

        # Setup directories and paths
        self.root = root
        self.image_dir = os.path.join(self.root, 'sixteen_band/sixteen_band')
        self.cache_dir = os.path.join(self.root, 'cached')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        stage_1_dir = os.path.join(self.cache_dir, 'stage_1')
        if not os.path.exists(stage_1_dir):
            os.makedirs(stage_1_dir)

        stage_2_dir = os.path.join(self.cache_dir, 'stage_2')
        if not os.path.exists(stage_2_dir):
            os.makedirs(stage_2_dir)

        masks_dir = os.path.join(self.cache_dir, 'masks')
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        # Memory Cache
        self.images = dict()  # type Dict[str, np.ndarray]
        self.labels = dict()  # type Dict[str, np.ndarray]
        self._wkt_data = data  # type Dict[str, Dict[int, str]]
        self._x_max_y_min = None  # type Dict[str, Tuple[int, int]]

        # Logging
        self._setup_logging()

        # Load
        self._set_files()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Logs to file

        # Check if log directory exist, if not create it
        log_dir = os.path.join(self.root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H.log')
        handler = logging.FileHandler(os.path.join(log_dir, log_file_name),
                                      mode='a')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _set_files(self):
        step_size = self.patch_size - self.overlap_pixels
        self.files = []  # type: List[Tuple[np.ndarray, np.ndarray, str]]

        # Preprocess Images, ensure their existences.
        for [image_id, class_poly] in self._wkt_data:
            stage_1_file_path = self.get_stage_1_file_path(image_id)
            if not os.path.exists(stage_1_file_path + ".data.npy"):
                self._preprocess_image_stage_1(image_id, stage_1_file_path)

            # Save mask for image
            mask_path = self.get_mask_file_path(image_id)
            if not os.path.exists(mask_path + ".mask.npy"):
                self._save_mask((image_id, class_poly), stage_1_file_path, mask_path)

            stage_2_file_path = self.get_stage_2_file_path(image_id)
            if not os.path.exists(stage_2_file_path + ".data.npy"):
                # Saves image to file with 3 bands
                self._preprocess_image_stage_2(image_id, stage_1_file_path,stage_2_file_path)

        # Load Images
        for index, [image_id, _] in enumerate(self._wkt_data):
            # Load image with 3 bands
            image = np.load(Path(self.get_stage_2_file_path(image_id) +
                                 ".data.npy"),  allow_pickle=True)

            height, width = image.shape[1], image.shape[2]
            chunk_offsets = self._gen_chunk_offsets(width, height, step_size)

            # Load all masks for image
            all_class_mask = np.load(Path(self.get_mask_file_path(image_id) + ".mask.npy"),  allow_pickle=True)

            for c_index, (x, y) in enumerate(chunk_offsets):

                # Get the masks for the classes that we want to validate and train on.
                patch_y_mask = all_class_mask[self.training_classes, y:y + self.patch_size,
                               x:x + self.patch_size]

                # Add the mask for all where there is no mask
                if self.add_negative_class:
                    patch_y_mask = np.concatenate((patch_y_mask, np.expand_dims(
                        np.logical_not(np.any(patch_y_mask, axis=0)), axis=0)), axis=0)

                patch = image[:, y:y + self.patch_size, x:x + self.patch_size]

                # All files associated with image @ index are put into train
                # or val.
                if index in self.train_indices:
                    self.train_files.append((patch, patch_y_mask, image_id))
                elif index in self.val_indices:
                    self.val_files.append((patch, patch_y_mask, image_id))
                else:
                    raise ValueError(f"Index {index} not in train or val indexes")

                self.print_progress_bar(c_index + 1, len(chunk_offsets),
                                      prefix=f"Chunking Image {image_id}...")

            self.logger.info(f"\nTotal Data Loaded"
                             f" {(100 / len(self._wkt_data)) * (index + 1)}% "
                             f"...\n")

        self.logger.debug(f"Train LEN: {len(self.train_files)}")
        self.logger.debug(f"Val LEN: {len(self.val_files)}")
        self.logger.info(f"Total files: Train - {len(self.train_files)}, "
                         f"Val - {len(self.val_files)}")
        if len(self.train_files) == 0 or len(self.val_files) == 0:
            raise ValueError("No training files were found. Please check the "
                                "data and try again.")

        self.save_count_plot(self.train_files, "train")
        self.save_count_plot(self.val_files, "val")
        self.plot_pixel_area_percentages(self.train_files, "train")
        self.plot_pixel_area_percentages(self.val_files, "val")

    def save_count_plot(self, files, type: str):
        timestamp = int(time.time())

        # Create an array of values (you can replace this with your data)
        data = np.array([file[2] for file in files])
        np.savetxt(os.path.join(self.run_dir, "files.txt"), data, fmt="%s")
        value_counts = sns.countplot(data, palette="Set3", legend=False)

        # Customize the plot
        value_counts.set(xlabel="Object Class", ylabel="Count")
        plt.title("Count of Occurrences for Each Object Class")

        # Save the plot to a file (e.g., PNG)
        plt.savefig(os.path.join(self.run_dir, f"{type}_counts_"
                                               f"({timestamp}).png"))

    def plot_pixel_area_percentages(self, files, type: str):
        timestamp = int(time.time())

        # Auto balance the classes so that the negative class is not over represented.
        pixel_area_stats = np.zeros((self.num_classes,))

        # Get the pixel area statistics for each class and set it
        for file in files:
            (_, patch_y_mask, __) = file
            for i in range(self.num_classes):
                pixel_area_stats[i] += np.sum(patch_y_mask[i])

        pixel_area_perecentages = np.round((pixel_area_stats / np.sum(
            pixel_area_stats)) * 100, 2)

        classes = self.training_classes + [
            10] if self.add_negative_class else self.training_classes
        categories = [metric_indx[str(classes[i])] for i in
                      range(self.num_classes)]

        # Create a bar chart with Seaborn
        sns.barplot(x=categories, y=pixel_area_perecentages, palette="Set3", hue=categories,
                    legend=False)

        # Add labels and a title
        plt.xlabel("Object Class")
        plt.ylabel("Pixel Area Percentage")
        plt.title("Pixel Area Percentage for Each Object Class")

        # Save the plot to a file (e.g., PNG)
        plt.savefig(os.path.join(self.run_dir,
                                 f"pixel_area_percentages_({timestamp}).png"))

    def calculate_weights(self, masks):
        num_classes = masks.shape[1]
        total_pixels_labeled = np.sum(masks)
        all_patches_class_area_totals = np.sum(masks, axis=(0, 2, 3))

        ratios = all_patches_class_area_totals / total_pixels_labeled

        # How many labeled pixels in mask vs all labeled pixels
        ps = []
        for mask in masks:
            p = 0
            for i in range(num_classes):
                pixel_ratio = np.sum(mask[i]) / total_pixels_labeled + epsilon
                scale_value = (100 / ratios[i]) / (num_classes - 1)
                p += pixel_ratio * scale_value
            classes_ = p * ((100 / num_classes) / 100)
            ps.append(np.round(classes_ / 100, 3))
        return ps



        weights = []
        for mask in masks:
            # Calculate the sum of pixel values for each channel
            channel_sums = mask.sum(dim=(0, 1))

            # Calculate the balance score
            balance_score = (
                        1.0 - torch.abs(channel_sums[0] - channel_sums[1]) /
                        (channel_sums[0] + channel_sums[1] + 0.01))

            # Normalize the balance score to a weight between 0 and 1
            weight = balance_score.item()
            weights.append(weight)

        return weights

    def get_file_weights(self):
        # Calculate weights based on file masks
        return (
            self.calculate_weights(np.array([file[1] for file in self.train_files])),
            self.calculate_weights(np.array([file[1] for file in self.val_files]))
        )

    def get_files(self):
        return self.train_files, self.val_files

    def print_progress_bar(self, iteration, total, prefix='', suffix='',
                           length=50,
                           fill='█'):
        percent = ("{:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        sys.stdout.write(
            '\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix))
        sys.stdout.flush()

    def get_mask_file_path(self, image_id: str) -> str:
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "classes": self.training_classes,
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "masks", image_id)

    def get_stage_1_file_path(self, image_id):
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "align_images": self.align_images,
            "interpolation_method": self.interpolation_method,
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "stage_1", filename)

    def get_stage_2_file_path(self, image_id):
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "align_images": self.align_images,
            "interpolation_method": self.interpolation_method,
            "training_band_groups": [group.to_dict()
                                     for group in self.training_band_groups]
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "stage_2", filename)

    def _save_mask(self, img_class, file_path, mask_path):
        image = np.load(Path(file_path + ".data.npy"),  allow_pickle=True)
        im_size = image.shape[1:]
        class_to_polygons = self._load_polygons(img_class, im_size)
        # Load all classes as masks
        mask = np.array(
            [mask_for_polygons(im_size, class_to_polygons.get(str(cls + 1)))
             for cls in range(len(class_to_polygons.keys()))])

        np.save(Path(mask_path + ".mask.npy"), mask, allow_pickle=True)

    def _load_data(self, index: int):
        return self.files[index]

    def __getitem__(self, index):
        patch, patch_y_mask, image_id = self._load_data(index)
        if not self.val and self.augment:
            patch, patch_y_mask = self._augmentation(patch, patch_y_mask)

        patch_y_mask = torch.from_numpy(patch_y_mask.astype(np.bool_)).long()
        if self.return_id:
            return (self.normalize(torch.tensor(patch, dtype=torch.float32)),
                    patch_y_mask,
                    image_id)
        return self.normalize(torch.tensor(patch, dtype=torch.float32)), patch_y_mask

    def _gen_chunk_offsets(self, width: int, height: int, step_size: int) -> \
            List[Tuple[int, int]]:
        """
         Returns a list of (x, y) offsets corresponding to chunks of the image
         To account for the left over pixels it will generate a chunk a step back
         from the edge leftovers.
         :param height: The height of the image.
         :param step_size: The step size to use when generating the chunks.
         :return: A list of (x, y) offsets.
         """

        x_offsets = list(range(0, width - step_size + 1, step_size))
        y_offsets = list(range(0, height - step_size + 1, step_size))
        if width % step_size != 0:
            x_offsets.append(width - step_size)
        if height % step_size != 0 or height:
            y_offsets.append(height - step_size)

        # Erase the end offsets that are greater than the image size minus
        # the patch size.
        upper_limit = width - self.patch_size
        x_offsets = [x for x in x_offsets if x <= upper_limit]
        upper_limit = height - self.patch_size
        y_offsets = [y for y in y_offsets if y <= upper_limit]

        chunk_offsets = [(x, y) for x in x_offsets for y in y_offsets]

        return chunk_offsets

    def _load_polygons(self, img_class: Tuple[str, Dict[int, MultiPolygon]],
                       im_size:Tuple[int, int]) -> Dict[str, MultiPolygon]:
        """
        Load the polygons for the image id and scale them to the image size.
        :param im_id: The image id.
        :param im_size: The image size.
        :return: A dictionary of class type to polygon.
        """
        image_id, class_poly = img_class
        height, width = im_size
        self.logger.debug(f'Loading polygons for image: {image_id}')
        x_max, y_min = self._get_x_max_y_min(image_id)
        x_scaler, y_scaler = self._get_scalers(height, width, x_max, y_min)

        items_ = {
            str(poly_type): shapely.affinity.scale(shapely.wkt.loads(poly),
                                                   xfact=x_scaler,
                                                   yfact=y_scaler,
                                                   origin=(0, 0, 0)) for
            poly_type, poly in class_poly.items()}
        self.logger.debug(f'Loaded polygons for image: {image_id}')
        return items_

    def _get_scalers(self, h, w, x_max, y_min) -> Tuple[float,
    float]:
        """
        Get the scalers for the x and y axis as according to the DSTL
        competition documentation there is some scaling and preprocessing
        that needs to occur to correc the training data..... so annoying
        """
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        x_scaler = w_ / x_max
        y_scaler = h_ / y_min
        return x_scaler, y_scaler

    def _get_x_max_y_min(self, im_id: str) -> Tuple[float, float]:
        """
        Get the max x and y values from grid sizes file that is provided from the dataset in competition.
        According to the DSTL competition documentation there is some scaling and preprocessing
        that needs to occur to correct the training data..... so annoying
        :param im_id:
        :return:
        """
        if self._x_max_y_min is None:
            with open(os.path.join(self.root,
                                   'grid_sizes.csv/grid_sizes.csv')) as f:
                self._x_max_y_min = {im_id: (float(x), float(y))
                                     for im_id, x, y in
                                     islice(csv.reader(f), 1, None)}
        return self._x_max_y_min[im_id]

    def _scale_percentile(self, matrix: np.ndarray) -> np.ndarray:
        """ Fixes the pixel value range to 2%-98% original distribution of values.
        """
        w, h, d = matrix.shape
        matrix = matrix.reshape([w * h, d])
        # Get 2nd and 98th percentile
        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        return matrix.reshape([w, h, d]).clip(0, 1)

    def _preprocess_for_alignment(self, image: np.ndarray) -> np.ndarray:
        # attempts to remove single-dimensional entries
        image = np.squeeze(image)
        # checks if the shape of the image is 2D, indicating a grayscale image (single channel).
        if len(image.shape) == 2:
            # If the image is grayscale, it expands the dimensions to make it a 3D array (assuming a single channel).
            # This is done so that the image can be concatenated with the other bands.
            image = self._scale_percentile(np.expand_dims(image, 2))
        else:  # If the image is not grayscale (assumed to have 3 channels, indicating color), it converts it to grayscale.
            assert image.shape[2] == 3, image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def _preprocess_image_stage_1(self, image_id: str, file_path: str):

        key = lambda x: f'{image_id}_{x}'

        # Get paths
        rgb_path = os.path.join(self.root, 'three_band/', f'{image_id}.tif')
        p_path = os.path.join(self.image_dir, f"{key('P')}.tif")
        m_path = os.path.join(self.image_dir, f"{key('M')}.tif")
        a_path = os.path.join(self.image_dir, f"{key('A')}.tif")

        # Open streams
        im_rgb_src = rasterio.open(rgb_path, driver='GTiff', dtype=np.float32)
        im_p_src = rasterio.open(p_path, driver='GTiff', dtype=np.float32)
        im_m_src = rasterio.open(m_path, driver='GTiff', dtype=np.float32)
        im_a_src = rasterio.open(a_path, driver='GTiff', dtype=np.float32)

        # Load data from streams
        im_rgb = im_rgb_src.read()
        im_p = im_p_src.read()
        im_m = im_m_src.read()
        im_a = im_a_src.read()

        w, h = im_rgb.shape[1:]

        # TODO This has not been tested yet
        if self.align_images:
            im_p, _ = self._aligned(im_rgb, im_p, key=key('P'))
            im_m, aligned = self._aligned(im_rgb, im_m, im_m[:3, :, :],
                                          key=key('M'))
            im_ref = im_m[-1, :, :] if aligned else im_rgb[0, :, :]
            im_a, _ = self._aligned(im_ref, im_a, im_a[0, :, :], key=key('A'))

        # Get the reference image for scaling the other image bands to it.
        # Allows for experimenting with different reference band scales.
        if self.img_ref_scale == 'RGB':
            ref_img = im_rgb
        elif self.img_ref_scale == 'P':
            ref_img = im_p
        elif self.img_ref_scale == 'M':
            ref_img = im_m
        elif self.img_ref_scale == 'A':
            ref_img = im_a
        else:
            raise ValueError(f'Invalid reference file type: {self.reference_file_type}')

        # Resize the images to be the same size as RGB.
        # Sometimes panchromatic is a couple of pixels different to RGB
        if im_p.shape != ref_img.shape[1:]:
            im_p = cv2.resize(im_p.transpose([1, 2, 0]), (h, w),
                              interpolation=self.interpolation_method)
            im_p = np.expand_dims(im_p, 0)
        im_rgb = cv2.resize(im_rgb.transpose([1, 2, 0]), (h, w),
                            interpolation=self.interpolation_method).transpose([2,0,1])
        im_m = cv2.resize(im_m.transpose([1, 2, 0]), (h, w), interpolation=self.interpolation_method).transpose([2,0,1])
        im_a = cv2.resize(im_a.transpose([1, 2, 0]), (h, w), interpolation=self.interpolation_method).transpose([2,0,1])

        # Scale images between 0-1 based off their maximum and minimum bounds
        # P and M images are 11bit integers, A is 14bit integers, scale values to be
        # between 0-1 floats
        im_p = (im_p / 2047.0)
        im_rgb = (im_rgb / 2047.0)
        im_m = (im_m / 2047.0)
        im_a = (im_a / 16383.0)
        image = np.concatenate([im_p, im_rgb, im_m, im_a], axis=0)

        # Save images
        np.save(Path(file_path + ".data.npy"), image)
        self.logger.debug(f"Saving image to {file_path}.data.npy")

        # Close streams
        im_rgb_src.close()
        im_p_src.close()
        im_m_src.close()
        im_a_src.close()

        del im_rgb, im_p, im_m, im_a, im_rgb_src, im_p_src, im_m_src, im_a_src, \
            rgb_path, p_path, m_path, a_path

        return image

    def _preprocess_image_stage_2(self, image_id: str, src_path: str, dst_path: str):
        image = np.load(src_path + ".data.npy",  allow_pickle=True)
        image = np.array([
            array_3d_merge(image[group.bands - 1, :, :].squeeze(),
                           group.filter_config) if group.filter_config is not None else image[group.bands - 1, :, :].squeeze() if group.strategy is None else group.strategy(image[group.bands - 1, :, :].squeeze()) for group in self.training_band_groups
        ], dtype=np.float32)

        # Save images
        self.logger.debug(f"Saving image to {dst_path}.data.npy")
        np.save(Path(dst_path + ".data.npy"), image, allow_pickle=True)

    def _aligned(self, im_ref, im, im_to_align=None, key=None):
        w, h = im.shape[1:]
        im_ref = cv2.resize(im_ref, (h, w),
                            interpolation=self.interpolation_method)
        im_ref = self._preprocess_for_alignment(im_ref)
        if im_to_align is None:
            im_to_align = im
        im_to_align = self._preprocess_for_alignment(im_to_align)
        assert im_ref.shape[1:] == im_to_align.shape[1:]
        try:
            cc, warp_matrix = self._get_alignment(im_ref, im_to_align, key)
        except cv2.error as e:
            self.logger.info(f'Error getting alignment: {e}')
            return im, False
        else:
            im = cv2.warpAffine(im, warp_matrix, (h, w),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            im[im == 0] = np.mean(im)
            return im, True

    def _get_alignment(self, im_ref, im_to_align, key):
        self.logger.info(f'Getting alignment for {key}')
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)
        cc, warp_matrix = cv2.findTransformECC(
            im_ref, im_to_align, warp_matrix, warp_mode, criteria)

        matrix_str = str(warp_matrix).replace('\n', '')
        self.logger.info(
            f"Got alignment for {key} with cc {cc:.3f}: {matrix_str}")
        return cc, warp_matrix