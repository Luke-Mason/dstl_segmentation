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

class DSTLDataset(BaseDataSet):

    def __init__(self, files, **kwargs):
        self._files = files

        super(DSTLDataset, self).__init__(**kwargs)

    def _set_files(self):
        # It was already preprocessed.
        self.files = self._files

    def _load_data(self, index: int):
        return self.files[index]

    def __getitem__(self, index):
        patch, patch_y_mask, image_id = self._load_data(index)
        if self.augment:
            patch, patch_y_mask = self._augmentation(patch, patch_y_mask)

        patch_y_mask = torch.from_numpy(patch_y_mask.astype(np.bool_)).long()
        if self.return_id:
            return (self.normalize(torch.tensor(patch, dtype=torch.float32)),
                    patch_y_mask,
                    image_id)
        return self.normalize(torch.tensor(patch, dtype=torch.float32)), patch_y_mask
