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
from datasets import DSTLDataset

class DSTLLoader(BaseDataLoader):
    def __init__(
            self,
            files,
            weights,
            batch_size,
            num_workers=1,
            shuffle=False,
            flip=False,
            rotate=False,
            blur=False,
            augment=False,
            return_id=False
    ):
        # Scale the bands to be between 0 - 255
        # Min Max for Type P: [[0, 2047]]
        # Min Max for Type RGB: [[1, 2047], [157, 2047], [91, 2047]]
        # Min Max for Type M: [[156, 2047], [115, 2047], [87, 2047], [55, 2047], [1, 2047], [84, 2047], [160, 2047], [111, 2047]]
        # Min Max for Type A: [[671, 15562], [489, 16383], [434, 16383], [390, 16383], [1, 16383], [129, 16383], [186, 16383], [1, 16383]]
        # P is 11bit, RGB is 11bit, M is 11bit, A is 14bit

        dstl_data_path = os.environ.get('DSTL_DATA_PATH')
        self.MEAN = [0.219613, 0.219613, 0.219613]
        self.STD = [0.110741, 0.110741, 0.110741]

        self.dataset = DSTLDataset(
            files,
            root=dstl_data_path,
            mean=self.MEAN,
            std=self.STD,
            augment=augment,
            flip=flip,
            rotate=rotate,
            blur=blur,
            return_id=return_id
        )

        # Convert the image indexes into files indexes.
        super(DSTLLoader, self).__init__(self.dataset, batch_size, shuffle,
                                         num_workers, weights)

